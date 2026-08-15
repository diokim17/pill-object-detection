from __future__ import annotations

import json
import random
import re
import shutil
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import Counter


PathLike = Union[str, Path]
TargetDict = Dict[str, torch.Tensor]
MetadataList = List[Dict[str, Any]]


@dataclass(frozen=True)
class ParsedImageName:
    """이미지 파일명에서 파싱한 정보를 저장합니다.

    Attributes:
        combination_key:
            annotation 폴더 이름의 기준이 되는 문자열입니다.
            예: ``K-001900-016548-019607-029451``
        pill_ids:
            이미지에 포함된 알약 ID 목록입니다.
            예: ("001900", "016548", "019607", "029451")
        camera_angle:
            파일명에서 읽은 촬영 각도입니다. 파싱할 수 없으면 None입니다.
        suffix_tokens:
            첫 번째 언더스코어 뒤에 있는 나머지 토큰입니다.
    """

    combination_key: str
    pill_ids: Tuple[str, ...]
    camera_angle: Optional[int]
    suffix_tokens: Tuple[str, ...]
    is_copy_paste: bool = False
    is_additional_ts: bool = False


@dataclass
class CachedObject:
    """JSON에서 읽어 메모리에 저장하는 객체 단위 annotation입니다."""

    bbox_xyxy: Tuple[float, float, float, float]
    category_id: int
    area: float
    iscrowd: int
    annotation_id: Optional[int]
    ignore: int
    metadata: Dict[str, Any]


@dataclass
class CachedSample:
    """이미지 한 장에 해당하는 캐시 데이터입니다."""

    image_path: Path
    image_id: int
    parsed_name: ParsedImageName
    objects: List[CachedObject]
    width: Optional[int]
    height: Optional[int]


class PillDetectionDataset(Dataset):
    """알약 multi-class object detection용 PyTorch Dataset입니다.

    Args:
        root:
            데이터셋 루트 경로입니다.
        image_dir_name:
            root 내부 이미지 폴더 이름입니다.
        annotation_dir_name:
            root 내부 annotation 폴더 이름입니다.
        transforms:
            이미지와 bounding box를 함께 변환하는 함수입니다.

            Albumentations 스타일의 경우 다음과 같이 호출 가능한 객체여야 합니다.

            ``transforms(image=np_image, bboxes=boxes, labels=labels)``

            반환값은 최소한 ``image``와 ``bboxes``를 포함해야 합니다.
            ``labels``를 반환하면 변환 이후 label도 갱신합니다.

            일반 callable 스타일의 경우 ``transforms(image, target)`` 형태도
            지원합니다. 반환값은 ``(image, target)``이어야 합니다.
        image_extensions:
            데이터셋으로 수집할 이미지 확장자 목록입니다.
        label_offset:
            원본 category_id를 연속 label로 매핑할 때 시작 인덱스입니다.
            torchvision detection은 보통 1, YOLO는 보통 0을 사용합니다.
        strict:
            True이면 annotation 누락, 객체 수 불일치, 잘못된 JSON 등에 대해
            예외를 발생시킵니다. False이면 경고 후 가능한 샘플만 사용합니다.
        validate_image_size:
            True이면 JSON의 width/height와 실제 이미지 크기가 일치하는지
            __getitem__에서 확인합니다.
        include_raw_metadata:
            True이면 각 객체 metadata에 JSON의 images/categories/annotation
            원본 레코드를 추가로 보관합니다. 메모리 사용량이 증가할 수 있습니다.
        additional_ts_mapping_annotation_dir:
            추가 수집 TS의 ``category_id=1 / Drug``를 기존 Train 클래스 ID로
            변환할 때 참조할 기존 Train annotation 폴더입니다. 지정하지 않으면
            현재 annotation 폴더 안의 기존 Train JSON으로 매핑을 구성합니다.

    Returns from __getitem__:
        image:
            PIL.Image.Image, numpy.ndarray 또는 torch.Tensor입니다.
            transform 설정에 따라 타입이 달라집니다.
        target:
            torchvision detection 스타일 딕셔너리입니다.
            ``boxes``, ``labels``, ``image_id``, ``area``, ``iscrowd``를 포함합니다.
        metadata:
            이미지 및 객체 단위 부가정보를 담은 딕셔너리입니다.
            객체 정보는 ``metadata["objects"]``에 들어 있습니다.
    """

    _FILENAME_PATTERN = re.compile(
        r"^(?P<combo>K(?:-\d{6}){3,4})_(?P<suffix>.+)$"
    )
    _COPY_PASTE_FILENAME_PATTERN = re.compile(r"^copy_paste_\d+$")
    _ADDITIONAL_TS_FILENAME_PATTERN = re.compile(
        r"^(?P<drug>K-\d{6})_(?P<suffix>.+)$"
    )

    def __init__(
        self,
        root: PathLike,
        transforms: Optional[Callable[..., Any]] = None,
        image_dir_name: str = "train_images",
        annotation_dir_name: str = "train_annotations",
        image_extensions: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
        label_offset: int = 1,
        strict: bool = True,
        validate_image_size: bool = False,
        include_raw_metadata: bool = False,
        additional_ts_mapping_annotation_dir: Optional[PathLike] = None,
    ) -> None:
        super().__init__()

        if label_offset < 0:
            raise ValueError("label_offset은 0 이상의 정수여야 합니다.")

        self.root = Path(root)
        self.image_dir = self.root / image_dir_name
        self.annotation_dir = self.root / annotation_dir_name
        self.transforms = transforms
        self.image_extensions = tuple(ext.lower() for ext in image_extensions)
        self.label_offset = label_offset
        self.strict = strict
        self.validate_image_size = validate_image_size
        self.include_raw_metadata = include_raw_metadata
        self.additional_ts_mapping_annotation_dir = (
            Path(additional_ts_mapping_annotation_dir)
            if additional_ts_mapping_annotation_dir is not None
            else self.annotation_dir
        )

        self._validate_directories()

        # 추가 TS의 categories.id=1 / Drug는 실제 학습 클래스가 아니므로,
        # 기존 Train JSON만 사용해 식별자별 실제 category_id 매핑을 구성합니다.
        (
            self._additional_ts_category_maps,
            self._additional_ts_category_records,
        ) = self._build_additional_ts_category_mapping()

        # JSON을 정확히 한 번만 읽은 뒤 임시 CachedSample 목록을 생성합니다.
        raw_samples, category_records = self._build_cache_once()

        # COCO category_id는 1900, 16548처럼 비연속적일 수 있으므로
        # 학습에 적합한 연속 label로 매핑합니다.
        category_ids = sorted(category_records.keys())
        self.cat2label: Dict[int, int] = {
            category_id: index + self.label_offset
            for index, category_id in enumerate(category_ids)
        }
        self.label2cat: Dict[int, int] = {
            label: category_id for category_id, label in self.cat2label.items()
        }

        # category 이름 및 원본 category 레코드도 접근할 수 있게 보관합니다.
        self.category_id_to_name: Dict[int, str] = {
            category_id: str(record.get("name", category_id))
            for category_id, record in category_records.items()
        }
        self.category_records = category_records

        # 클래스 매핑이 생성된 뒤 최종 samples를 구성합니다.
        self.samples: List[Dict[str, Any]] = [
            self._finalize_sample(sample) for sample in raw_samples
        ]

    def _validate_directories(self) -> None:
        """필수 폴더 존재 여부를 확인합니다."""

        if not self.root.exists():
            raise FileNotFoundError(f"데이터셋 root가 존재하지 않습니다: {self.root}")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"이미지 폴더가 존재하지 않습니다: {self.image_dir}")
        if not self.annotation_dir.is_dir():
            raise FileNotFoundError(
                f"annotation 폴더가 존재하지 않습니다: {self.annotation_dir}"
            )

    @classmethod
    def parse_image_filename(cls, image_path: PathLike) -> ParsedImageName:
        """이미지 파일명을 파싱합니다.

        지원 예시:
            K-001900-016548-019607-029451_0_2_0_2_70_000_200.png
            K-001900-016548-019607_0_2_0_2_75_000_200.png

        촬영 각도는 suffix 토큰 중 5번째 값(index 4)을 우선 사용합니다.
        현재 설명된 파일 규칙에서는 70, 75, 90이 이 위치에 있습니다.
        """

        path = Path(image_path)
        match = cls._FILENAME_PATTERN.match(path.stem)
        if match is None:
            if cls._COPY_PASTE_FILENAME_PATTERN.fullmatch(path.stem):
                return ParsedImageName(
                    combination_key=path.stem,
                    pill_ids=(),
                    camera_angle=None,
                    suffix_tokens=(),
                    is_copy_paste=True,
                )
            additional_ts_match = cls._ADDITIONAL_TS_FILENAME_PATTERN.match(path.stem)
            if additional_ts_match is not None:
                combination_key = additional_ts_match.group("drug")
                suffix_tokens = tuple(additional_ts_match.group("suffix").split("_"))
                camera_angle: Optional[int] = None
                if len(suffix_tokens) > 4:
                    try:
                        camera_angle = int(suffix_tokens[4])
                    except ValueError:
                        camera_angle = None
                return ParsedImageName(
                    combination_key=combination_key,
                    pill_ids=(combination_key.split("-")[1],),
                    camera_angle=camera_angle,
                    suffix_tokens=suffix_tokens,
                    is_additional_ts=True,
                )
            raise ValueError(
                "지원하지 않는 이미지 파일명 형식입니다: "
                f"{path.name}"
            )

        combination_key = match.group("combo")
        suffix_tokens = tuple(match.group("suffix").split("_"))
        pill_ids = tuple(combination_key.split("-")[1:])

        camera_angle: Optional[int] = None
        if len(suffix_tokens) > 4:
            try:
                camera_angle = int(suffix_tokens[4])
            except ValueError:
                camera_angle = None

        return ParsedImageName(
            combination_key=combination_key,
            pill_ids=pill_ids,
            camera_angle=camera_angle,
            suffix_tokens=suffix_tokens,
        )

    def _iter_image_files(self) -> List[Path]:
        """지원 확장자를 가진 이미지 파일을 정렬해 반환합니다."""

        files = [
            path
            for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.image_extensions
        ]
        return sorted(files, key=lambda path: path.name)

    def _handle_problem(self, message: str, exception_type: type[Exception] = RuntimeError) -> None:
        """strict 설정에 따라 예외를 발생시키거나 경고만 출력합니다."""

        if self.strict:
            raise exception_type(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _find_json_files(self, image_path: Path, parsed: ParsedImageName) -> List[Path]:
        """이미지 한 장에 대응하는 객체별 JSON 파일들을 찾습니다.

        annotation 폴더 예시:
            train_annotations/K-001900-...-029451_json/

        하위의 각 K-{ID} 폴더에서 이미지 stem과 동일한 JSON 파일을 찾습니다.
        """

        if parsed.is_copy_paste:
            expected_name = f"{image_path.stem}.json"
            direct_path = self.annotation_dir / expected_name
            json_files = (
                [direct_path]
                if direct_path.is_file()
                else sorted(self.annotation_dir.rglob(expected_name))
            )
            if len(json_files) != 1:
                self._handle_problem(
                    f"합성 이미지 {image_path.name}의 동일 basename JSON은 "
                    f"정확히 1개여야 하지만 {len(json_files)}개입니다."
                )
            return json_files

        if parsed.is_additional_ts:
            expected_name = f"{image_path.stem}.json"
            direct_path = self.annotation_dir / expected_name
            json_files = (
                [direct_path]
                if direct_path.is_file()
                else sorted(self.annotation_dir.rglob(expected_name))
            )
            if len(json_files) != 1:
                self._handle_problem(
                    f"추가 TS 이미지 {image_path.name}의 동일 basename JSON은 "
                    f"정확히 1개여야 하지만 {len(json_files)}개입니다."
                )
            return json_files

        annotation_group_dir = self.annotation_dir / f"{parsed.combination_key}_json"
        if not annotation_group_dir.is_dir():
            self._handle_problem(
                f"annotation 그룹 폴더가 없습니다: {annotation_group_dir}",
                FileNotFoundError,
            )
            return []

        expected_name = f"{image_path.stem}.json"
        json_files = sorted(annotation_group_dir.rglob(expected_name))

        expected_count = len(parsed.pill_ids)
        if len(json_files) != expected_count:
            self._handle_problem(
                f"이미지 {image_path.name}의 알약 ID 수는 {expected_count}개지만 "
                f"대응 JSON은 {len(json_files)}개입니다."
            )

        return json_files

    def _build_cache_once(
        self,
    ) -> Tuple[List[CachedSample], Dict[int, Dict[str, Any]]]:
        """모든 JSON을 한 번만 읽어 전체 annotation 캐시를 생성합니다."""

        raw_samples: List[CachedSample] = []
        category_records: Dict[int, Dict[str, Any]] = dict(
            self._additional_ts_category_records
        )

        image_files = self._iter_image_files()
        if not image_files:
            self._handle_problem(f"이미지 폴더가 비어 있습니다: {self.image_dir}")

        for image_id, image_path in enumerate(image_files):
            try:
                parsed = self.parse_image_filename(image_path)
            except ValueError as exc:
                self._handle_problem(str(exc), ValueError)
                continue

            json_files = self._find_json_files(image_path, parsed)
            if not json_files:
                # strict=False일 때 annotation 없는 샘플은 제외합니다.
                continue

            objects: List[CachedObject] = []
            sample_width: Optional[int] = None
            sample_height: Optional[int] = None

            sample_image_id = image_id

            if parsed.is_copy_paste:
                json_path = json_files[0]
                try:
                    with json_path.open("r", encoding="utf-8") as file:
                        payload = json.load(file)
                    synthetic_objects, image_record, synthetic_categories = (
                        self._parse_copy_paste_json_payload(
                            payload=payload,
                            json_path=json_path,
                            image_path=image_path,
                        )
                    )
                except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
                    self._handle_problem(
                        f"JSON 파싱 실패: {json_path}\n원인: {exc}",
                        type(exc) if isinstance(exc, Exception) else RuntimeError,
                    )
                else:
                    objects.extend(synthetic_objects)
                    sample_width = self._optional_int(image_record.get("width"))
                    sample_height = self._optional_int(image_record.get("height"))
                    if image_record.get("id") not in (None, ""):
                        sample_image_id = int(image_record["id"])
                    for category_id, category_record in synthetic_categories.items():
                        category_records.setdefault(category_id, category_record)
            else:
                for json_path in json_files:
                    try:
                        with json_path.open("r", encoding="utf-8") as file:
                            payload = json.load(file)
                        cached_object, image_record, category_record = self._parse_json_payload(
                            payload=payload,
                            json_path=json_path,
                        )
                    except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
                        self._handle_problem(
                            f"JSON 파싱 실패: {json_path}\n원인: {exc}",
                            type(exc) if isinstance(exc, Exception) else RuntimeError,
                        )
                        continue

                    objects.append(cached_object)

                    width = image_record.get("width")
                    height = image_record.get("height")
                    if width is not None:
                        sample_width = int(width)
                    if height is not None:
                        sample_height = int(height)

                    if parsed.is_additional_ts:
                        mapped = self._resolve_additional_ts_category(
                            cached_object.metadata
                        )
                        if mapped is None:
                            self._handle_problem(
                                "추가 TS의 실제 Train category_id를 매핑할 수 없어 "
                                f"샘플을 제외합니다: {json_path}",
                                ValueError,
                            )
                            objects.pop()
                            continue
                        mapped_category_id, mapping_key = mapped
                        cached_object.metadata["source_category_id"] = cached_object.category_id
                        cached_object.metadata["source_category_name"] = category_record.get("name")
                        cached_object.metadata["category_mapping_key"] = mapping_key
                        cached_object.category_id = mapped_category_id
                        category_record = dict(
                            self._additional_ts_category_records[mapped_category_id]
                        )
                        cached_object.metadata["category_name"] = category_record.get("name")

                    category_id = cached_object.category_id
                    if category_record:
                        category_records.setdefault(category_id, category_record)
                    else:
                        category_records.setdefault(
                            category_id,
                            {
                                "id": category_id,
                                "name": cached_object.metadata.get("drug_name", str(category_id)),
                                "supercategory": "pill",
                            },
                        )

            if not objects:
                self._handle_problem(
                    f"유효한 annotation 객체가 없는 이미지입니다: {image_path.name}"
                )
                continue

            # 기존 Train에서만 파일명 ID와 JSON의 drug ID를 비교합니다.
            if not parsed.is_copy_paste and not parsed.is_additional_ts:
                json_pill_ids = {
                    self._normalize_pill_id(obj.metadata.get("drug_code")) for obj in objects
                }
                filename_pill_ids = set(parsed.pill_ids)
                if None not in json_pill_ids and json_pill_ids != filename_pill_ids:
                    self._handle_problem(
                        f"파일명 알약 ID와 JSON drug_N이 일치하지 않습니다. "
                        f"image={image_path.name}, filename={sorted(filename_pill_ids)}, "
                        f"json={sorted(json_pill_ids)}"
                    )

            raw_samples.append(
                CachedSample(
                    image_path=image_path,
                    image_id=sample_image_id,
                    parsed_name=parsed,
                    objects=objects,
                    width=sample_width,
                    height=sample_height,
                )
            )

        if not raw_samples:
            raise RuntimeError("사용 가능한 image/annotation 샘플을 찾지 못했습니다.")

        return raw_samples, category_records

    def _build_additional_ts_category_mapping(
        self,
    ) -> Tuple[Dict[str, Dict[str, int]], Dict[int, Dict[str, Any]]]:
        """기존 Train JSON에서 추가 TS 식별자 → 실제 category_id 맵을 만듭니다."""

        # 기존 Train 및 Copy-Paste 전용 데이터셋에서는 이 분기를 완전히 건너뛰어
        # 기존 클래스 수집 방식과 로딩 동작을 그대로 보존합니다.
        has_additional_ts = any(
            self._ADDITIONAL_TS_FILENAME_PATTERN.match(path.stem)
            for path in self._iter_image_files()
        )
        if not has_additional_ts:
            return (
                {key: {} for key in ("drug_N", "dl_mapping_code", "item_seq", "drug_name")},
                {},
            )

        mapping_sets: Dict[str, Dict[str, set[int]]] = {
            key: {} for key in ("drug_N", "dl_mapping_code", "item_seq", "drug_name")
        }
        category_records: Dict[int, Dict[str, Any]] = {}
        root = self.additional_ts_mapping_annotation_dir
        if not root.is_dir():
            raise FileNotFoundError(f"추가 TS 매핑용 Train annotation 폴더가 없습니다: {root}")

        for json_path in root.rglob("*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as file:
                    payload = json.load(file)
            except (OSError, json.JSONDecodeError):
                continue
            images = payload.get("images") or []
            annotations = payload.get("annotations") or []
            categories = payload.get("categories") or []
            if not images or not annotations:
                continue
            image_record = images[0]
            category_by_id = {
                int(record["id"]): dict(record)
                for record in categories
                if isinstance(record, Mapping) and record.get("id") is not None
            }
            for annotation in annotations:
                try:
                    category_id = int(annotation["category_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                category_record = category_by_id.get(category_id, {})
                if category_id == 1 and str(category_record.get("name", "")).strip() == "Drug":
                    continue
                category_records.setdefault(
                    category_id,
                    category_record or {
                        "id": category_id,
                        "name": image_record.get("dl_name", str(category_id)),
                        "supercategory": "pill",
                    },
                )
                values = {
                    "drug_N": self._normalize_mapping_code(image_record.get("drug_N")),
                    "dl_mapping_code": self._normalize_mapping_code(
                        image_record.get("dl_mapping_code")
                    ),
                    "item_seq": self._normalize_mapping_text(image_record.get("item_seq")),
                    "drug_name": self._normalize_mapping_text(
                        image_record.get("dl_name") or category_record.get("name")
                    ),
                }
                for key, value in values.items():
                    if value is not None:
                        mapping_sets[key].setdefault(value, set()).add(category_id)

        mappings: Dict[str, Dict[str, int]] = {}
        for key, records in mapping_sets.items():
            conflicts = {value: ids for value, ids in records.items() if len(ids) > 1}
            if conflicts:
                preview = list(conflicts.items())[:5]
                raise ValueError(f"추가 TS 매핑 키 충돌({key}): {preview}")
            mappings[key] = {value: next(iter(ids)) for value, ids in records.items()}
        return mappings, category_records

    def _resolve_additional_ts_category(
        self, metadata: Mapping[str, Any]
    ) -> Optional[Tuple[int, str]]:
        """우선순위에 따라 추가 TS 객체의 실제 Train category_id를 찾습니다."""

        candidates = (
            ("drug_N", self._normalize_mapping_code(metadata.get("drug_code"))),
            ("dl_mapping_code", self._normalize_mapping_code(metadata.get("mapping_code"))),
            ("item_seq", self._normalize_mapping_text(metadata.get("item_seq"))),
            ("drug_name", self._normalize_mapping_text(metadata.get("drug_name"))),
        )
        for key, value in candidates:
            if value is not None and value in self._additional_ts_category_maps[key]:
                return self._additional_ts_category_maps[key][value], key
        return None

    @staticmethod
    def _normalize_mapping_text(value: Any) -> Optional[str]:
        if value in (None, ""):
            return None
        text = str(value).strip()
        return text or None

    @classmethod
    def _normalize_mapping_code(cls, value: Any) -> Optional[str]:
        normalized = cls._normalize_pill_id(value)
        if normalized is None:
            return None
        return normalized.lstrip("0") or "0"

    def _parse_copy_paste_json_payload(
        self,
        payload: Mapping[str, Any],
        json_path: Path,
        image_path: Path,
    ) -> Tuple[List[CachedObject], Dict[str, Any], Dict[int, Dict[str, Any]]]:
        """합성 이미지 하나의 multi-object JSON을 전부 CachedObject로 변환합니다."""

        images = payload.get("images", [])
        annotations = payload.get("annotations", [])
        categories = payload.get("categories", [])

        if len(images) != 1:
            raise ValueError(f"합성 JSON의 images 항목은 정확히 1개여야 합니다: {len(images)}")
        if not annotations:
            raise ValueError("annotations 항목이 비어 있습니다.")

        image_record = dict(images[0])
        json_file_name = image_record.get("file_name") or image_record.get("imgfile")
        if json_file_name != image_path.name:
            raise ValueError(
                f"합성 image/JSON file_name 불일치: image={image_path.name}, json={json_file_name}"
            )

        categories_by_id: Dict[int, Dict[str, Any]] = {}
        for record in categories:
            category_record = dict(record)
            category_id = int(category_record["id"])
            if category_id in categories_by_id:
                raise ValueError(f"합성 JSON categories.id 중복: {category_id}")
            categories_by_id[category_id] = category_record

        image_id = self._optional_int(image_record.get("id"))
        objects: List[CachedObject] = []
        for annotation in annotations:
            annotation_record = dict(annotation)
            category_id = int(annotation_record["category_id"])
            if category_id not in categories_by_id:
                raise ValueError(
                    f"annotation category_id에 대응하는 categories 레코드가 없습니다: {category_id}"
                )

            annotation_image_id = self._optional_int(annotation_record.get("image_id"))
            if (
                image_id is not None
                and annotation_image_id is not None
                and annotation_image_id != image_id
            ):
                raise ValueError(
                    f"합성 JSON image_id 불일치: image={image_id}, annotation={annotation_image_id}"
                )

            bbox = annotation_record.get("bbox")
            if not isinstance(bbox, Sequence) or len(bbox) != 4:
                raise ValueError(f"bbox 형식이 올바르지 않습니다: {bbox}")
            x, y, width, height = map(float, bbox)
            if width <= 0 or height <= 0:
                raise ValueError(f"bbox의 width/height는 양수여야 합니다: {bbox}")

            area_value = annotation_record.get("area", width * height)
            area = float(area_value)
            if area <= 0:
                area = width * height

            category_record = categories_by_id[category_id]
            metadata = self._extract_metadata(
                image_record=image_record,
                annotation_record=annotation_record,
                category_record=category_record,
                json_path=json_path,
            )
            objects.append(
                CachedObject(
                    bbox_xyxy=(x, y, x + width, y + height),
                    category_id=category_id,
                    area=area,
                    iscrowd=int(annotation_record.get("iscrowd", 0)),
                    annotation_id=self._optional_int(annotation_record.get("id")),
                    ignore=int(annotation_record.get("ignore", 0)),
                    metadata=metadata,
                )
            )

        return objects, image_record, categories_by_id

    def _parse_json_payload(
        self,
        payload: Mapping[str, Any],
        json_path: Path,
    ) -> Tuple[CachedObject, Dict[str, Any], Dict[str, Any]]:
        """객체 하나에 해당하는 JSON payload를 CachedObject로 변환합니다."""

        images = payload.get("images", [])
        annotations = payload.get("annotations", [])
        categories = payload.get("categories", [])

        if not images:
            raise ValueError("images 항목이 비어 있습니다.")
        if not annotations:
            raise ValueError("annotations 항목이 비어 있습니다.")

        image_record = dict(images[0])
        annotation_record = dict(annotations[0])
        category_record = dict(categories[0]) if categories else {}

        bbox = annotation_record.get("bbox")
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            raise ValueError(f"bbox 형식이 올바르지 않습니다: {bbox}")

        x, y, width, height = map(float, bbox)
        if width <= 0 or height <= 0:
            raise ValueError(f"bbox의 width/height는 양수여야 합니다: {bbox}")

        x2 = x + width
        y2 = y + height
        category_id = int(annotation_record["category_id"])

        # 추가 데이터 일부는 실제 약 정보가 images에 있음에도 COCO category가
        # {id: 1, name: 'Drug'}라는 placeholder로 저장되어 있습니다.
        # 이 경우 drug_N과 dl_name을 사용해 실제 category 정보를 복원합니다.
        is_placeholder_category = (
            category_id == 1
            and self._optional_int(category_record.get("id")) == 1
            and str(category_record.get("name", "")).strip().lower() == "drug"
        )
        if is_placeholder_category:
            normalized_pill_id = self._normalize_pill_id(image_record.get("drug_N"))
            if normalized_pill_id is None or not normalized_pill_id.isdigit():
                raise ValueError(
                    "placeholder category를 복원할 drug_N이 올바르지 않습니다: "
                    f"{image_record.get('drug_N')}"
                )

            category_id = int(normalized_pill_id)
            drug_name = image_record.get("dl_name")
            if drug_name is None or not str(drug_name).strip():
                drug_name = f"K-{normalized_pill_id}"

            # 아래 로직 전체가 보정된 값을 사용하도록 복사본을 갱신합니다.
            annotation_record["category_id"] = category_id
            category_record = {
                **category_record,
                "id": category_id,
                "name": str(drug_name),
                "supercategory": category_record.get("supercategory", "pill"),
            }

        # area 값이 없거나 잘못된 경우 bbox 면적으로 보정합니다.
        area_value = annotation_record.get("area", width * height)
        area = float(area_value)
        if area <= 0:
            area = width * height

        metadata = self._extract_metadata(
            image_record=image_record,
            annotation_record=annotation_record,
            category_record=category_record,
            json_path=json_path,
        )

        return (
            CachedObject(
                bbox_xyxy=(x, y, x2, y2),
                category_id=category_id,
                area=area,
                iscrowd=int(annotation_record.get("iscrowd", 0)),
                annotation_id=self._optional_int(annotation_record.get("id")),
                ignore=int(annotation_record.get("ignore", 0)),
                metadata=metadata,
            ),
            image_record,
            category_record,
        )

    def _extract_metadata(
        self,
        image_record: Mapping[str, Any],
        annotation_record: Mapping[str, Any],
        category_record: Mapping[str, Any],
        json_path: Path,
    ) -> Dict[str, Any]:
        """JSON의 부가 정보를 객체별 metadata 딕셔너리로 정리합니다."""

        metadata: Dict[str, Any] = {
            # 파일/식별 정보
            "annotation_json_path": str(json_path),
            "annotation_id": annotation_record.get("id"),
            "source_image_id": image_record.get("id"),
            "source_image_file_name": image_record.get("file_name"),
            "drug_code": image_record.get("drug_N"),
            "mapping_code": image_record.get("dl_mapping_code"),
            "drug_index": image_record.get("dl_idx"),
            "item_seq": image_record.get("item_seq"),
            # 이름/성분/회사
            "drug_name": image_record.get("dl_name"),
            "drug_name_en": image_record.get("dl_name_en"),
            "material": image_record.get("dl_material"),
            "material_en": image_record.get("dl_material_en"),
            "company": image_record.get("dl_company"),
            "company_en": image_record.get("dl_company_en"),
            "manufacturer": image_record.get("di_company_mf"),
            "manufacturer_en": image_record.get("di_company_mf_en"),
            # 외형
            "drug_status": image_record.get("drug_S"),
            "custom_shape": image_record.get("dl_custom_shape"),
            "shape": image_record.get("drug_shape"),
            "color1": image_record.get("color_class1"),
            "color2": image_record.get("color_class2"),
            "chart": image_record.get("chart"),
            "form_code_name": image_record.get("form_code_name"),
            "length_long": image_record.get("leng_long"),
            "length_short": image_record.get("leng_short"),
            "thickness": image_record.get("thick"),
            # 각인/분할선
            "front_print": image_record.get("print_front"),
            "back_print": image_record.get("print_back"),
            "front_line": image_record.get("line_front"),
            "back_line": image_record.get("line_back"),
            "mark_code_front": image_record.get("mark_code_front"),
            "mark_code_back": image_record.get("mark_code_back"),
            "mark_code_front_analysis": image_record.get("mark_code_front_anal"),
            "mark_code_back_analysis": image_record.get("mark_code_back_anal"),
            # 촬영 환경
            "direction": image_record.get("drug_dir"),
            "background": image_record.get("back_color"),
            "light": image_record.get("light_color"),
            "camera_latitude": image_record.get("camera_la"),
            "camera_longitude": image_record.get("camera_lo"),
            "source_size": image_record.get("size"),
            # 허가/분류 정보
            "permit_date": image_record.get("di_item_permit_date"),
            "class_number": image_record.get("di_class_no"),
            "etc_otc_code": image_record.get("di_etc_otc_code"),
            "edi_code": image_record.get("di_edi_code"),
            "registration_timestamp": image_record.get("img_regist_ts"),
            "change_date": image_record.get("change_date"),
            # 외부 이미지 참조
            "image_url": image_record.get("img_key"),
            # COCO category 정보
            "category_name": category_record.get("name"),
            "supercategory": category_record.get("supercategory"),
        }

        if self.include_raw_metadata:
            metadata["raw_image_record"] = dict(image_record)
            metadata["raw_annotation_record"] = dict(annotation_record)
            metadata["raw_category_record"] = dict(category_record)

        return metadata

    def _finalize_sample(self, sample: CachedSample) -> Dict[str, Any]:
        """CachedSample을 __getitem__에서 바로 사용할 최종 구조로 변환합니다."""

        boxes: List[Tuple[float, float, float, float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []
        annotation_ids: List[int] = []
        ignore_flags: List[int] = []
        object_metadata: MetadataList = []

        for obj in sample.objects:
            label = self.cat2label[obj.category_id]

            boxes.append(obj.bbox_xyxy)
            labels.append(label)
            areas.append(obj.area)
            iscrowd.append(obj.iscrowd)
            annotation_ids.append(-1 if obj.annotation_id is None else obj.annotation_id)
            ignore_flags.append(obj.ignore)

            metadata = deepcopy(obj.metadata)
            metadata["category_id"] = obj.category_id
            metadata["class_index"] = label
            metadata["class_name"] = self.category_id_to_name.get(
                obj.category_id,
                str(obj.category_id),
            )
            object_metadata.append(metadata)

        target: TargetDict = {
            "boxes": self._boxes_tensor(boxes),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([sample.image_id], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
            # 아래 항목은 필수는 아니지만 COCO 원본과의 추적 및 분석에 유용합니다.
            "annotation_id": torch.tensor(annotation_ids, dtype=torch.int64),
            "ignore": torch.tensor(ignore_flags, dtype=torch.int64),
        }

        metadata: Dict[str, Any] = {
            "image_path": str(sample.image_path),
            "file_name": sample.image_path.name,
            "image_stem": sample.image_path.stem,
            "image_id": sample.image_id,
            "width": sample.width,
            "height": sample.height,
            "combination_key": sample.parsed_name.combination_key,
            "pill_ids": list(sample.parsed_name.pill_ids),
            "pill_codes": [f"K-{pill_id}" for pill_id in sample.parsed_name.pill_ids],
            "num_pills": len(sample.objects),
            "camera_angle_from_filename": sample.parsed_name.camera_angle,
            "filename_suffix_tokens": list(sample.parsed_name.suffix_tokens),
            "objects": object_metadata,
        }

        return {
            "image_path": sample.image_path,
            "target": target,
            "metadata": metadata,
        }

    @staticmethod
    def _boxes_tensor(boxes: Sequence[Sequence[float]]) -> torch.Tensor:
        """빈 객체도 항상 (N, 4) 형태가 되도록 boxes tensor를 만듭니다."""

        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32)
        return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        """None을 허용하면서 값을 int로 변환합니다."""

        if value is None or value == "":
            return None
        return int(value)

    @staticmethod
    def _normalize_pill_id(value: Any) -> Optional[str]:
        """K-001900 또는 001900을 파일명 형식의 6자리 ID로 정규화합니다."""

        if value is None:
            return None
        text = str(value).strip()
        if text.startswith("K-"):
            text = text[2:]
        return text.zfill(6) if text.isdigit() else text

    def __len__(self) -> int:
        """사용 가능한 이미지 샘플 수를 반환합니다."""

        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, TargetDict, Dict[str, Any]]:
        """이미지, COCO 스타일 target, metadata를 반환합니다."""

        sample = self.samples[index]
        image_path: Path = sample["image_path"]

        # 초기화 시 JSON은 이미 모두 캐시했으므로, 여기서는 이미지만 디스크에서 읽습니다.
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")

        target: TargetDict = {
            key: value.clone() for key, value in sample["target"].items()
        }
        metadata: Dict[str, Any] = deepcopy(sample["metadata"])

        if self.validate_image_size:
            actual_width, actual_height = image.size
            expected_width = metadata.get("width")
            expected_height = metadata.get("height")
            if (
                expected_width is not None
                and expected_height is not None
                and (actual_width, actual_height) != (expected_width, expected_height)
            ):
                raise ValueError(
                    f"이미지 크기 불일치: {image_path.name}, "
                    f"actual={(actual_width, actual_height)}, "
                    f"json={(expected_width, expected_height)}"
                )

        if self.transforms is not None:
            image, target, metadata = self._apply_transforms(image, target, metadata)

        return image, target, metadata

    def _apply_transforms(
        self,
        image: Image.Image,
        target: TargetDict,
        metadata: Dict[str, Any],
    ) -> Tuple[Any, TargetDict, Dict[str, Any]]:
        """Albumentations 또는 일반 detection transform을 적용합니다."""

        boxes_before = target["boxes"]
        labels_before = target["labels"]

        # 먼저 Albumentations 스타일을 시도합니다.
        try:
            transformed = self.transforms(
                image=np.asarray(image),
                bboxes=boxes_before.cpu().numpy().tolist(),
                labels=labels_before.cpu().numpy().tolist(),
            )
        except TypeError:
            # Albumentations 스타일이 아니면 torchvision 사용자 정의 transform처럼
            # (image, target) -> (image, target) 호출을 시도합니다.
            transformed = self.transforms(image, target)
            if not isinstance(transformed, tuple) or len(transformed) != 2:
                raise TypeError(
                    "일반 transform은 (image, target) 형태의 2개 값을 반환해야 합니다."
                )
            transformed_image, transformed_target = transformed
            return transformed_image, transformed_target, metadata

        if not isinstance(transformed, Mapping):
            raise TypeError("Albumentations 스타일 transform은 dict를 반환해야 합니다.")
        if "image" not in transformed or "bboxes" not in transformed:
            raise KeyError("transform 반환값에는 image와 bboxes가 필요합니다.")

        transformed_image = transformed["image"]
        transformed_boxes = self._boxes_tensor(transformed["bboxes"])
        transformed_labels = torch.as_tensor(
            transformed.get("labels", labels_before.tolist()),
            dtype=torch.int64,
        )

        # Albumentations에서 min_visibility 등으로 bbox가 제거될 수 있습니다.
        # labels뿐 아니라 area, iscrowd, annotation_id, ignore, metadata도 같은 인덱스로
        # 맞춰야 합니다. 가장 안전한 방법은 object_indices를 추가 label_field로
        # 전달하는 것이지만, 기존 transform과의 호환을 위해 label 기반 매칭을 제공합니다.
        # 동일 클래스 객체가 여러 개 존재할 수 있으므로 bbox 개수가 달라진 경우에는
        # metadata 정렬을 완벽히 보장할 수 없습니다. 그런 경우 아래 경고를 출력합니다.
        if len(transformed_boxes) != len(boxes_before):
            warnings.warn(
                "transform 과정에서 bbox 개수가 변경되었습니다. "
                "target의 area/iscrowd 및 metadata 객체 목록은 bbox 개수에 맞춰 앞에서부터 "
                "잘립니다. 정확한 객체 추적이 필요하면 Albumentations label_fields에 "
                "object_indices를 함께 사용하도록 transform wrapper를 구성하십시오.",
                RuntimeWarning,
                stacklevel=2,
            )

        kept_count = len(transformed_boxes)
        target["boxes"] = transformed_boxes
        target["labels"] = transformed_labels.reshape(-1)

        # resize/rotation 이후에는 기존 area가 유효하지 않으므로 새 bbox 면적으로 계산합니다.
        if kept_count > 0:
            widths = (transformed_boxes[:, 2] - transformed_boxes[:, 0]).clamp(min=0)
            heights = (transformed_boxes[:, 3] - transformed_boxes[:, 1]).clamp(min=0)
            target["area"] = widths * heights
        else:
            target["area"] = torch.zeros((0,), dtype=torch.float32)

        for key in ("iscrowd", "annotation_id", "ignore"):
            target[key] = target[key][:kept_count]

        metadata["objects"] = metadata["objects"][:kept_count]
        metadata["num_pills_after_transform"] = kept_count

        return transformed_image, target, metadata

    @property
    def num_classes(self) -> int:
        """실제 알약 클래스 수를 반환합니다. background는 포함하지 않습니다."""

        return len(self.cat2label)

    def get_class_name(self, label: int) -> str:
        """학습용 label에 대응하는 알약 이름을 반환합니다."""

        if label not in self.label2cat:
            raise KeyError(f"존재하지 않는 label입니다: {label}")
        category_id = self.label2cat[label]
        return self.category_id_to_name[category_id]

    def get_category_id(self, label: int) -> int:
        """학습용 label을 원본 COCO category_id로 변환합니다."""

        if label not in self.label2cat:
            raise KeyError(f"존재하지 않는 label입니다: {label}")
        return self.label2cat[label]

    def get_label(self, category_id: int) -> int:
        """원본 COCO category_id를 학습용 연속 label로 변환합니다."""

        if category_id not in self.cat2label:
            raise KeyError(f"존재하지 않는 category_id입니다: {category_id}")
        return self.cat2label[category_id]

    def get_sample_summary(self, index: int) -> Dict[str, Any]:
        """이미지를 읽지 않고 샘플의 요약 정보만 반환합니다."""

        sample = self.samples[index]
        target = sample["target"]
        metadata = sample["metadata"]
        return {
            "index": index,
            "file_name": metadata["file_name"],
            "pill_ids": list(metadata["pill_ids"]),
            "num_pills": int(target["boxes"].shape[0]),
            "labels": target["labels"].tolist(),
            "category_ids": [
                self.label2cat[label] for label in target["labels"].tolist()
            ],
            "camera_angle": metadata["camera_angle_from_filename"],
        }

    def print_class_statistics(self) -> None:
        """모든 클래스의 class ID, pill ID, 약 이름, 객체 개수를 출력합니다."""

        instance_counts: Counter[int] = Counter()

        for sample in self.samples:
            instance_counts.update(sample["target"]["labels"].tolist())

        header = (
            f"{'class id':>10}  "
            f"{'pill id':>12}  "
            f"{'drug name':<40}  "
            f"{'data count':>12}"
        )

        print(header)
        print("-" * len(header))

        for class_id in sorted(self.label2cat):
            category_id = self.label2cat[class_id]
            pill_id = f"K-{category_id:06d}"
            drug_name = self.category_id_to_name.get(
                category_id,
                "Unknown",
            )
            data_count = instance_counts[class_id]

            print(
                f"{class_id:>10}  "
                f"{pill_id:>12}  "
                f"{drug_name:<40}  "
                f"{data_count:>12}"
            )

        print("-" * len(header))
        print(f"클래스 수: {self.num_classes}")
        print(f"전체 객체 수: {sum(instance_counts.values())}")


def detection_collate_fn(
    batch: Sequence[Tuple[Any, TargetDict, Dict[str, Any]]]
) -> Tuple[List[Any], List[TargetDict], List[Dict[str, Any]]]:
    """객체 탐지 DataLoader용 collate 함수입니다.

    이미지마다 객체 수가 3개 또는 4개로 다르므로 기본 collate 함수처럼
    target을 하나의 tensor로 stack할 수 없습니다. 따라서 이미지, target,
    metadata를 각각 list 형태로 묶어 반환합니다.
    """

    images, targets, metadata = zip(*batch)
    return list(images), list(targets), list(metadata)


def xyxy_to_yolo(
    box: Sequence[float], image_width: int, image_height: int
) -> Optional[Tuple[float, float, float, float]]:
    """픽셀 ``xyxy`` bbox를 YOLO의 정규화된 ``xywh``로 변환합니다."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width와 image_height는 양수여야 합니다.")

    x1, y1, x2, y2 = map(float, box)
    x1 = max(0.0, min(x1, float(image_width)))
    y1 = max(0.0, min(y1, float(image_height)))
    x2 = max(0.0, min(x2, float(image_width)))
    y2 = max(0.0, min(y2, float(image_height)))

    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None

    return (
        ((x1 + x2) / 2.0) / image_width,
        ((y1 + y2) / 2.0) / image_height,
        width / image_width,
        height / image_height,
    )


def split_indices_by_combination_key(
    dataset: PillDetectionDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """같은 알약 조합이 서로 다른 split에 섞이지 않도록 인덱스를 나눕니다."""

    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("train_ratio, val_ratio, test_ratio는 0 이상이고 합이 1이어야 합니다.")

    grouped: Dict[str, List[int]] = {}
    for index, sample in enumerate(dataset.samples):
        key = str(sample["metadata"]["combination_key"])
        grouped.setdefault(key, []).append(index)

    keys = sorted(grouped)
    random.Random(seed).shuffle(keys)
    if len(keys) < sum(ratio > 0 for ratio in ratios):
        raise ValueError("0보다 큰 각 split에 하나씩 배정할 만큼 combination_key가 충분하지 않습니다.")

    # 그룹 수를 기준으로 경계를 정해 combination_key 누수를 원천 차단합니다.
    train_end = round(len(keys) * train_ratio)
    val_end = train_end + round(len(keys) * val_ratio)
    train_end = min(max(train_end, int(train_ratio > 0)), len(keys))
    val_end = min(max(val_end, train_end + int(val_ratio > 0)), len(keys))

    key_splits = {
        "train": keys[:train_end],
        "val": keys[train_end:val_end],
        "test": keys[val_end:],
    }
    if test_ratio > 0 and not key_splits["test"]:
        donor = "val" if len(key_splits["val"]) > 1 else "train"
        key_splits["test"].append(key_splits[donor].pop())

    return {
        split: sorted(index for key in split_keys for index in grouped[key])
        for split, split_keys in key_splits.items()
    }


def prepare_ultralytics_dataset(
    root: PathLike,
    output_dir: PathLike,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    image_dir_name: str = "train_images",
    annotation_dir_name: str = "train_annotations",
    strict: bool = False,
) -> Dict[str, Any]:
    """원본 데이터셋을 분할하고 Ultralytics YOLO 폴더와 ``data.yaml``을 생성합니다.

    출력 폴더에 기존 이미지/라벨이 있으면 결과 혼합을 막기 위해 실패합니다.
    반환값에는 dataset, split별 indices/통계, data.yaml 경로가 포함됩니다.
    """

    dataset = PillDetectionDataset(
        root=root,
        image_dir_name=image_dir_name,
        annotation_dir_name=annotation_dir_name,
        label_offset=0,
        strict=strict,
        validate_image_size=True,
    )

    return prepare_ultralytics_dataset_from_dataset(
        dataset=dataset,
        output_dir=output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )


def prepare_ultralytics_dataset_from_dataset(
    dataset: PillDetectionDataset,
    output_dir: PathLike,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """기생성된 Dataset을 Ultralytics YOLO 폴더 형식으로 변환합니다.

    ``dataset.label_offset`` 값과 관계없이 YOLO 클래스 ID는 항상
    ``0``부터 ``num_classes - 1``까지 기록합니다. 출력 폴더에 기존
    이미지/라벨이 있으면 결과 혼합을 막기 위해 실패합니다.

    Args:
        dataset:
            변환할 :class:`PillDetectionDataset` 인스턴스입니다.
        output_dir:
            ``images/{train,val,test}``, ``labels/{train,val,test}``,
            ``data.yaml``을 생성할 출력 경로입니다.
        train_ratio, val_ratio, test_ratio:
            combination_key 그룹을 나눌 split 비율입니다. 합은 1이어야 합니다.
        seed:
            split 재현성을 위한 난수 seed입니다.
    """

    if not isinstance(dataset, PillDetectionDataset):
        raise TypeError("dataset은 PillDetectionDataset 인스턴스여야 합니다.")

    split_indices = split_indices_by_combination_key(
        dataset, train_ratio, val_ratio, test_ratio, seed
    )
    output_path = Path(output_dir).resolve()
    directories = {
        split: {
            "images": output_path / "images" / split,
            "labels": output_path / "labels" / split,
        }
        for split in ("train", "val", "test")
    }
    for split_dirs in directories.values():
        for directory in split_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
            if any(directory.iterdir()):
                raise FileExistsError(f"출력 폴더가 비어 있지 않습니다: {directory}")

    statistics: Dict[str, Dict[str, int]] = {}
    for split, indices in split_indices.items():
        object_count = 0
        skipped_count = 0
        for index in indices:
            sample = dataset.samples[index]
            source = Path(sample["image_path"])
            target = sample["target"]
            with Image.open(source) as image:
                image_width, image_height = image.size

            lines: List[str] = []
            for box, label in zip(target["boxes"].tolist(), target["labels"].tolist()):
                converted = xyxy_to_yolo(box, image_width, image_height)
                if converted is None:
                    skipped_count += 1
                    continue
                yolo_label = int(label) - dataset.label_offset
                if not 0 <= yolo_label < dataset.num_classes:
                    raise ValueError(
                        "YOLO 클래스 ID 변환 범위 오류: "
                        f"label={label}, label_offset={dataset.label_offset}"
                    )
                lines.append(f"{yolo_label} " + " ".join(f"{value:.6f}" for value in converted))
                object_count += 1

            shutil.copy2(source, directories[split]["images"] / source.name)
            (directories[split]["labels"] / f"{source.stem}.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )

        statistics[split] = {
            "images": len(indices), "objects": object_count, "skipped_boxes": skipped_count
        }

    names = [
        dataset.get_class_name(class_id + dataset.label_offset)
        for class_id in range(dataset.num_classes)
    ]
    yaml_lines = [
        f"path: {json.dumps(str(output_path), ensure_ascii=False)}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {dataset.num_classes}",
        "names:",
        *(f"  {index}: {json.dumps(name, ensure_ascii=False)}" for index, name in enumerate(names)),
    ]
    yaml_path = output_path / "data.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    group_sets = {
        split: {dataset.samples[index]["metadata"]["combination_key"] for index in indices}
        for split, indices in split_indices.items()
    }
    if not (group_sets["train"].isdisjoint(group_sets["val"])
            and group_sets["train"].isdisjoint(group_sets["test"])
            and group_sets["val"].isdisjoint(group_sets["test"])):
        raise RuntimeError("combination_key가 split 사이에 중복되었습니다.")

    return {
        "dataset": dataset,
        "output_dir": output_path,
        "yaml_path": yaml_path,
        "split_indices": split_indices,
        "statistics": statistics,
    }
