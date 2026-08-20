"""
yolo_dataset.py

PillDetectionDataset을 이용하여 원본 알약 데이터셋을
Ultralytics YOLO 형식으로 변환합니다.

역할
----
1. PillDetectionDataset으로 원본 이미지 / annotation 로드
2. split_utils.make_group_split()으로 train / val / test 분할
3. 원본 xyxy bbox를 YOLO normalized xywh 형식으로 변환
4. images/{train,val,test}, labels/{train,val,test} 생성
5. data.yaml 생성

Faster R-CNN과 동일한 Dataset / split 규칙을 사용하기 위해
Dataset 및 split 로직은 이 파일에서 별도로 구현하지 않습니다.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Union

import yaml
from PIL import Image

from src.PillDetectionDataset import PillDetectionDataset

from src.utils import (
    build_group_mapping,
    make_group_split,
    groups_to_indices,
    validate_split,
)


PathLike = Union[str, Path]


# ============================================================
# YOLO bbox 변환
# ============================================================

def xyxy_to_yolo(
    box: Sequence[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    xyxy bbox를 YOLO normalized xywh 형식으로 변환합니다.

    Args:
        box:
            [x1, y1, x2, y2]

        image_width:
            원본 이미지 width

        image_height:
            원본 이미지 height

    Returns:
        (x_center, y_center, width, height)
        모든 값은 0~1 범위로 정규화됩니다.
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"잘못된 이미지 크기입니다: "
            f"width={image_width}, height={image_height}"
        )

    x1, y1, x2, y2 = map(float, box)

    # 이미지 범위 밖 bbox 방지
    x1 = max(0.0, min(x1, image_width))
    y1 = max(0.0, min(y1, image_height))
    x2 = max(0.0, min(x2, image_width))
    y2 = max(0.0, min(y2, image_height))

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    if bbox_width <= 0 or bbox_height <= 0:
        raise ValueError(
            f"유효하지 않은 bbox입니다: "
            f"{[x1, y1, x2, y2]}"
        )

    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height

    normalized_width = bbox_width / image_width
    normalized_height = bbox_height / image_height

    return (
        x_center,
        y_center,
        normalized_width,
        normalized_height,
    )


# ============================================================
# 출력 디렉터리 생성
# ============================================================

def _prepare_output_directories(
    output_root: Path,
    clean: bool = True,
) -> None:
    """
    YOLO 데이터셋 출력 폴더를 준비합니다.

    clean=True이면 기존 images / labels를 제거한 뒤 다시 생성합니다.
    이전 실험 파일이 남아 데이터가 섞이는 것을 방지합니다.
    """

    images_root = output_root / "images"
    labels_root = output_root / "labels"

    if clean:
        if images_root.exists():
            shutil.rmtree(images_root)

        if labels_root.exists():
            shutil.rmtree(labels_root)

        data_yaml = output_root / "data.yaml"

        if data_yaml.exists():
            data_yaml.unlink()

    for split_name in ("train", "val", "test"):
        (images_root / split_name).mkdir(
            parents=True,
            exist_ok=True,
        )

        (labels_root / split_name).mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# 이미지 크기 확인
# ============================================================

def _get_image_size(
    image_path: Path,
    metadata: Dict,
) -> tuple[int, int]:
    """
    metadata에 width / height가 있으면 사용하고,
    없으면 실제 이미지에서 읽습니다.
    """

    width = metadata.get("width")
    height = metadata.get("height")

    if width is not None and height is not None:
        width = int(width)
        height = int(height)

        if width > 0 and height > 0:
            return width, height

    with Image.open(image_path) as image:
        return image.size


# ============================================================
# YOLO label 생성
# ============================================================

def _write_yolo_label(
    dataset: PillDetectionDataset,
    sample_index: int,
    label_path: Path,
) -> int:
    """
    Dataset sample 하나를 YOLO txt label로 저장합니다.

    PillDetectionDataset의 label이 1부터 시작하는 경우
    YOLO class index는 0부터 시작해야 하므로
    dataset.label_offset만큼 빼서 저장합니다.

    Returns:
        저장한 객체 수
    """

    sample = dataset.samples[sample_index]

    image_path = Path(sample["image_path"])
    target = sample["target"]
    metadata = sample["metadata"]

    image_width, image_height = _get_image_size(
        image_path=image_path,
        metadata=metadata,
    )

    boxes = target["boxes"].tolist()
    labels = target["labels"].tolist()

    if len(boxes) != len(labels):
        raise ValueError(
            f"bbox / label 개수가 다릅니다: "
            f"{image_path.name}, "
            f"boxes={len(boxes)}, labels={len(labels)}"
        )

    lines: List[str] = []

    for box, label in zip(boxes, labels):
        # Faster R-CNN Dataset label
        # 1 ~ N
        label = int(label)

        # YOLO class index
        # 0 ~ N-1
        yolo_class_id = label - dataset.label_offset

        if not 0 <= yolo_class_id < dataset.num_classes:
            raise ValueError(
                f"YOLO class id 범위 오류: "
                f"label={label}, "
                f"yolo_class_id={yolo_class_id}"
            )

        (
            x_center,
            y_center,
            bbox_width,
            bbox_height,
        ) = xyxy_to_yolo(
            box=box,
            image_width=image_width,
            image_height=image_height,
        )

        lines.append(
            f"{yolo_class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{bbox_width:.6f} "
            f"{bbox_height:.6f}"
        )

    label_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return len(lines)


# ============================================================
# split 하나 export
# ============================================================

def _export_split(
    dataset: PillDetectionDataset,
    indices: Sequence[int],
    output_root: Path,
    split_name: str,
) -> Dict[str, int]:
    """
    train / val / test 중 하나의 split을 YOLO 형식으로 저장합니다.
    """

    image_output_dir = (
        output_root
        / "images"
        / split_name
    )

    label_output_dir = (
        output_root
        / "labels"
        / split_name
    )

    image_count = 0
    object_count = 0

    for sample_index in indices:
        sample = dataset.samples[sample_index]

        source_image_path = Path(
            sample["image_path"]
        )

        destination_image_path = (
            image_output_dir
            / source_image_path.name
        )

        destination_label_path = (
            label_output_dir
            / f"{source_image_path.stem}.txt"
        )

        # 원본 이미지 복사
        shutil.copy2(
            source_image_path,
            destination_image_path,
        )

        # YOLO label 생성
        num_objects = _write_yolo_label(
            dataset=dataset,
            sample_index=sample_index,
            label_path=destination_label_path,
        )

        image_count += 1
        object_count += num_objects

    return {
        "images": image_count,
        "objects": object_count,
    }


# ============================================================
# data.yaml 생성
# ============================================================

def _create_data_yaml(
    dataset: PillDetectionDataset,
    output_root: Path,
) -> Path:
    """
    Ultralytics YOLO용 data.yaml을 생성합니다.
    """

    # YOLO class index → class name
    names: Dict[int, str] = {}

    for label in sorted(dataset.label2cat):
        yolo_class_id = (
            int(label)
            - dataset.label_offset
        )

        names[yolo_class_id] = (
            dataset.get_class_name(label)
        )

    data = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": dataset.num_classes,
        "names": names,
    }

    data_yaml_path = (
        output_root
        / "data.yaml"
    )

    with data_yaml_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            data,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    return data_yaml_path


# ============================================================
# YOLO Dataset 생성
# ============================================================

def build_yolo_dataset(
    dataset_root: PathLike,
    output_root: PathLike,
    image_dir_name: str = "train_images",
    annotation_dir_name: str = "train_annotations",
    label_offset: int = 1,
    strict: bool = False,
    validate_image_size: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    clean: bool = True,
) -> Path:
    """
    원본 알약 데이터셋을 YOLO 형식으로 변환합니다.

    Faster R-CNN과 동일하게:

    - PillDetectionDataset 사용
    - combination_key 기준 make_group_split 사용
    - 동일 train / val / test 비율
    - 동일 random seed

    Args:
        dataset_root:
            원본 데이터셋 루트

        output_root:
            YOLO 데이터셋을 생성할 경로
            예: data/processed

        image_dir_name:
            원본 이미지 폴더명

        annotation_dir_name:
            원본 annotation 폴더명

        label_offset:
            PillDetectionDataset label 시작 값.
            Faster와 동일하게 1 사용을 권장합니다.

        strict:
            Dataset validation strict 여부

        validate_image_size:
            이미지 크기 검증 여부

        train_ratio:
            train split 비율

        val_ratio:
            validation split 비율

        test_ratio:
            test split 비율

        random_seed:
            split random seed

        clean:
            기존 YOLO images / labels를 삭제하고 다시 생성할지 여부

    Returns:
        생성된 data.yaml Path
    """

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. 원본 Dataset
    # ========================================================

    dataset = PillDetectionDataset(
        root=dataset_root,
        transforms=None,
        image_dir_name=image_dir_name,
        annotation_dir_name=annotation_dir_name,
        label_offset=label_offset,
        strict=strict,
        validate_image_size=validate_image_size,
    )

    print(
        f"Dataset 로드 완료: "
        f"{len(dataset)} images, "
        f"{dataset.num_classes} classes"
    )

    # ========================================================
    # 2. Faster R-CNN과 동일한 combination_key split
    # ========================================================

    group_mapping, group_keys = build_group_mapping(
        dataset
    )

    train_groups, valid_groups, test_groups = make_group_split(
        group_keys=group_keys,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=random_seed,
    )

    train_indices = groups_to_indices(
        train_groups,
        group_mapping,
    )

    valid_indices = groups_to_indices(
        valid_groups,
        group_mapping,
    )

    test_indices = groups_to_indices(
        test_groups,
        group_mapping,
    )

    validate_split(
        dataset=dataset,
        train_groups=train_groups,
        valid_groups=valid_groups,
        test_groups=test_groups,
        train_indices=train_indices,
        valid_indices=valid_indices,
        test_indices=test_indices,
    )

    split_indices = {
        "train": train_indices,
        "val": valid_indices,
        "test": test_indices,
    }

    print(
        "Split 완료:",
        f"train={len(train_indices)},",
        f"val={len(valid_indices)},",
        f"test={len(test_indices)}",
    )

    # ========================================================
    # 3. 출력 폴더 생성
    # ========================================================

    _prepare_output_directories(
        output_root=output_root,
        clean=clean,
    )

    # ========================================================
    # 4. YOLO 형식 export
    # ========================================================

    split_stats = {}

    for split_name in (
        "train",
        "val",
        "test",
    ):
        stats = _export_split(
            dataset=dataset,
            indices=split_indices[split_name],
            output_root=output_root,
            split_name=split_name,
        )

        split_stats[split_name] = stats

        print(
            f"{split_name:>5}: "
            f"{stats['images']} images, "
            f"{stats['objects']} objects"
        )

    # ========================================================
    # 5. data.yaml
    # ========================================================

    data_yaml_path = _create_data_yaml(
        dataset=dataset,
        output_root=output_root,
    )

    # ========================================================
    # 6. 최종 검증
    # ========================================================

    for split_name in (
        "train",
        "val",
        "test",
    ):
        image_dir = (
            output_root
            / "images"
            / split_name
        )

        label_dir = (
            output_root
            / "labels"
            / split_name
        )

        image_count = len(
            list(image_dir.iterdir())
        )

        label_count = len(
            list(label_dir.glob("*.txt"))
        )

        if image_count != label_count:
            raise RuntimeError(
                f"{split_name} 이미지/라벨 개수 불일치: "
                f"images={image_count}, "
                f"labels={label_count}"
            )

    print()
    print("YOLO Dataset 생성 완료")
    print("output:", output_root.resolve())
    print("data.yaml:", data_yaml_path)

    return data_yaml_path