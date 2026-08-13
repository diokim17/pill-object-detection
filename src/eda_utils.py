"""경구약제 객체 탐지 데이터셋용 공용 EDA 유틸리티.

이 데이터셋은 하나의 이미지에 포함된 약품 Annotation이 여러 JSON 파일에
분산되어 있다. 이 모듈은 분산 JSON 통합, 이미지 단위 재구성, Annotation
누락 탐지, 이미지/JSON 매칭, Bounding Box 검증 및 학습용 클래스 ID 매핑을
팀원들이 동일한 기준으로 수행할 수 있도록 제공한다.

주의
----
이 함수들은 원본 이미지나 JSON을 직접 수정하지 않는다. 발견된 오류를
수정하거나 데이터를 제외하는 정책은 팀에서 결정한 후 별도로 적용해야 한다.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# EDA 및 오류 분석을 위해 ann_df에 함께 보관할 이미지/약품 메타데이터
METADATA_FIELDS = (
    "back_color",
    "light_color",
    "camera_la",
    "camera_lo",
    "drug_dir",
    "drug_shape",
    "color_class1",
    "print_front",
    "print_back",
)


def find_annotation_files(annotation_dir: str | Path) -> list[Path]:
    """Annotation 디렉터리 아래의 모든 JSON 경로를 정렬해 반환한다.

    정렬된 경로를 사용하면 실행할 때마다 ann_df의 행 순서가 달라지는 것을
    방지할 수 있다.

    Parameters
    ----------
    annotation_dir:
        ``train_annotations`` 디렉터리 경로.
    """
    return sorted(Path(annotation_dir).rglob("*.json"))


def load_annotation_dataframe(
    annotation_dir: str | Path,
) -> pd.DataFrame:
    """분산된 COCO 형식 JSON을 객체 단위 DataFrame으로 통합한다.

    반환되는 ``ann_df``는 한 행이 하나의 Annotation(객체)을 의미한다.
    같은 이미지에 약품이 여러 개 있으면 동일한 ``file_name``이 여러 행에
    나타나는 것이 정상이다.

    Parameters
    ----------
    annotation_dir:
        ``train_annotations`` 디렉터리 경로.

    Returns
    -------
    pandas.DataFrame
        이미지 정보, 클래스, bbox, 면적, 메타데이터 및 원본 JSON 경로가
        포함된 객체 단위 테이블.
    """
    records: list[dict] = []

    for json_path in find_annotation_files(annotation_dir):
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        # 원본 category_id를 사람이 읽을 수 있는 약품명으로 변환하기 위한 맵
        category_map = {
            category["id"]: category["name"]
            for category in data.get("categories", [])
        }

        # JSON에 images/annotations가 여러 개 있어도 image_id로 안전하게 연결
        annotations_by_image: dict[int, list[dict]] = {}
        for annotation in data.get("annotations", []):
            annotations_by_image.setdefault(
                annotation["image_id"], []
            ).append(annotation)

        for image in data.get("images", []):
            for annotation in annotations_by_image.get(image["id"], []):
                x, y, width, height = annotation["bbox"]

                record = {
                    "file_name": image["file_name"],
                    "image_id": image["id"],
                    "image_width": image["width"],
                    "image_height": image["height"],
                    "annotation_id": annotation.get("id"),
                    "category_id": annotation["category_id"],
                    "category_name": category_map.get(
                        annotation["category_id"], "Unknown"
                    ),
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": width,
                    "bbox_h": height,
                    "area": annotation.get("area"),
                    "iscrowd": annotation.get("iscrowd", 0),
                    # 문제가 발견됐을 때 원본 JSON을 추적하기 위한 경로
                    "json_path": str(json_path),
                }

                record.update(
                    {field: image.get(field) for field in METADATA_FIELDS}
                )
                records.append(record)

    return pd.DataFrame(records)


def build_image_annotation_map(ann_df: pd.DataFrame) -> pd.DataFrame:
    """객체 단위 ann_df를 이미지 한 장당 한 행인 테이블로 변환한다.

    Dataset 클래스에서 이미지 한 장의 bbox와 label을 한 번에 가져올 때
    사용할 수 있다. 함수 마지막의 두 assert는 그룹화 과정에서 이미지나
    Annotation이 누락되지 않았는지 확인한다.
    """
    rows = []

    for file_name, group in ann_df.groupby("file_name", sort=True):
        rows.append(
            {
                "file_name": file_name,
                "num_objects": len(group),
                "category_ids": group["category_id"].tolist(),
                "category_names": group["category_name"].tolist(),
                "bboxes": group[
                    ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]
                ].values.tolist(),
                "image_width": group["image_width"].iloc[0],
                "image_height": group["image_height"].iloc[0],
            }
        )

    result = pd.DataFrame(rows)

    # 고유 이미지 수와 객체 수가 변환 전후에 동일해야 한다.
    assert ann_df["file_name"].nunique() == len(result)
    assert len(ann_df) == result["num_objects"].sum()

    return result


def extract_expected_category_ids(file_name: str) -> set[int]:
    """이미지 파일명 앞부분에서 예상 약품 category_id를 추출한다.

    예를 들어 ``K-003351-032310-038162_...png``에서는
    ``{3351, 32310, 38162}``를 반환한다. 파일명 규칙을 이용한 추론이므로
    데이터셋 파일명 규칙이 변경되면 이 함수도 수정해야 한다.
    """
    prefix = file_name.split("_", maxsplit=1)[0]
    return {int(value) for value in re.findall(r"\d{6}", prefix)}


def find_missing_annotations(
    annotation_dir: str | Path,
    ann_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """파일명 기준 예상 JSON과 실제 JSON을 비교해 누락 약품을 찾는다.

    Returns
    -------
    missing_images:
        JSON 일부가 누락된 이미지 목록과 예상/실제 JSON 개수.
    missing_drugs:
        이미지별로 누락된 category_id와 약품명.
    missing_summary:
        약품별 누락 횟수와 영향을 받은 이미지 목록.

    Notes
    -----
    파일명으로 누락된 클래스는 찾을 수 있지만 누락된 bbox 좌표는 복원할
    수 없다. 해당 이미지를 제외하거나 재라벨링할지는 팀 결정이 필요하다.
    """
    json_counts: Counter[str] = Counter()

    for json_path in find_annotation_files(annotation_dir):
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        for image in data.get("images", []):
            json_counts[image["file_name"]] += 1

    count_df = (
        pd.Series(json_counts, name="json_count")
        .rename_axis("file_name")
        .reset_index()
    )
    count_df["expected_category_ids"] = count_df["file_name"].map(
        extract_expected_category_ids
    )
    count_df["expected_json_count"] = count_df[
        "expected_category_ids"
    ].map(len)
    count_df["missing_json_count"] = (
        count_df["expected_json_count"] - count_df["json_count"]
    )

    missing_images = count_df.loc[
        count_df["missing_json_count"] > 0
    ].copy()

    category_name_map = (
        ann_df[["category_id", "category_name"]]
        .drop_duplicates()
        .set_index("category_id")["category_name"]
        .to_dict()
    )

    missing_records = []
    for row in missing_images.itertuples(index=False):
        actual_ids = set(
            ann_df.loc[
                ann_df["file_name"] == row.file_name,
                "category_id",
            ]
        )

        # 예상 ID 집합 - 실제 ID 집합 = 누락된 약품 ID
        for category_id in sorted(row.expected_category_ids - actual_ids):
            missing_records.append(
                {
                    "file_name": row.file_name,
                    "missing_category_id": category_id,
                    "missing_category_name": category_name_map.get(
                        category_id, "Unknown"
                    ),
                    "expected_category_ids": sorted(
                        row.expected_category_ids
                    ),
                    "actual_category_ids": sorted(actual_ids),
                }
            )

    missing_drugs = pd.DataFrame(missing_records)

    if not missing_drugs.empty:
        missing_summary = (
            missing_drugs.groupby(
                ["missing_category_id", "missing_category_name"],
                as_index=False,
            )
            .agg(
                missing_count=("file_name", "size"),
                affected_images=("file_name", list),
            )
            .sort_values("missing_count", ascending=False)
        )
    else:
        missing_summary = pd.DataFrame(
            columns=[
                "missing_category_id",
                "missing_category_name",
                "missing_count",
                "affected_images",
            ]
        )

    # 계산된 누락 개수와 실제 식별한 누락 약품 수가 같아야 한다.
    assert len(missing_drugs) == int(
        missing_images["missing_json_count"].sum()
    )

    return missing_images, missing_drugs, missing_summary


def compare_image_and_annotation_files(
    image_dir: str | Path,
    ann_df: pd.DataFrame,
    extensions: Iterable[str] = (".png", ".jpg", ".jpeg"),
) -> tuple[set[str], set[str]]:
    """실제 이미지 파일과 Annotation 파일명을 양방향으로 비교한다.

    Returns
    -------
    images_without_annotations:
        실제 이미지는 있지만 Annotation에 없는 파일명 집합.
    annotations_without_images:
        Annotation에는 있지만 실제 이미지가 없는 파일명 집합.
    """
    extensions = {extension.lower() for extension in extensions}
    actual_names = {
        path.name
        for path in Path(image_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    }
    annotated_names = set(ann_df["file_name"].unique())

    return (
        actual_names - annotated_names,
        annotated_names - actual_names,
    )


def validate_bounding_boxes(
    ann_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bounding Box 좌표와 저장된 area 값의 유효성을 검사한다.

    Returns
    -------
    checked:
        ``bbox_x2``, ``bbox_y2``, ``calculated_area``를 추가한 복사본.
    invalid_coordinates:
        음수, 크기 0 이하 또는 이미지 범위를 벗어난 bbox 행.
    invalid_areas:
        원본 area와 ``bbox_w * bbox_h``가 일치하지 않는 행.

    원본 ann_df는 수정하지 않는다.
    """
    checked = ann_df.copy()
    checked["bbox_x2"] = checked["bbox_x"] + checked["bbox_w"]
    checked["bbox_y2"] = checked["bbox_y"] + checked["bbox_h"]
    checked["calculated_area"] = (
        checked["bbox_w"] * checked["bbox_h"]
    )

    invalid_coordinates = checked.loc[
        (checked["bbox_x"] < 0)
        | (checked["bbox_y"] < 0)
        | (checked["bbox_w"] <= 0)
        | (checked["bbox_h"] <= 0)
        | (checked["bbox_x2"] > checked["image_width"])
        | (checked["bbox_y2"] > checked["image_height"])
    ].copy()

    invalid_areas = checked.loc[
        ~np.isclose(
            checked["area"],
            checked["calculated_area"],
        )
    ].copy()
    invalid_areas["area_difference"] = (
        invalid_areas["area"] - invalid_areas["calculated_area"]
    )

    return checked, invalid_coordinates, invalid_areas


def create_label_mapping(
    ann_df: pd.DataFrame,
    start: int = 1,
) -> tuple[dict[int, int], dict[int, int]]:
    """불연속 원본 category_id를 연속된 학습용 label로 매핑한다.

    Parameters
    ----------
    ann_df:
        ``category_id`` 컬럼을 포함하는 객체 단위 DataFrame.
    start:
        label의 시작 번호. background에 0을 예약하는 torchvision
        Faster R-CNN은 1, YOLO 계열은 일반적으로 0을 사용한다.

    Returns
    -------
    category_to_label, label_to_category
        원본 ID→학습 label 맵과 그 역방향 맵.
    """
    category_ids = sorted(
        int(value)
        for value in ann_df["category_id"].unique()
    )
    category_to_label = {
        category_id: label
        for label, category_id in enumerate(
            category_ids,
            start=start,
        )
    }
    label_to_category = {
        label: category_id
        for category_id, label in category_to_label.items()
    }

    return category_to_label, label_to_category


# 사용 예시
# ---------
# 아래 코드는 팀 프로젝트 노트북에서 필요한 부분만 복사해 사용할 수 있다.
#
# from src.eda_utils import (
#     build_image_annotation_map,
#     compare_image_and_annotation_files,
#     create_label_mapping,
#     find_missing_annotations,
#     load_annotation_dataframe,
#     validate_bounding_boxes,
# )
#
# ann_df = load_annotation_dataframe(TRAIN_ANNOTATION_DIR)
# image_annotation_map = build_image_annotation_map(ann_df)
#
# missing_images, missing_drugs, missing_summary = find_missing_annotations(
#     TRAIN_ANNOTATION_DIR,
#     ann_df,
# )
#
# images_without_annotations, annotations_without_images = (
#     compare_image_and_annotation_files(TRAIN_IMAGE_DIR, ann_df)
# )
#
# ann_df, invalid_bbox, invalid_area = validate_bounding_boxes(ann_df)
#
# # Faster R-CNN: start=1 / YOLO: start=0
# category_to_label, label_to_category = create_label_mapping(
#     ann_df,
#     start=1,
# )
