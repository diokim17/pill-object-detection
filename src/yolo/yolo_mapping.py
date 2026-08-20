# ============================================================
# YOLO Class ID → Original Category ID Mapping
# ============================================================

import json
from pathlib import Path


def build_yolo_to_category_id(
    annotation_dir: Path,
) -> dict:
    """
    YOLO class ID(0~N-1)를 원본 COCO category_id로 역매핑합니다.

    기존 YOLO 제출 코드와 동일하게:
    1. annotation의 category_id 수집
    2. 중복 제거
    3. 오름차순 정렬
    4. 0부터 순서대로 YOLO class ID 부여
    """

    category_ids = set()

    for json_path in annotation_dir.rglob("*.json"):

        with json_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        for category in data.get(
            "categories",
            [],
        ):
            category_ids.add(
                int(category["id"])
            )

    sorted_ids = sorted(
        category_ids
    )

    return {
        yolo_id: category_id
        for yolo_id, category_id
        in enumerate(sorted_ids)
    }


def to_category_id(
    yolo_cls: int,
    yolo_to_category_id: dict,
) -> int:
    """
    YOLO class ID를 원본 category_id로 변환합니다.
    """

    return yolo_to_category_id[
        int(yolo_cls)
    ]