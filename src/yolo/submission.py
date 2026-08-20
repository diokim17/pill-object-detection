from __future__ import annotations

import csv
import gc
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO

from src.yolo.yolo_mapping import to_category_id


SUBMISSION_COLUMNS = [
    "annotation_id",
    "image_id",
    "category_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "score",
]


def list_test_images(
    test_image_dir: str | Path,
    *,
    suffix: str = ".png",
) -> list[Path]:
    """숫자 파일명을 image_id 순서로 정렬하여 반환합니다."""
    test_image_dir = Path(test_image_dir)

    files = sorted(
        test_image_dir.glob(f"*{suffix}"),
        key=lambda path: int(path.stem),
    )

    if not files:
        raise FileNotFoundError(f"Test 이미지가 없습니다: {test_image_dir}")

    return files


def run_inference_sanity_check(
    weights_path: str | Path,
    *,
    test_image_dir: str | Path,
    yolo_to_category_id: Mapping[int, int],
    imgsz: int,
    conf: float,
    max_det: int,
    device: int | str,
) -> None:
    """전체 제출 전 첫 이미지 1장으로 모델/매핑/입력 경로를 검사합니다."""
    model = YOLO(str(weights_path))
    test_files = list_test_images(test_image_dir)

    with Image.open(test_files[0]) as image:
        first_image_size = image.size

    result = model.predict(
        source=str(test_files[0]),
        imgsz=imgsz,
        conf=conf,
        max_det=max_det,
        device=device,
        verbose=False,
    )[0]

    category_ids = [
        to_category_id(int(cls_id), yolo_to_category_id)
        for cls_id in result.boxes.cls.cpu().numpy()
    ]

    print(
        "테스트 이미지:", len(test_files),
        "| 첫 장 크기:", first_image_size,
    )
    print(
        "첫 이미지:", test_files[0].name,
        "| 검출 수:", len(result.boxes),
        "| category_id 예:", category_ids,
    )



def create_submission(
    weights_path: str | Path,
    *,
    test_image_dir: str | Path,
    submission_path: str | Path,
    yolo_to_category_id: Mapping[int, int],
    imgsz: int,
    conf: float,
    max_det: int,
    device: int | str,
    preprocess_fn: Callable[[Path], object] | None = None,
    clear_cache_every: int = 100,
) -> Path:
    """Test 전체를 추론하여 Kaggle 제출 CSV를 생성합니다."""
    model = YOLO(str(weights_path))
    test_files = list_test_images(test_image_dir)
    submission_path = Path(submission_path)

    rows = []
    annotation_id = 1

    for index, file_path in enumerate(test_files, start=1):
        if preprocess_fn is None:
            image = str(file_path)
        else:
            image = preprocess_fn(file_path)

        result = model.predict(
            source=image,
            imgsz=imgsz,
            conf=conf,
            max_det=max_det,
            device=device,
            verbose=False,
        )[0]

        image_id = int(file_path.stem)

        for xyxy, confidence, yolo_cls in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            x1, y1, x2, y2 = xyxy
            category_id = to_category_id(
                int(yolo_cls),
                yolo_to_category_id,
            )

            rows.append(
                [
                    annotation_id,
                    image_id,
                    category_id,
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2 - x1)),
                    int(round(y2 - y1)),
                    round(float(confidence), 4),
                ]
            )
            annotation_id += 1

        del result

        if index % clear_cache_every == 0 or index == 1:
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"{index}/{len(test_files)} 처리 중...")

    submission_path.parent.mkdir(parents=True, exist_ok=True)

    with submission_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(SUBMISSION_COLUMNS)
        writer.writerows(rows)

    print("Submission 작성 완료:", submission_path)
    print("행 수:", len(rows))

    return submission_path


def validate_submission(
    submission_path: str | Path,
    *,
    valid_category_ids,
    expected_image_count: int | None = None,
) -> pd.DataFrame:
    """제출 CSV의 컬럼, bbox, score, category_id를 검증합니다."""
    submission_path = Path(submission_path)
    df = pd.read_csv(submission_path)

    if df.columns.tolist() != SUBMISSION_COLUMNS:
        raise AssertionError(
            f"제출 컬럼 순서 오류: {df.columns.tolist()}"
        )

    if not df["annotation_id"].is_unique:
        raise AssertionError("annotation_id가 중복되어 있습니다.")

    if df.isnull().any().any():
        raise AssertionError("제출 파일에 결측값이 있습니다.")

    if not (df["bbox_w"] > 0).all():
        raise AssertionError("bbox_w가 0 이하인 값이 있습니다.")

    if not (df["bbox_h"] > 0).all():
        raise AssertionError("bbox_h가 0 이하인 값이 있습니다.")

    if not df["score"].between(0, 1).all():
        raise AssertionError("score 범위가 0~1을 벗어났습니다.")

    if not set(df["category_id"]).issubset(set(valid_category_ids)):
        raise AssertionError(
            "원본 annotation에 없는 category_id가 포함되어 있습니다."
        )

    if expected_image_count is not None:
        image_count = df["image_id"].nunique()

        if image_count != expected_image_count:
            print(
                "[주의] 예측 bbox가 0개인 이미지가 있으면 제출 CSV의 "
                "image_id unique 수는 test 이미지 수보다 작을 수 있습니다."
            )
            print(
                f"test 이미지 수={expected_image_count}, "
                f"submission image_id 수={image_count}"
            )

    print("Submission 검증 완료")
    print("shape:", df.shape)
    print(
        "category_id 범위:",
        df["category_id"].min(),
        "~",
        df["category_id"].max(),
    )
    print(
        "score 범위:",
        round(df["score"].min(), 4),
        "~",
        round(df["score"].max(), 4),
    )

    return df
