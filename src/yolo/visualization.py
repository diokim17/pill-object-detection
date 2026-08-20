"""
visualization.py

Ultralytics YOLO Object Detection 결과 시각화 모듈.

기능
----
1. YOLO txt Ground Truth 읽기
2. normalized xywh -> pixel xyxy 변환
3. Ground Truth bbox 시각화
4. Prediction bbox 시각화
5. 동일 이미지의 GT / Prediction 비교 시각화
6. 지정 split에서 랜덤 이미지를 선택하여 5행 형태로 비교

사용 예시
---------
from src.yolo.visualization import visualize_gt_vs_prediction

visualize_gt_vs_prediction(
    model=model,
    image_dir="data/processed_single_crop/images/test",
    label_dir="data/processed_single_crop/labels/test",
    imgsz=640,
    conf=0.05,
    device=0,
    n_images=10,
    seed=42,
    class_names=model.names,
)
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


PathLike = Union[str, Path]


# ============================================================
# YOLO bbox 변환
# ============================================================

def yolo_xywh_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    YOLO normalized xywh를 pixel xyxy bbox로 변환합니다.
    """

    x1 = (x_center - width / 2.0) * image_width
    y1 = (y_center - height / 2.0) * image_height
    x2 = (x_center + width / 2.0) * image_width
    y2 = (y_center + height / 2.0) * image_height

    return x1, y1, x2, y2


# ============================================================
# Class name
# ============================================================

def _get_class_name(
    class_id: int,
    class_names: Optional[Union[Mapping[int, str], Sequence[str]]],
) -> str:
    """
    class ID에 대응하는 이름을 반환합니다.

    class_names가 없으면 class ID만 반환합니다.
    """

    if class_names is None:
        return str(class_id)

    if isinstance(class_names, Mapping):
        return str(
            class_names.get(
                class_id,
                class_id,
            )
        )

    if 0 <= class_id < len(class_names):
        return str(
            class_names[class_id]
        )

    return str(class_id)


# ============================================================
# Text drawing
# ============================================================

def _draw_text_with_background(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    fill: str,
) -> None:
    """
    bbox label을 읽기 쉽게 배경과 함께 그립니다.
    """

    x, y = xy

    try:
        bbox = draw.textbbox(
            (x, y),
            text,
        )

        padding = 2

        draw.rectangle(
            (
                bbox[0] - padding,
                bbox[1] - padding,
                bbox[2] + padding,
                bbox[3] + padding,
            ),
            fill=fill,
        )

    except AttributeError:
        # Pillow 구버전 fallback
        pass

    draw.text(
        (x, y),
        text,
        fill="white",
    )


# ============================================================
# GT 읽기
# ============================================================

def read_yolo_label(
    label_path: PathLike,
) -> list[dict]:
    """
    YOLO txt label을 읽습니다.

    Returns:
        [
            {
                "class_id": int,
                "x_center": float,
                "y_center": float,
                "width": float,
                "height": float,
            },
            ...
        ]
    """

    label_path = Path(label_path)

    if not label_path.exists():
        raise FileNotFoundError(
            f"YOLO label을 찾을 수 없습니다: {label_path}"
        )

    objects = []

    lines = [
        line.strip()
        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        values = line.split()

        if len(values) != 5:
            raise ValueError(
                f"잘못된 YOLO label 형식: "
                f"{label_path}, line={line_number}, "
                f"value={line}"
            )

        class_id, xc, yc, width, height = values

        objects.append(
            {
                "class_id": int(float(class_id)),
                "x_center": float(xc),
                "y_center": float(yc),
                "width": float(width),
                "height": float(height),
            }
        )

    return objects


# ============================================================
# GT 시각화
# ============================================================

def draw_ground_truth(
    image_path: PathLike,
    label_path: PathLike,
    class_names: Optional[
        Union[Mapping[int, str], Sequence[str]]
    ] = None,
    line_width: int = 4,
) -> Image.Image:
    """
    원본 이미지에 Ground Truth bbox를 그립니다.

    GT bbox는 초록색으로 표시합니다.
    """

    image_path = Path(image_path)
    label_path = Path(label_path)

    image = Image.open(
        image_path
    ).convert("RGB")

    draw = ImageDraw.Draw(image)

    image_width, image_height = (
        image.size
    )

    objects = read_yolo_label(
        label_path
    )

    for obj in objects:

        class_id = obj["class_id"]

        x1, y1, x2, y2 = (
            yolo_xywh_to_xyxy(
                x_center=obj["x_center"],
                y_center=obj["y_center"],
                width=obj["width"],
                height=obj["height"],
                image_width=image_width,
                image_height=image_height,
            )
        )

        draw.rectangle(
            (x1, y1, x2, y2),
            outline="green",
            width=line_width,
        )

        class_name = _get_class_name(
            class_id,
            class_names,
        )

        label = (
            f"GT {class_name}"
        )

        text_y = max(
            0,
            y1 - 14,
        )

        _draw_text_with_background(
            draw=draw,
            xy=(x1, text_y),
            text=label,
            fill="green",
        )

    return image


# ============================================================
# IoU 계산
# ============================================================

def calculate_iou(
    box_a: Sequence[float],
    box_b: Sequence[float],
) -> float:
    """두 xyxy bbox의 IoU를 계산합니다."""

    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _get_gt_boxes(
    image_path: PathLike,
    label_path: PathLike,
) -> list[dict]:
    """YOLO GT label을 pixel xyxy bbox로 변환합니다."""

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    gt_boxes = []

    for obj in read_yolo_label(label_path):
        gt_boxes.append(
            {
                "class_id": obj["class_id"],
                "box": yolo_xywh_to_xyxy(
                    x_center=obj["x_center"],
                    y_center=obj["y_center"],
                    width=obj["width"],
                    height=obj["height"],
                    image_width=image_width,
                    image_height=image_height,
                ),
            }
        )

    return gt_boxes


# ============================================================
# Prediction 시각화
# ============================================================

def draw_prediction(
    image_path: PathLike,
    result,
    label_path: PathLike,
    class_names: Optional[
        Union[Mapping[int, str], Sequence[str]]
    ] = None,
    line_width: int = 4,
    match_iou: float = 0.75,
) -> Image.Image:
    """
    Prediction bbox를 그립니다.

    정답: GT와 class 동일 + IoU >= match_iou -> 파랑
    오답: 위 조건 불충족 -> 빨강
    """

    image_path = Path(image_path)
    label_path = Path(label_path)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    if result.boxes is None or len(result.boxes) == 0:
        return image

    gt_boxes = _get_gt_boxes(image_path, label_path)
    matched_gt_indices = set()

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()

    # confidence가 높은 prediction부터 1:1 matching
    order = scores.argsort()[::-1]

    for prediction_index in order:
        prediction_box = tuple(map(float, boxes[prediction_index]))
        class_id = int(classes[prediction_index])
        score = float(scores[prediction_index])

        best_gt_index = None
        best_iou = 0.0

        for gt_index, gt in enumerate(gt_boxes):
            if gt_index in matched_gt_indices:
                continue

            if gt["class_id"] != class_id:
                continue

            iou = calculate_iou(prediction_box, gt["box"])

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        is_correct = (
            best_gt_index is not None
            and best_iou >= match_iou
        )

        if is_correct:
            matched_gt_indices.add(best_gt_index)
            box_color = "blue"
            status = "OK"
        else:
            box_color = "red"
            status = "WRONG"

        x1, y1, x2, y2 = prediction_box

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=box_color,
            width=line_width,
        )

        class_name = _get_class_name(class_id, class_names)

        label = (
            f"{status} {class_name} "
            f"{score:.2f} IoU={best_iou:.2f}"
        )

        _draw_text_with_background(
            draw=draw,
            xy=(x1, max(0, y1 - 14)),
            text=label,
            fill=box_color,
        )

    return image


# ============================================================
# 단일 이미지 GT vs Prediction
# ============================================================

def visualize_single_gt_vs_prediction(
    model,
    image_path: PathLike,
    label_path: PathLike,
    imgsz: int = 640,
    conf: float = 0.05,
    device: Union[int, str] = 0,
    class_names: Optional[
        Union[Mapping[int, str], Sequence[str]]
    ] = None,
    figsize: tuple[int, int] = (10, 6),
    match_iou: float = 0.75,
) -> None:
    """
    이미지 하나의 GT와 Prediction을 좌우로 비교합니다.
    """

    image_path = Path(
        image_path
    )

    result = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
    )[0]

    if class_names is None:
        class_names = getattr(
            model,
            "names",
            None,
        )

    gt_image = draw_ground_truth(
        image_path=image_path,
        label_path=label_path,
        class_names=class_names,
    )

    pred_image = draw_prediction(
        image_path=image_path,
        result=result,
        label_path=label_path,
        class_names=class_names,
        match_iou=match_iou,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
    )

    axes[0].imshow(
        gt_image
    )
    axes[0].axis(
        "off"
    )
    axes[0].set_title(
        "Ground Truth"
    )

    axes[1].imshow(
        pred_image
    )
    axes[1].axis(
        "off"
    )
    axes[1].set_title(
        "Prediction"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 여러 이미지 GT vs Prediction
# ============================================================

def visualize_gt_vs_prediction(
    model,
    image_dir: PathLike,
    label_dir: PathLike,
    imgsz: int = 640,
    conf: float = 0.05,
    device: Union[int, str] = 0,
    n_images: int = 10,
    seed: int = 42,
    rows: int = 5,
    match_iou: float = 0.75,
    class_names: Optional[
        Union[Mapping[int, str], Sequence[str]]
    ] = None,
    image_extensions: Sequence[str] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
    ),
) -> None:
    """
    지정 dataset split에서 이미지를 랜덤 선택하여
    GT와 Prediction을 비교합니다.

    기본 n_images=10, rows=5이면:

        GT | Prediction | GT | Prediction
        GT | Prediction | GT | Prediction
        GT | Prediction | GT | Prediction
        GT | Prediction | GT | Prediction
        GT | Prediction | GT | Prediction

    형태로 출력됩니다.

    Args:
        model:
            Ultralytics YOLO model

        image_dir:
            평가 이미지 폴더

        label_dir:
            YOLO txt GT label 폴더

        imgsz:
            YOLO inference image size

        conf:
            confidence threshold

        device:
            inference device

        n_images:
            시각화할 이미지 수

        seed:
            랜덤 샘플 seed

        rows:
            figure 행 수

        class_names:
            class ID -> class name.
            None이면 model.names 사용
    """

    image_dir = Path(
        image_dir
    )

    label_dir = Path(
        label_dir
    )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"이미지 폴더를 찾을 수 없습니다: "
            f"{image_dir}"
        )

    if not label_dir.exists():
        raise FileNotFoundError(
            f"라벨 폴더를 찾을 수 없습니다: "
            f"{label_dir}"
        )

    allowed_extensions = {
        ext.lower()
        for ext in image_extensions
    }

    image_paths = sorted(
        p
        for p in image_dir.iterdir()
        if (
            p.is_file()
            and p.suffix.lower()
            in allowed_extensions
        )
    )

    if not image_paths:
        raise RuntimeError(
            f"이미지가 없습니다: {image_dir}"
        )

    valid_image_paths = []

    for image_path in image_paths:

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        if label_path.exists():
            valid_image_paths.append(
                image_path
            )

    if not valid_image_paths:
        raise RuntimeError(
            "이미지와 YOLO label이 매칭되는 "
            "샘플이 없습니다."
        )

    n_images = min(
        int(n_images),
        len(valid_image_paths),
    )

    if n_images <= 0:
        raise ValueError(
            "n_images는 1 이상이어야 합니다."
        )

    rng = random.Random(
        seed
    )

    selected_images = rng.sample(
        valid_image_paths,
        k=n_images,
    )

    results = model.predict(
        source=[
            str(path)
            for path in selected_images
        ],
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
    )

    if class_names is None:
        class_names = getattr(
            model,
            "names",
            None,
        )

    # 한 이미지당 GT + Prediction = 2칸
    total_panels = (
        n_images * 2
    )

    rows = min(
        int(rows),
        n_images,
    )

    if rows <= 0:
        rows = 1

    cols = math.ceil(
        total_panels / rows
    )

    # GT/Prediction pair가 깨지지 않도록
    # columns는 짝수로 맞춤
    if cols % 2 != 0:
        cols += 1

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            4 * cols,
            4 * rows,
        ),
        squeeze=False,
    )

    axes = axes.flatten()

    for i, (
        image_path,
        result,
    ) in enumerate(
        zip(
            selected_images,
            results,
        )
    ):

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        gt_image = draw_ground_truth(
            image_path=image_path,
            label_path=label_path,
            class_names=class_names,
        )

        pred_image = draw_prediction(
            image_path=image_path,
            result=result,
            label_path=label_path,
            class_names=class_names,
            match_iou=match_iou,
        )

        gt_ax = axes[
            i * 2
        ]

        pred_ax = axes[
            i * 2 + 1
        ]

        gt_ax.imshow(
            gt_image
        )
        gt_ax.axis(
            "off"
        )
        gt_ax.set_title(
            f"GT\n{image_path.name}",
            fontsize=8,
        )

        pred_ax.imshow(
            pred_image
        )
        pred_ax.axis(
            "off"
        )
        pred_ax.set_title(
            "Prediction",
            fontsize=8,
        )

    # 남는 subplot 제거
    for ax in axes[
        total_panels:
    ]:
        ax.axis(
            "off"
        )

    plt.tight_layout()
    plt.show()
