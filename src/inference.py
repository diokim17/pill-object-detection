from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as F


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


# xyxy bbox를 xywh 형식으로 변환
def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    )


# 이미지 파일명에서 image_id 추출
def get_image_id(image_path):
    return int(Path(image_path).stem)

# Faster R-CNN 입력용 letterbox 전처리
def letterbox_image(
    image,
    image_size=640,
):
    image = ImageOps.exif_transpose(image).convert("RGB")

    original_width, original_height = image.size

    scale = min(
        image_size / original_width,
        image_size / original_height,
    )

    resized_width = int(
        round(original_width * scale)
    )
    resized_height = int(
        round(original_height * scale)
    )

    resized = image.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )

    pad_left = (
        image_size - resized_width
    ) // 2

    pad_top = (
        image_size - resized_height
    ) // 2

    canvas = Image.new(
        "RGB",
        (image_size, image_size),
        color=(0, 0, 0),
    )

    canvas.paste(
        resized,
        (pad_left, pad_top),
    )

    image_tensor = F.to_tensor(canvas)

    metadata = {
        "original_width": original_width,
        "original_height": original_height,
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
    }

    return image_tensor, metadata


# 640x640 기준 bbox를 원본 이미지 좌표로 복원
def restore_boxes_to_original(
    boxes,
    metadata,
):
    boxes = boxes.clone()

    boxes[:, [0, 2]] -= metadata["pad_left"]
    boxes[:, [1, 3]] -= metadata["pad_top"]

    boxes /= metadata["scale"]

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(
        0,
        metadata["original_width"],
    )

    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(
        0,
        metadata["original_height"],
    )

    return boxes


# 공통 prediction을 submission row 형식으로 변환
def predictions_to_submission(
    predictions,
    score_threshold=0.05,
    label_to_category_id=None,
):
    rows = []
    annotation_id = 1

    for prediction in predictions:

        image_id = prediction["image_id"]
        boxes = prediction["boxes"]
        labels = prediction["labels"]
        scores = prediction["scores"]

        for box, label, score in zip(
            boxes,
            labels,
            scores,
        ):
            score = float(score)

            if score < score_threshold:
                continue

            label = int(label)

            if label_to_category_id is not None:
                category_id = label_to_category_id[label]
            else:
                category_id = label

            bbox_x, bbox_y, bbox_w, bbox_h = xyxy_to_xywh(
                box
            )

            if bbox_w <= 0 or bbox_h <= 0:
                continue

            rows.append(
                {
                    "annotation_id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox_x": bbox_x,
                    "bbox_y": bbox_y,
                    "bbox_w": bbox_w,
                    "bbox_h": bbox_h,
                    "score": score,
                }
            )

            annotation_id += 1

    return pd.DataFrame(
        rows,
        columns=SUBMISSION_COLUMNS,
    )


# submission csv 저장
def save_submission(
    submission_df,
    output_path,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission_df.to_csv(
        output_path,
        index=False,
    )

# Faster R-CNN 모델 추론
def inference_faster_rcnn(
    model,
    image_paths,
    device,
    image_size=640,
):
    predictions = []

    model.eval()

    with torch.inference_mode():

        for image_path in image_paths:

            with Image.open(image_path) as image:
                image_tensor, metadata = (
                    letterbox_image(
                        image,
                        image_size=image_size,
                    )
                )

            output = model(
                [image_tensor.to(device)]
            )[0]

            boxes = (
                output["boxes"]
                .detach()
                .cpu()
            )

            labels = (
                output["labels"]
                .detach()
                .cpu()
            )

            scores = (
                output["scores"]
                .detach()
                .cpu()
            )

            if len(boxes) > 0:
                boxes = restore_boxes_to_original(
                    boxes,
                    metadata,
                )

            predictions.append(
                {
                    "image_id": get_image_id(
                        image_path
                    ),
                    "boxes": boxes.tolist(),
                    "labels": labels.tolist(),
                    "scores": scores.tolist(),
                }
            )

    return predictions


# YOLO 모델 추론
def inference_yolo():
    pass
