from pathlib import Path

import pandas as pd
import torch
from PIL import Image
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
):
    predictions = []

    model.eval()

    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            image_tensor = F.to_tensor(image).to(device)

            output = model([image_tensor])[0]

            predictions.append(
                {
                    "image_id": get_image_id(image_path),
                    "boxes": output["boxes"].cpu().tolist(),
                    "labels": output["labels"].cpu().tolist(),
                    "scores": output["scores"].cpu().tolist(),
                }
            )

    return predictions

# YOLO 모델 추론
def inference_yolo():
    pass
