"""
yolo_dataset_single_object.py

PillDetectionDataset을 이용하여 원본 알약 데이터셋을
Ultralytics YOLO single-object crop 형식으로 변환합니다.

역할
----
1. PillDetectionDataset으로 원본 이미지 / annotation 로드
2. split_utils.make_group_split()으로 train / val / test 분할
3. train split은 객체별 single-object crop 이미지로 생성
4. val / test split은 원본 이미지를 그대로 유지
5. bbox를 YOLO normalized xywh 형식으로 변환
6. images/{train,val,test}, labels/{train,val,test} 생성
7. data.yaml 생성

실험 목적
---------
기존 YOLO baseline과 동일한 Dataset / split / class mapping을 유지하고,
train 입력만 multi-object 이미지에서 single-object crop 이미지로 변경하여
single-object 학습 효과를 비교합니다.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

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
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"잘못된 이미지 크기입니다: "
            f"width={image_width}, height={image_height}"
        )

    x1, y1, x2, y2 = map(float, box)

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
# bbox utility
# ============================================================

def _boxes_intersect(
    box_a: Sequence[float],
    box_b: Sequence[float],
) -> bool:
    """
    두 xyxy bbox가 면적을 가지고 겹치는지 확인합니다.
    """

    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)

    return inter_w > 0 and inter_h > 0


def _make_single_crop_box(
    target_box: Sequence[float],
    other_boxes: Sequence[Sequence[float]],
    image_width: int,
    image_height: int,
    crop_scale_range: Tuple[float, float],
    random_offset: bool,
    rng: random.Random,
    max_trials: int = 100,
) -> tuple[int, int, int, int]:
    """
    target bbox를 완전히 포함하면서 다른 객체 bbox와 겹치지 않는
    single-object crop 영역을 생성합니다.

    crop_scale_range:
        target bbox 대비 crop width / height 배율 범위.
        예: (1.3, 2.0)

    다른 객체를 포함하지 않는 후보를 찾지 못하면,
    target bbox에 최소 padding만 둔 안전한 crop으로 fallback합니다.
    """

    tx1, ty1, tx2, ty2 = map(float, target_box)

    target_w = tx2 - tx1
    target_h = ty2 - ty1

    if target_w <= 0 or target_h <= 0:
        raise ValueError(
            f"유효하지 않은 target bbox입니다: {target_box}"
        )

    min_scale, max_scale = map(float, crop_scale_range)

    if min_scale < 1.0:
        raise ValueError(
            f"crop_scale_range 최소값은 1.0 이상이어야 합니다: "
            f"{crop_scale_range}"
        )

    if max_scale < min_scale:
        raise ValueError(
            f"crop_scale_range가 잘못되었습니다: "
            f"{crop_scale_range}"
        )

    target_cx = (tx1 + tx2) / 2.0
    target_cy = (ty1 + ty2) / 2.0

    for _ in range(max_trials):
        scale = rng.uniform(min_scale, max_scale)

        crop_w = min(
            float(image_width),
            target_w * scale,
        )
        crop_h = min(
            float(image_height),
            target_h * scale,
        )

        # target bbox가 crop 안에 완전히 들어가기 위해
        # crop 좌상단이 가질 수 있는 범위
        x1_min = max(0.0, tx2 - crop_w)
        x1_max = min(tx1, image_width - crop_w)

        y1_min = max(0.0, ty2 - crop_h)
        y1_max = min(ty1, image_height - crop_h)

        if x1_min > x1_max or y1_min > y1_max:
            continue

        if random_offset:
            crop_x1 = rng.uniform(x1_min, x1_max)
            crop_y1 = rng.uniform(y1_min, y1_max)
        else:
            crop_x1 = target_cx - crop_w / 2.0
            crop_y1 = target_cy - crop_h / 2.0

            crop_x1 = min(
                max(crop_x1, x1_min),
                x1_max,
            )
            crop_y1 = min(
                max(crop_y1, y1_min),
                y1_max,
            )

        crop_x2 = crop_x1 + crop_w
        crop_y2 = crop_y1 + crop_h

        crop_box = (
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
        )

        has_other_object = any(
            _boxes_intersect(
                crop_box,
                other_box,
            )
            for other_box in other_boxes
        )

        if has_other_object:
            continue

        return (
            int(round(crop_x1)),
            int(round(crop_y1)),
            int(round(crop_x2)),
            int(round(crop_y2)),
        )

    # ========================================================
    # Fallback
    # ========================================================
    # 다른 객체가 매우 가까워 넓은 crop이 불가능하면
    # target bbox 주변에 아주 작은 padding만 적용합니다.
    # 그래도 다른 bbox와 겹치면 target bbox 자체를 사용합니다.

    padding = 2.0

    fallback_box = (
        max(0.0, tx1 - padding),
        max(0.0, ty1 - padding),
        min(float(image_width), tx2 + padding),
        min(float(image_height), ty2 + padding),
    )

    if any(
        _boxes_intersect(
            fallback_box,
            other_box,
        )
        for other_box in other_boxes
    ):
        fallback_box = (
            tx1,
            ty1,
            tx2,
            ty2,
        )

    return (
        int(round(fallback_box[0])),
        int(round(fallback_box[1])),
        int(round(fallback_box[2])),
        int(round(fallback_box[3])),
    )


# ============================================================
# Single-object YOLO label 저장
# ============================================================

def _write_single_yolo_label(
    dataset: PillDetectionDataset,
    label: int,
    target_box: Sequence[float],
    crop_box: Sequence[int],
    label_path: Path,
) -> None:
    """
    원본 target bbox를 crop 좌표계로 변환한 뒤,
    YOLO txt label 하나를 저장합니다.
    """

    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    tx1, ty1, tx2, ty2 = map(float, target_box)

    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1

    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(
            f"잘못된 crop 크기입니다: {crop_box}"
        )

    # 원본 좌표 -> crop 내부 좌표
    cropped_target_box = (
        tx1 - crop_x1,
        ty1 - crop_y1,
        tx2 - crop_x1,
        ty2 - crop_y1,
    )

    # Faster R-CNN Dataset label: 1 ~ N
    label = int(label)

    # YOLO class index: 0 ~ N-1
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
        box=cropped_target_box,
        image_width=crop_width,
        image_height=crop_height,
    )

    label_path.write_text(
        (
            f"{yolo_class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{bbox_width:.6f} "
            f"{bbox_height:.6f}"
        ),
        encoding="utf-8",
    )


# ============================================================
# Train: single-object crop export
# ============================================================

def _export_train_single_object(
    dataset: PillDetectionDataset,
    indices: Sequence[int],
    output_root: Path,
    crop_scale_range: Tuple[float, float],
    random_offset: bool,
    random_seed: int,
) -> Dict[str, int]:
    """
    train split의 각 객체를 각각 하나의 single-object crop으로 저장합니다.

    원본 이미지 1장에 객체가 4개라면:
        image__obj_000.png
        image__obj_001.png
        image__obj_002.png
        image__obj_003.png

    각 label txt에는 객체 하나만 존재합니다.
    """

    image_output_dir = (
        output_root
        / "images"
        / "train"
    )

    label_output_dir = (
        output_root
        / "labels"
        / "train"
    )

    rng = random.Random(random_seed)

    source_image_count = 0
    crop_image_count = 0
    object_count = 0
    fallback_tight_count = 0

    for sample_index in indices:
        sample = dataset.samples[sample_index]

        source_image_path = Path(
            sample["image_path"]
        )

        target = sample["target"]

        boxes = target["boxes"].tolist()
        labels = target["labels"].tolist()

        if len(boxes) != len(labels):
            raise ValueError(
                f"bbox / label 개수가 다릅니다: "
                f"{source_image_path.name}, "
                f"boxes={len(boxes)}, labels={len(labels)}"
            )

        if not boxes:
            continue

        with Image.open(source_image_path) as image:
            image = image.convert("RGB")
            image_width, image_height = image.size

            source_image_count += 1

            for object_index, (box, label) in enumerate(
                zip(boxes, labels)
            ):
                other_boxes = [
                    other_box
                    for idx, other_box in enumerate(boxes)
                    if idx != object_index
                ]

                crop_box = _make_single_crop_box(
                    target_box=box,
                    other_boxes=other_boxes,
                    image_width=image_width,
                    image_height=image_height,
                    crop_scale_range=crop_scale_range,
                    random_offset=random_offset,
                    rng=rng,
                )

                crop_x1, crop_y1, crop_x2, crop_y2 = crop_box

                # fallback으로 target bbox와 거의 동일하게 잘렸는지 기록
                target_w = float(box[2]) - float(box[0])
                target_h = float(box[3]) - float(box[1])

                crop_w = crop_x2 - crop_x1
                crop_h = crop_y2 - crop_y1

                if (
                    crop_w <= target_w + 5
                    and crop_h <= target_h + 5
                ):
                    fallback_tight_count += 1

                cropped_image = image.crop(
                    (
                        crop_x1,
                        crop_y1,
                        crop_x2,
                        crop_y2,
                    )
                )

                output_stem = (
                    f"{source_image_path.stem}"
                    f"__obj_{object_index:03d}"
                )

                destination_image_path = (
                    image_output_dir
                    / f"{output_stem}.png"
                )

                destination_label_path = (
                    label_output_dir
                    / f"{output_stem}.txt"
                )

                cropped_image.save(
                    destination_image_path
                )

                _write_single_yolo_label(
                    dataset=dataset,
                    label=int(label),
                    target_box=box,
                    crop_box=crop_box,
                    label_path=destination_label_path,
                )

                crop_image_count += 1
                object_count += 1

    return {
        "source_images": source_image_count,
        "images": crop_image_count,
        "objects": object_count,
        "tight_fallbacks": fallback_tight_count,
    }


# ============================================================
# Val / Test: 원본 export
# ============================================================

def _write_original_yolo_label(
    dataset: PillDetectionDataset,
    sample_index: int,
    label_path: Path,
) -> int:
    """
    val / test용 원본 이미지의 전체 객체를 YOLO txt로 저장합니다.
    """

    sample = dataset.samples[sample_index]

    image_path = Path(sample["image_path"])
    target = sample["target"]

    boxes = target["boxes"].tolist()
    labels = target["labels"].tolist()

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    if len(boxes) != len(labels):
        raise ValueError(
            f"bbox / label 개수가 다릅니다: "
            f"{image_path.name}, "
            f"boxes={len(boxes)}, labels={len(labels)}"
        )

    lines: List[str] = []

    for box, label in zip(boxes, labels):
        label = int(label)

        yolo_class_id = (
            label
            - dataset.label_offset
        )

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


def _export_original_split(
    dataset: PillDetectionDataset,
    indices: Sequence[int],
    output_root: Path,
    split_name: str,
) -> Dict[str, int]:
    """
    val / test split은 기존 baseline처럼 원본 이미지를 그대로 저장합니다.
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

        shutil.copy2(
            source_image_path,
            destination_image_path,
        )

        num_objects = _write_original_yolo_label(
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
# YOLO Single-object Dataset 생성
# ============================================================

def build_yolo_single_object_dataset(
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
    crop_scale_range: Tuple[float, float] = (1.3, 2.0),
    random_offset: bool = True,
    clean: bool = True,
) -> Path:
    """
    원본 알약 데이터셋을 YOLO single-object 학습 데이터셋으로 변환합니다.

    핵심 조건
    ---------
    - PillDetectionDataset 사용
    - 기존과 동일한 combination_key split
    - train만 객체별 single-object crop
    - val / test는 원본 multi-object 이미지 유지
    - 기존 class mapping 유지
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
    # 2. 기존과 동일한 combination_key split
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
    # 4. Train -> Single-object crop
    # ========================================================

    train_stats = _export_train_single_object(
        dataset=dataset,
        indices=train_indices,
        output_root=output_root,
        crop_scale_range=crop_scale_range,
        random_offset=random_offset,
        random_seed=random_seed,
    )

    print(
        "train:",
        f"{train_stats['source_images']} source images ->",
        f"{train_stats['images']} single crops,",
        f"{train_stats['objects']} objects,",
        f"tight_fallbacks={train_stats['tight_fallbacks']}",
    )

    # ========================================================
    # 5. Val / Test -> 원본 유지
    # ========================================================

    for split_name, indices in (
        ("val", valid_indices),
        ("test", test_indices),
    ):
        stats = _export_original_split(
            dataset=dataset,
            indices=indices,
            output_root=output_root,
            split_name=split_name,
        )

        print(
            f"{split_name:>5}: "
            f"{stats['images']} images, "
            f"{stats['objects']} objects"
        )

    # ========================================================
    # 6. data.yaml
    # ========================================================

    data_yaml_path = _create_data_yaml(
        dataset=dataset,
        output_root=output_root,
    )

    # ========================================================
    # 7. 최종 검증
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

    # train label은 반드시 한 줄이어야 함
    invalid_train_labels = []

    train_label_dir = (
        output_root
        / "labels"
        / "train"
    )

    for label_path in train_label_dir.glob("*.txt"):
        lines = [
            line
            for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if len(lines) != 1:
            invalid_train_labels.append(
                label_path.name
            )

    if invalid_train_labels:
        raise RuntimeError(
            "single-object train label에 "
            "객체가 1개가 아닌 파일이 있습니다: "
            f"{invalid_train_labels[:10]}"
        )

    print()
    print("YOLO Single-object Dataset 생성 완료")
    print("output:", output_root.resolve())
    print("data.yaml:", data_yaml_path)

    return data_yaml_path
