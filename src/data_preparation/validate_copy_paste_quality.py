#!/usr/bin/env python3
"""Validate Copy-Paste images and COCO-style per-image annotations.

Purpose
-------
Reproduce the non-destructive quality checks used for the project's CP v3
(CP63), CP v4 (CP126), and CP v5 (CP500 final) dataset directories: basename pairing, annotation counts, bbox image
boundaries, bbox overlap, a 20 px minimum object gap, and category-level bbox
area outliers.

Input structure
---------------
DATASET_ROOT/images/*.png
DATASET_ROOT/annotations/*.json

Each JSON must contain ``images``, ``annotations``, and ``categories``. The
script expects one image record per JSON and COCO ``[x, y, width, height]``
bboxes. It only reads inputs. Use ``--report`` to write a JSON summary.

Examples
--------
python src/data_preparation/validate_copy_paste_quality.py data/copy_paste_v3
python src/data_preparation/validate_copy_paste_quality.py data/copy_paste_v5 \
  --expected-images 500 --report reports/cp500_validation.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_MIN_GAP = 20.0


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _quantile(values: list[float], fraction: float) -> float:
    """Linear quantile compatible with the original NumPy-based IQR check."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _gap_violation(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    gap: float,
) -> bool:
    return not (
        a[2] + gap <= b[0]
        or b[2] + gap <= a[0]
        or a[3] + gap <= b[1]
        or b[3] + gap <= a[1]
    )


def validate_dataset(
    dataset_root: Path,
    *,
    min_gap: float = DEFAULT_MIN_GAP,
    expected_images: int | None = None,
    size_iqr_multiplier: float = 1.5,
    allowed_object_counts: tuple[int, ...] = (3, 4),
) -> dict[str, Any]:
    """Return a validation summary without modifying the dataset."""
    image_dir = dataset_root / "images"
    annotation_dir = dataset_root / "annotations"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(
            f"Expected '{image_dir}' and '{annotation_dir}'. "
            "Pass the Copy-Paste dataset root containing images/ and annotations/."
        )

    image_paths = {path.stem: path for path in image_dir.glob("*.png")}
    annotation_paths = {path.stem: path for path in annotation_dir.glob("*.json")}
    missing_annotations = sorted(set(image_paths) - set(annotation_paths))
    missing_images = sorted(set(annotation_paths) - set(image_paths))
    issues: list[dict[str, Any]] = []
    object_counts: Counter[int] = Counter()
    category_areas: defaultdict[int, list[tuple[str, int, float]]] = defaultdict(list)
    total_annotations = 0
    image_ids: set[Any] = set()
    annotation_ids: set[Any] = set()

    for stem in sorted(set(image_paths) & set(annotation_paths)):
        image_path = image_paths[stem]
        annotation_path = annotation_paths[stem]
        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            issues.append({"type": "json_parse_error", "file": annotation_path.name, "detail": str(exc)})
            continue

        for key in ("images", "annotations", "categories"):
            if not isinstance(payload.get(key), list):
                issues.append({"type": "missing_or_invalid_key", "file": annotation_path.name, "key": key})
        images = payload.get("images") if isinstance(payload.get("images"), list) else []
        annotations = payload.get("annotations") if isinstance(payload.get("annotations"), list) else []
        categories = payload.get("categories") if isinstance(payload.get("categories"), list) else []
        if len(images) != 1:
            issues.append({"type": "image_record_count", "file": annotation_path.name, "count": len(images)})

        with Image.open(image_path) as image:
            actual_width, actual_height = image.size
        if images:
            record = images[0]
            image_id = record.get("id")
            if image_id in image_ids:
                issues.append({"type": "duplicate_image_id", "file": annotation_path.name, "image_id": image_id})
            image_ids.add(image_id)
            if record.get("file_name") not in (None, image_path.name):
                issues.append({
                    "type": "image_record_basename",
                    "file": annotation_path.name,
                    "recorded": record.get("file_name"),
                    "actual": image_path.name,
                })
            width = record.get("width", actual_width)
            height = record.get("height", actual_height)
            if width != actual_width or height != actual_height:
                issues.append({
                    "type": "image_size_mismatch",
                    "file": image_path.name,
                    "recorded": [width, height],
                    "actual": [actual_width, actual_height],
                })

        valid_categories = {item.get("id") for item in categories if isinstance(item, dict)}
        object_counts[len(annotations)] += 1
        if len(annotations) not in allowed_object_counts:
            issues.append({
                "type": "annotation_object_count",
                "file": annotation_path.name,
                "count": len(annotations),
                "allowed": list(allowed_object_counts),
            })
        total_annotations += len(annotations)
        rectangles: list[tuple[float, float, float, float]] = []
        for index, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                issues.append({"type": "invalid_annotation", "file": annotation_path.name, "index": index})
                continue
            annotation_id = annotation.get("id")
            if annotation_id in annotation_ids:
                issues.append({"type": "duplicate_annotation_id", "file": annotation_path.name, "annotation_id": annotation_id})
            annotation_ids.add(annotation_id)
            if images and annotation.get("image_id") != images[0].get("id"):
                issues.append({
                    "type": "annotation_image_id",
                    "file": annotation_path.name,
                    "index": index,
                    "annotation_image_id": annotation.get("image_id"),
                    "image_id": images[0].get("id"),
                })
            bbox = annotation.get("bbox") if isinstance(annotation, dict) else None
            if not (
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(_finite_number(value) for value in bbox)
            ):
                issues.append({"type": "invalid_bbox", "file": annotation_path.name, "index": index, "bbox": bbox})
                continue
            x, y, width, height = map(float, bbox)
            if width <= 0 or height <= 0:
                issues.append({"type": "invalid_bbox_size", "file": annotation_path.name, "index": index, "bbox": bbox})
                continue
            rectangle = (x, y, x + width, y + height)
            if x < 0 or y < 0 or rectangle[2] > actual_width or rectangle[3] > actual_height:
                issues.append({"type": "boundary", "file": annotation_path.name, "index": index, "bbox": bbox})
            category_id = annotation.get("category_id")
            if category_id not in valid_categories:
                issues.append({"type": "unknown_category", "file": annotation_path.name, "index": index, "category_id": category_id})
            if isinstance(category_id, int):
                category_areas[category_id].append((annotation_path.name, index, width * height))
            rectangles.append(rectangle)

        for left in range(len(rectangles)):
            for right in range(left + 1, len(rectangles)):
                if _overlap(rectangles[left], rectangles[right]):
                    issues.append({"type": "bbox_overlap", "file": annotation_path.name, "objects": [left, right]})
                elif _gap_violation(rectangles[left], rectangles[right], min_gap):
                    issues.append({
                        "type": "minimum_gap",
                        "file": annotation_path.name,
                        "objects": [left, right],
                        "required_px": min_gap,
                    })

    # CP500 repair used category-level bbox area with Tukey's 1.5 IQR rule.
    for category_id, rows in category_areas.items():
        areas = [row[2] for row in rows]
        if len(areas) < 4:
            continue
        q1, q3 = _quantile(areas, 0.25), _quantile(areas, 0.75)
        spread = q3 - q1
        lower = max(1.0, q1 - size_iqr_multiplier * spread)
        upper = q3 + size_iqr_multiplier * spread
        for filename, index, area in rows:
            if area < lower or area > upper:
                issues.append({
                    "type": "category_size_outlier",
                    "file": filename,
                    "index": index,
                    "category_id": category_id,
                    "area": area,
                    "allowed_area": [lower, upper],
                })

    issue_counts = Counter(issue["type"] for issue in issues)
    if missing_annotations:
        issue_counts["missing_annotation"] += len(missing_annotations)
    if missing_images:
        issue_counts["missing_image"] += len(missing_images)
    if expected_images is not None and (len(image_paths) != expected_images or len(annotation_paths) != expected_images):
        issue_counts["expected_count_mismatch"] += 1

    passed = not issue_counts
    return {
        "status": "pass" if passed else "fail",
        "dataset_root": str(dataset_root),
        "images": len(image_paths),
        "annotations": len(annotation_paths),
        "paired_basenames": len(set(image_paths) & set(annotation_paths)),
        "total_objects": total_annotations,
        "annotation_object_count_distribution": {str(key): value for key, value in sorted(object_counts.items())},
        "minimum_gap_px": min_gap,
        "allowed_annotation_object_counts": list(allowed_object_counts),
        "size_outlier_rule": f"category bbox area Tukey IQR x {size_iqr_multiplier}",
        "missing_annotation_files": missing_annotations,
        "missing_image_files": missing_images,
        "issue_counts": dict(sorted(issue_counts.items())),
        "issues": issues,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", type=Path, help="Dataset root containing images/ and annotations/")
    parser.add_argument("--min-gap", type=float, default=DEFAULT_MIN_GAP, help="Required bbox edge gap in pixels (default: 20)")
    parser.add_argument("--expected-images", type=int, help="Optional expected image and JSON count")
    parser.add_argument("--size-iqr-multiplier", type=float, default=1.5, help="Category bbox-area Tukey IQR multiplier")
    parser.add_argument(
        "--allowed-object-counts",
        type=int,
        nargs="+",
        default=[3, 4],
        help="Allowed annotation counts per image (default: 3 4)",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON report path; parent directory is created")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = validate_dataset(
            args.dataset_root,
            min_gap=args.min_gap,
            expected_images=args.expected_images,
            size_iqr_multiplier=args.size_iqr_multiplier,
            allowed_object_counts=tuple(args.allowed_object_counts),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
