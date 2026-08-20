from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO


def compute_map_75_95(metrics) -> float:
    """대회 기준 mAP@0.75:0.95를 Ultralytics all_ap에서 계산합니다."""
    all_ap = np.asarray(metrics.box.all_ap)

    if all_ap.ndim != 2:
        raise ValueError(f"all_ap shape 이상: {all_ap.shape}")

    if all_ap.shape[1] != 10:
        raise ValueError(
            "IoU 0.50~0.95, step 0.05 기준 10개 AP가 필요합니다. "
            f"현재 shape: {all_ap.shape}"
        )

    # columns: 0.50, 0.55, ..., 0.95
    return float(all_ap[:, 5:10].mean())


def evaluate_yolo(
    weights_path: str | Path,
    *,
    data_yaml: str | Path,
    split: str,
    imgsz: int,
    device: int | str,
    plots: bool = False,
    project: str | Path | None = None,
    name: str | None = None,
) -> dict:
    """best weight를 지정 split에서 평가하고 핵심 metric을 dict로 반환합니다."""
    model = YOLO(str(weights_path))

    kwargs = {
        "data": str(data_yaml),
        "split": split,
        "imgsz": imgsz,
        "device": device,
        "plots": plots,
    }

    if project is not None:
        kwargs["project"] = str(project)

    if name is not None:
        kwargs["name"] = name
        kwargs["exist_ok"] = True

    metrics = model.val(**kwargs)

    result = {
        "map_50_95": float(metrics.box.map),
        "map_50": float(metrics.box.map50),
        "map_75": float(metrics.box.map75),
        "map_75_95": compute_map_75_95(metrics),
        "save_dir": Path(metrics.save_dir),
        "metrics": metrics,
    }

    return result


def print_evaluation(result: dict, *, prefix: str = "eval") -> None:
    """평가 결과를 동일한 포맷으로 출력합니다."""
    print(f"[{prefix}] mAP@0.50:0.95 : {result['map_50_95']:.5f}")
    print(f"[{prefix}] mAP@0.50      : {result['map_50']:.5f}")
    print(f"[{prefix}] mAP@0.75      : {result['map_75']:.5f}")
    print(f"[{prefix}] mAP@0.75:0.95 : {result['map_75_95']:.5f}")
    print(f"[{prefix}] 저장 위치      : {result['save_dir']}")
