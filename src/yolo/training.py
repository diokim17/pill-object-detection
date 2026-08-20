from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from ultralytics import YOLO, settings


def setup_wandb(cfg: Any):
    """W&B를 설정하고 run을 시작합니다. 비활성화 시 None을 반환합니다."""
    if not cfg.wandb.enabled:
        settings.update({"wandb": False})
        os.environ["WANDB_MODE"] = "disabled"
        return None

    import wandb

    settings.update({"wandb": True})
    os.environ["WANDB_MODE"] = cfg.wandb.mode

    wandb.login()

    return wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        entity=cfg.wandb.entity,
        job_type="train",
        tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def finish_wandb(run) -> None:
    """시작된 W&B run을 종료합니다."""
    if run is None:
        return

    import wandb
    wandb.finish()


def build_train_kwargs(
    cfg: Any,
    *,
    data_yaml: str | Path,
    device: int | str,
) -> dict:
    """실험 간 비교를 위해 YOLO train 인자를 한 곳에서 구성합니다."""
    project_root = Path(cfg.paths.project_root)

    return {
        "data": str(data_yaml),
        "epochs": cfg.train.epochs,
        "imgsz": cfg.train.imgsz,
        "batch": cfg.train.batch,
        "workers": cfg.train.workers,
        "seed": cfg.project.seed,
        "deterministic": True,
        "device": device,
        "project": str(project_root / "outputs" / "yolo"),
        "name": cfg.train.checkpoint_name,
        "exist_ok": True,

        # YOLO 내장 augmentation
        "hsv_h": cfg.yolo_augmentation.hsv_h,
        "hsv_s": cfg.yolo_augmentation.hsv_s,
        "hsv_v": cfg.yolo_augmentation.hsv_v,
        "degrees": cfg.yolo_augmentation.degrees,
        "translate": cfg.yolo_augmentation.translate,
        "scale": cfg.yolo_augmentation.scale,
        "shear": cfg.yolo_augmentation.shear,
        "perspective": cfg.yolo_augmentation.perspective,
        "flipud": cfg.yolo_augmentation.flipud,
        "fliplr": cfg.yolo_augmentation.fliplr,
        "mosaic": cfg.yolo_augmentation.mosaic,
        "mixup": cfg.yolo_augmentation.mixup,
        "copy_paste": cfg.yolo_augmentation.copy_paste,
        "erasing": cfg.yolo_augmentation.erasing,
    }


def train_yolo(
    cfg: Any,
    *,
    data_yaml: str | Path,
    device: int | str,
):
    """YOLO 모델을 학습하고 (model, results)를 반환합니다."""
    model = YOLO(cfg.model.name)
    run = setup_wandb(cfg)

    try:
        results = model.train(
            **build_train_kwargs(
                cfg,
                data_yaml=data_yaml,
                device=device,
            )
        )
    finally:
        finish_wandb(run)

    return model, results


def get_best_weights(results) -> Path:
    """Ultralytics train 결과에서 best.pt 경로를 반환합니다."""
    best_weights = Path(results.save_dir) / "weights" / "best.pt"

    if not best_weights.is_file():
        raise FileNotFoundError(f"best.pt가 없습니다: {best_weights}")

    return best_weights
