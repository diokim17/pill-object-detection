from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path) -> DictConfig:
    """OmegaConf YAML 설정을 로드합니다."""
    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"config 파일이 없습니다: {config_path}")

    return OmegaConf.load(config_path)


def set_seed(seed: int) -> None:
    """Python, NumPy, PyTorch seed를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> int | str:
    """Ultralytics YOLO에서 사용할 device 값을 반환합니다."""
    return 0 if torch.cuda.is_available() else "cpu"


def print_runtime_info(seed: int, device: int | str) -> None:
    """현재 실행 환경의 핵심 정보를 출력합니다."""
    print(
        "torch:", torch.__version__,
        "| CUDA:", torch.cuda.is_available(),
        "| device:", device,
        "| seed:", seed,
    )


def ensure_project_paths(cfg: Any) -> dict[str, Path]:
    """실험에서 공통으로 사용하는 경로를 Path 객체로 변환하고 검사합니다."""
    paths = {
        "project_root": Path(cfg.paths.project_root),
        "dataset_dir": Path(cfg.paths.yolo_dataset_dir),
        "test_image_dir": Path(cfg.paths.test_image_dir),
        "annotation_dir": Path(cfg.paths.annotation_dir),
        "weights_path": Path(cfg.paths.weights_path),
        "submission_path": Path(cfg.paths.submission_path),
    }

    required = ("project_root", "test_image_dir", "annotation_dir")

    for key in required:
        if not paths[key].exists():
            raise FileNotFoundError(f"{key} 경로가 없습니다: {paths[key]}")

    return paths
