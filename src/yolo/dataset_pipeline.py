from __future__ import annotations

from pathlib import Path
from typing import Any

from src import pill_transforms_cp as pt

from src.yolo.yolo_dataset import build_yolo_dataset


def configure_copy_paste_preprocess(
    *,
    use_white_balance: bool = False,
    use_clahe: bool = True,
) -> None:
    """Copy&Paste 데이터 생성/추론에 사용할 전처리 설정을 맞춥니다."""
    pt.USE_WHITE_BALANCE = use_white_balance
    pt.USE_CLAHE = use_clahe

    # pill_transforms_cp 구현에 존재하는 경우에만 재생성
    if hasattr(pt, "rebuild_preprocess"):
        pt.rebuild_preprocess()


def build_base_yolo_dataset(cfg: Any, *, clean: bool = True) -> Path:
    """원본 annotation을 YOLO train/val/test 구조로 변환합니다."""
    runtime_yaml = build_yolo_dataset(
        dataset_root=cfg.paths.dataset_root,
        output_root=cfg.paths.yolo_dataset_dir,
        image_dir_name=cfg.dataset.image_dir_name,
        annotation_dir_name=cfg.dataset.annotation_dir_name,
        label_offset=cfg.dataset.label_offset,
        strict=cfg.dataset.strict,
        validate_image_size=cfg.dataset.validate_image_size,
        train_ratio=cfg.dataset.train_ratio,
        val_ratio=cfg.dataset.val_ratio,
        test_ratio=cfg.dataset.test_ratio,
        random_seed=cfg.project.seed,
        clean=clean,
    )

    runtime_yaml = Path(runtime_yaml)

    if not runtime_yaml.is_file():
        raise FileNotFoundError(f"data.yaml 생성 실패: {runtime_yaml}")

    return runtime_yaml


def build_copy_paste_dataset(
    cfg: Any,
    *,
    src_root: str | Path,
    dst_root: str | Path,
    overwrite: bool = True,
    verbose: bool = True,
) -> Path:
    """기본 YOLO 데이터셋에서 Copy&Paste 증강 데이터셋을 생성합니다."""
    configure_copy_paste_preprocess(
        use_white_balance=False,
        use_clahe=True,
    )

    aug_yaml = pt.build_augmented_yolo_dataset(
        src_root=str(src_root),
        dst_root=str(dst_root),
        geom_mult=1,
        n_synth=cfg.augmentation.copy_paste.n_synth,
        cp_mode=cfg.augmentation.copy_paste.mode,
        max_crops_per_class=cfg.augmentation.copy_paste.max_crops_per_class,
        rebuild_crop_cache=cfg.augmentation.copy_paste.rebuild_crop_cache,
        seed=cfg.project.seed,
        preprocess_val_test=True,
        crops_dir=None,
        overwrite=overwrite,
        verbose=verbose,
    )

    aug_yaml = Path(aug_yaml)

    if not aug_yaml.is_file():
        raise FileNotFoundError(f"Copy&Paste data.yaml 생성 실패: {aug_yaml}")

    return aug_yaml


def validate_yolo_dataset(dataset_dir: str | Path) -> None:
    """YOLO images/labels train/val/test 디렉터리가 정상 생성됐는지 확인합니다."""
    dataset_dir = Path(dataset_dir)

    for split_name in ("train", "val", "test"):
        image_dir = dataset_dir / "images" / split_name
        label_dir = dataset_dir / "labels" / split_name

        if not image_dir.is_dir():
            raise FileNotFoundError(f"{split_name} image dir 없음: {image_dir}")

        if not label_dir.is_dir():
            raise FileNotFoundError(f"{split_name} label dir 없음: {label_dir}")


def prepare_yolo_datasets(
    cfg: Any,
    *,
    copy_paste: bool = True,
    cp_dataset_dir: str | Path | None = None,
    clean_base: bool = True,
) -> dict[str, Path]:
    """실험용 기본/Copy&Paste YOLO 데이터셋을 한 번에 준비합니다."""
    base_yaml = build_base_yolo_dataset(cfg, clean=clean_base)
    base_dir = Path(cfg.paths.yolo_dataset_dir)
    validate_yolo_dataset(base_dir)

    result = {
        "base_dir": base_dir,
        "base_yaml": base_yaml,
    }

    if not copy_paste:
        result["train_yaml"] = base_yaml
        return result

    if cp_dataset_dir is None:
        cp_dataset_dir = Path(cfg.paths.project_root) / "data" / "processed_cp"
    else:
        cp_dataset_dir = Path(cp_dataset_dir)

    cp_yaml = build_copy_paste_dataset(
        cfg,
        src_root=base_dir,
        dst_root=cp_dataset_dir,
    )
    validate_yolo_dataset(cp_dataset_dir)

    result.update(
        {
            "cp_dir": cp_dataset_dir,
            "cp_yaml": cp_yaml,
            "train_yaml": cp_yaml,
        }
    )

    return result
