"""
pill_transforms.py
===================

경구약제(알약) Object Detection 프로젝트용 전처리 / 증강 통합 모듈입니다.

팀 노트북 4개가 모두 이 파일 하나만 import 하면 되도록 구성했습니다.

    ┌─────────────────────────────────────┬──────────────────────────────────┐
    │ 노트북                              │ 이 모듈에서 쓰는 것              │
    ├─────────────────────────────────────┼──────────────────────────────────┤
    │ pill_detection_dataset.ipynb        │ get_train_transforms()           │
    │ base_model_faster-rcnn_train.ipynb  │ get_train_transform()            │
    │                                     │ get_eval_transform()             │
    │ yolo11s_baseline.ipynb              │ build_augmented_yolo_dataset()   │
    │ 02_baseline.ipynb                   │ (Copy&Paste 원본 — 여기로 이식)  │
    └─────────────────────────────────────┴──────────────────────────────────┘

두 갈래 구조
------------
**[A] 온라인 transform** — PyTorch Dataset 에 붙여 매 배치마다 적용
      Albumentations 기반. `PillDetectionDataset(transforms=...)` 에 그대로 전달.
      Faster R-CNN 처럼 DataLoader 로 학습하는 모델용입니다.

**[B] 오프라인 증강** — 디스크에 증강된 YOLO 데이터셋을 미리 만들어 둠
      `cv2` + `numpy` 만 사용 (albumentations 불필요).
      Ultralytics YOLO 는 폴더를 통째로 읽으므로 이 방식이 맞습니다.
      `02_baseline.ipynb` 의 기하 증강 + Copy & Paste 를 그대로 이식했습니다.

기본값 (요청 반영)
------------------
- `DEFAULT_GEOM_MULT = 3`   → train 이미지 1장당 최종 3장 (원본 1 + 증강 2)
- `DEFAULT_N_SYNTH   = 600` → Copy & Paste 합성 이미지 600장
- **CLAHE 는 확률이 아니라 전 이미지에 항상 적용됩니다.**
  (train / val / test / 추론까지 동일 — 전처리를 학습에만 걸면 분포가 어긋납니다)
- 위 두 숫자는 `yolo11s_baseline.ipynb` 에서 인자로 덮어쓸 수 있습니다.

반영한 EDA / Dataset 특성
--------------------------
1. 원본 해상도가 976 x 1280 으로 전부 동일 → 종횡비를 왜곡하지 않는
   LongestMaxSize + Pad(letterbox) 전략
2. bbox 는 대부분 정사각형에 가깝고(AR 0.8~1.2 약 64%) 평균 area_ratio 약 5.6%
   → 작은 객체 보존을 위해 erosion_rate 를 낮게
3. 배경(back_color)·조명(light_color)이 전 샘플 동일 → 색상/명암 증강 필수
4. 촬영 각도 70/75/90도 → 온라인은 소각도, 오프라인은 알약 회전 불변성을 살려 ±180도
5. 클래스 불균형 51배 → Copy & Paste 로 희소 클래스 표본을 직접 늘림
6. 각인(print_front/back)이 클래스 정보 → **flip 기본 off**, 블러 최소

사용 예
-------
    # [A] Faster R-CNN / PillDetectionDataset
    from pill_transforms import get_train_transform, get_eval_transform
    train_transforms = get_train_transform(image_size=640)
    eval_transforms  = get_eval_transform(image_size=640)

    # [B] YOLO — 증강 데이터셋을 만들고 그 data.yaml 로 학습
    from pill_transforms import build_augmented_yolo_dataset
    aug_yaml = build_augmented_yolo_dataset(
        src_root="../data/processed",
        dst_root="../data/processed_aug",
        geom_mult=3,     # ← 노트북에서 조절
        n_synth=600,     # ← 노트북에서 조절
        # ★ Copy&Paste 재료를 미리 잘라 둔 크롭 폴더에서 가져오기
        crops_dir="/content/drive/MyDrive/.../cropped_pills_review",
    )

    # 또는 모듈 전역으로 한 번만 지정 (아래 '경로 설정' 블록 참고)
    import pill_transforms as pt
    pt.CROPPED_PILLS_DIR = "/content/drive/MyDrive/.../cropped_pills_review"
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

PathLike = Union[str, Path]


# ═══════════════════════════════════════════════════════════════════════════
#  Part 0. 기본 설정
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42

# ---------- ★ 노트북에서 덮어쓸 수 있는 두 값 ----------
DEFAULT_GEOM_MULT = 3     # train 원본 1장 → 최종 3장 (원본 1 + 증강 2)
DEFAULT_N_SYNTH = 600     # Copy & Paste 합성 이미지 수
# 200장 단위로 조절해주세요. (200->400->600)

# ---------- 전처리 (학습·평가·추론 전부 동일하게 적용) ----------
USE_WHITE_BALANCE = True   # Shades-of-Gray 화이트밸런스
USE_CLAHE = True           # ★ Lab 의 L 채널 CLAHE — 항상 켬(배경, 객체 구분)
CLAHE_CLIP = 3.0
CLAHE_GRID = 8

# ---------- 오프라인 기하 증강 ----------
ROT_LIMIT = 180            # 알약은 회전 불변이므로 크게
ROT_PROB = 0.8
SCALE_RANGE = (0.95, 1.05)
MIN_VISIBILITY = 0.2       # 잘린 뒤 원면적의 40% 미만이면 박스 삭제
USE_FLIP = False           # ★ 각인이 뒤집히므로 기본 off

P_BLUR = 0.20
P_NOISE = 0.30
P_TONE = 0.30
HSV_H_LIMIT = 8            # 색(color1)이 클래스 정보라 H 는 작게
HSV_S_LIMIT = 25
HSV_V_LIMIT = 30

# ---------- Copy & Paste ----------
PILLS_RANGE = (2, 4)       # 합성 이미지 1장에 붙일 알약 수 (원본 분포와 동일)
CP_EXTRA_RANGE = (1, 1)    # onto_train 모드에서 추가로 붙일 알약 수
CP_MODE = "mix"            # "synth" | "onto_train" | "mix"
CP_SYNTH_RATIO = 0.5       # mix 일 때 인공 배경 비율
CP_OVERLAP = 0.10          # 허용 겹침 비율
CP_SCALE_JIT = (0.95, 1.05)
CP_FEATHER = 2             # 경계 페더링(px)
SHADOW_ALPHA = (0.20, 0.55)
SHADOW_BLUR = (3, 12)
MAX_CROPS_PER_CLASS = 40   # 크롭 라이브러리 클래스당 최대 장수

# ═══════════════════════════════════════════════════════════════════════════
#  ★★★ 경로 설정 — cropped_pills_review 폴더 ★★★
# ═══════════════════════════════════════════════════════════════════════════
#  이미 잘라 놓은 알약 크롭 이미지를 Copy&Paste 재료로 사용합니다.
#  폴더 구조 (클래스별 하위 폴더):
#      cropped_pills_review/
#        ├─ 3351_일양하이트린정 2mg/
#        │    ├─ K-003351-013900-022074_0_2_0_2_75_000_200.png
#        │    └─ ...
#        └─ 1234_○○○정 5mg/
#             └─ ...
#
#  ▸ None  → 기존 동작(train 라벨 박스에서 직접 컷아웃)
#  ▸ 경로  → 이 폴더의 크롭 이미지를 사용
#
#  Colab 예시:
#     CROPPED_PILLS_DIR = "/content/drive/MyDrive/pill_project/cropped_pills_review"
#  Windows 예시:
#     CROPPED_PILLS_DIR = r"D:\pill_project\cropped_pills_review"
#
#  노트북에서 덮어쓰려면:
#     import pill_transforms as pt
#     pt.CROPPED_PILLS_DIR = "/content/drive/MyDrive/.../cropped_pills_review"
#  또는 함수 인자로:
#     build_augmented_yolo_dataset(..., crops_dir=".../cropped_pills_review")
# ═══════════════════════════════════════════════════════════════════════════

CROPPED_PILLS_DIR: Optional[str] = None

# 크롭 폴더에서 읽을 이미지 확장자
CROPPED_PILLS_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# 크롭 이미지에 이미 알파(투명) 채널이 있으면 배경 제거를 건너뛰고 그대로 사용
CROPPED_USE_ALPHA = True

# ---------- Cutout (그림자 배제) ----------
CUT_CHROMA_K = 2.0
CUT_MIN_CHROMA = 6.0
CUT_L_K = 1.0
CUT_SHRINK = 1
CUT_USE_GRABCUT_FALLBACK = True

# ---------- 정규화 상수 ----------
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# PillDetectionDataset 이 기대하는 bbox 포맷
BBOX_FORMAT = "pascal_voc"   # [x1, y1, x2, y2] 절대 픽셀
LABEL_FIELDS = ["labels"]


# ═══════════════════════════════════════════════════════════════════════════
#  Part 1. 한글 경로 안전 I/O  (Windows·Colab 공용)
# ═══════════════════════════════════════════════════════════════════════════

def imread_unicode(path: PathLike) -> Optional[np.ndarray]:
    """한글이 섞인 경로도 읽을 수 있는 cv2.imread 대체 (BGR 반환)."""
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def imread_unicode_unchanged(path: PathLike) -> Optional[np.ndarray]:
    """알파 채널까지 읽습니다 (크롭 캐시 RGBA 용)."""
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def imwrite_unicode(path: PathLike, img: np.ndarray) -> bool:
    """한글이 섞인 경로에도 저장할 수 있는 cv2.imwrite 대체."""
    ext = os.path.splitext(str(path))[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  Part 2. 전처리 — 화이트밸런스 + CLAHE (★ 항상 적용)
# ═══════════════════════════════════════════════════════════════════════════

class Preprocess:
    """Shades-of-Gray 화이트밸런스 + Lab 의 L 채널 CLAHE.

    ★ 가장 흔한 실수: 학습에만 전처리를 걸고 추론에 안 거는 것.
      이 클래스 하나를 train / val / test / 추론에 **전부** 사용하세요.

    - 화이트밸런스: 채널별 p-norm 으로 색 캐스트를 제거합니다.
      배경색·조명색이 알약 색(color1, 클래스 정보)으로 새는 것을 막습니다.
    - CLAHE: 밝기(L)만 국소 평탄화합니다. 알약과 그림자의 명암 대비를 벌립니다.
      Grayscale·Retinex 는 색 정보를 잃으므로 쓰지 않습니다.
    """

    def __init__(
        self,
        white_balance: bool = USE_WHITE_BALANCE,
        clahe: bool = USE_CLAHE,
        clip: float = CLAHE_CLIP,
        grid: int = CLAHE_GRID,
    ):
        self.white_balance = bool(white_balance)
        self.clahe = bool(clahe)
        self.clip = float(clip)
        self.grid = int(grid)

    @staticmethod
    def shades_of_gray(img_bgr: np.ndarray, p: int = 6) -> np.ndarray:
        f = img_bgr.astype(np.float32)
        norm = np.power(np.power(f, p).mean(axis=(0, 1)), 1.0 / p)
        norm = np.maximum(norm, 1e-6)
        return np.clip(f * (norm.mean() / norm), 0, 255).astype(np.uint8)

    def clahe_on_L(self, img_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(
            clipLimit=self.clip, tileGridSize=(self.grid, self.grid)
        ).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def apply_image(self, img_bgr: np.ndarray) -> np.ndarray:
        out = img_bgr
        if self.white_balance:
            out = self.shades_of_gray(out)
        if self.clahe:
            out = self.clahe_on_L(out)
        return out

    __call__ = apply_image


# 전역 인스턴스 — 학습 인코딩과 추론이 이 하나를 공유합니다.
PREPROCESS = Preprocess()


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """학습 인코딩과 추론에서 반드시 동일하게 호출할 것 (BGR in / BGR out)."""
    return PREPROCESS.apply_image(img_bgr)


def rebuild_preprocess() -> Preprocess:
    """★ 노트북에서 USE_WHITE_BALANCE / USE_CLAHE / CLAHE_CLIP / CLAHE_GRID 를
    바꾼 뒤 반드시 호출하세요. 전역 PREPROCESS(= preprocess() 가 쓰는 싱글턴)를
    현재 값으로 다시 만듭니다. (build_augmented_yolo_dataset() 는 내부에서
    항상 새로 만들어 쓰므로 이 호출이 필요 없습니다 — get_train_transform 등
    Faster R-CNN 경로에서만 필요합니다.)

    사용 예:
        import pill_transforms as pt
        pt.CLAHE_CLIP = 3.0
        pt.rebuild_preprocess()
    """
    global PREPROCESS
    PREPROCESS = Preprocess(
        white_balance=USE_WHITE_BALANCE, clahe=USE_CLAHE,
        clip=CLAHE_CLIP, grid=CLAHE_GRID,
    )
    return PREPROCESS


# ═══════════════════════════════════════════════════════════════════════════
#  Part 3. [A] 온라인 transform — Albumentations
#          (PillDetectionDataset / Faster R-CNN 용)
# ═══════════════════════════════════════════════════════════════════════════
#
# albumentations 는 여기서만 필요합니다. 설치돼 있지 않아도 Part 4~6
# (오프라인 증강 · Copy&Paste · YOLO 빌더)은 정상 동작합니다.

try:
    import albumentations as A

    HAS_ALBUMENTATIONS = True
except ImportError:  # pragma: no cover
    A = None
    HAS_ALBUMENTATIONS = False

# ToTensorV2 는 torch 를 필요로 하므로 별도로 import 합니다.
# (torch 가 없는 환경에서도 to_tensor=False 로 파이프라인을 쓸 수 있게)
try:
    from albumentations.pytorch import ToTensorV2

    HAS_TOTENSOR = True
except ImportError:  # pragma: no cover
    ToTensorV2 = None
    HAS_TOTENSOR = False

_ALB_HELP = (
    "이 함수는 albumentations 패키지가 필요합니다.\n"
    "  Colab : !pip install -q albumentations\n"
    "  로컬  : pip install albumentations\n"
    "※ YOLO 오프라인 증강(build_augmented_yolo_dataset)은 설치 없이도 동작합니다."
)


def _require_albumentations() -> None:
    if not HAS_ALBUMENTATIONS:
        raise ImportError(_ALB_HELP)


def _require_totensor() -> None:
    if not HAS_TOTENSOR:
        raise ImportError(
            "to_tensor=True 는 torch 가 필요합니다 "
            "(albumentations.pytorch.ToTensorV2).\n"
            "torch 를 설치하거나 to_tensor=False 로 호출하세요."
        )


def _pad_transform(image_size: int):
    """PadIfNeeded 의 인자명이 albumentations 버전마다 달라 둘 다 시도합니다."""
    common = dict(
        min_height=image_size,
        min_width=image_size,
        border_mode=cv2.BORDER_CONSTANT,
    )
    try:
        return A.PadIfNeeded(**common, fill=0)          # 1.4.21+
    except TypeError:
        return A.PadIfNeeded(**common, value=0)         # 구버전


def _rotate_transform(limit: float, p: float):
    common = dict(limit=limit, border_mode=cv2.BORDER_CONSTANT, p=p)
    try:
        return A.Rotate(**common, fill=0)
    except TypeError:
        return A.Rotate(**common, value=0)


def _bbox_params(min_visibility: float = 0.2, min_area: float = 4.0):
    try:
        return A.BboxParams(
            format=BBOX_FORMAT,
            label_fields=LABEL_FIELDS,
            min_visibility=min_visibility,
            min_area=min_area,
            clip=True,
        )
    except TypeError:  # clip 인자가 없는 구버전
        return A.BboxParams(
            format=BBOX_FORMAT,
            label_fields=LABEL_FIELDS,
            min_visibility=min_visibility,
            min_area=min_area,
        )


def get_train_transforms(
    image_size: int = 640,   # 원본 976x1280 → 640 정사각 letterbox
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    min_visibility: float = 0.2,
    to_tensor: bool = True,
    hflip_p: float = 0.0,    # ★ 각인 보존을 위해 기본 off. 필요하면 0.5
    vflip_p: float = 0.0,
):
    """학습용 전처리 + 증강 파이프라인 (Albumentations).

    구성 순서
    ---------
    1. RandomSizedBBoxSafeCrop        : bbox 를 보존하면서 스케일 변화를 학습
    2. LongestMaxSize + PadIfNeeded   : 976x1280 비율을 왜곡 없이 정사각 캔버스에
    3. HorizontalFlip / VerticalFlip  : 기본 off (각인이 뒤집히면 클래스 정보 손상)
    4. Rotate(소각도)                 : 70/75/90도 촬영 각도 변화를 모사
    5. ★ CLAHE(p=1.0)                : **모든 이미지에 항상 적용**
    6. OneOf(밝기 / 컬러)             : 항상 동일한 배경·조명 한계를 보완
    7. OneOf(GaussNoise / MotionBlur) : 촬영 노이즈·흔들림 대비
    8. Normalize + ToTensorV2         : 모델 입력 형태로 변환

    ★ 5번이 OneOf 밖으로 나온 이유
      CLAHE 는 "증강"이 아니라 **전처리**입니다. 학습 이미지 일부에만 걸리면
      val/test(항상 적용)와 분포가 어긋나므로 p=1.0 으로 전부 적용합니다.
      밝기·컬러 변형만 OneOf 로 묶어 색 왜곡이 누적되지 않게 했습니다.
    """
    _require_albumentations()

    transforms = [
        A.RandomSizedBBoxSafeCrop(
            height=image_size,
            width=image_size,
            erosion_rate=0.1,
            p=0.5,
        ),
        A.LongestMaxSize(max_size=image_size),
        _pad_transform(image_size),

        # 상하좌우 반전 — 각인 때문에 기본 비활성
        A.HorizontalFlip(p=hflip_p),
        A.VerticalFlip(p=vflip_p),

        # 회전 (소각도)
        _rotate_transform(limit=15, p=0.4),

        # ★ CLAHE — 전 이미지 적용 (전처리이므로 확률 없음)
        A.CLAHE(clip_limit=CLAHE_CLIP, tile_grid_size=(CLAHE_GRID, CLAHE_GRID), p=1.0),

        # 밝기 / 컬러는 둘 중 하나만 (누적되면 color1 정보가 왜곡됨)
        A.OneOf(
            [
                # 밝기
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0,
                ),
                # 컬러
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=15,
                    p=1.0,
                ),
            ],
            p=0.6,
        ),
        A.OneOf(
            [
                A.GaussNoise(p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ],
            p=0.2,
        ),
    ]

    if to_tensor:
        _require_totensor()
        transforms += [A.Normalize(mean=mean, std=std), ToTensorV2()]

    return A.Compose(
        transforms, bbox_params=_bbox_params(min_visibility=min_visibility)
    )


def get_valid_transforms(
    image_size: int = 640,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    to_tensor: bool = True,
):
    """검증 / 추론용 파이프라인.

    증강은 없지만 **CLAHE 는 학습과 동일하게 적용**합니다.
    (학습에만 전처리를 걸면 학습·평가 분포가 어긋납니다)

    Test 842장도 train 과 같은 976x1280 이라 크기 보정 없이 그대로 씁니다.
    """
    _require_albumentations()

    transforms = [
        A.LongestMaxSize(max_size=image_size),
        _pad_transform(image_size),
        # ★ 학습과 동일한 CLAHE
        A.CLAHE(clip_limit=CLAHE_CLIP, tile_grid_size=(CLAHE_GRID, CLAHE_GRID), p=1.0),
    ]

    if to_tensor:
        _require_totensor()
        transforms += [A.Normalize(mean=mean, std=std), ToTensorV2()]

    return A.Compose(
        transforms, bbox_params=_bbox_params(min_visibility=0.0, min_area=0.0)
    )


def get_test_transforms(
    image_size: int = 640,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    to_tensor: bool = True,
):
    """`get_valid_transforms` 의 별칭 (bbox 정답이 없어도 labels=[] 로 사용 가능)."""
    return get_valid_transforms(
        image_size=image_size, mean=mean, std=std, to_tensor=to_tensor
    )


# ---- base_model_faster-rcnn_train.ipynb 가 기대하는 단수형 이름 (별칭) ----
get_train_transform = get_train_transforms
get_eval_transform = get_valid_transforms
get_valid_transform = get_valid_transforms
get_test_transform = get_test_transforms


class SafeAlbumentationsTransform:
    """bbox 가 전부 사라지는 경우를 방지하는 wrapper.

    RandomSizedBBoxSafeCrop, Rotate 등은 이론상 모든 bbox 를 제거할 수 있습니다.
    `PillDetectionDataset._apply_transforms` 는 이 경우 빈 target 을 그대로
    반환하므로, 객체 0개 샘플을 피하고 싶다면 이 wrapper 로 감싸세요.

        train_transforms = SafeAlbumentationsTransform(get_train_transform())
    """

    def __init__(self, transform, max_retries: int = 3):
        self.transform = transform
        self.max_retries = int(max_retries)

    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[Sequence[Sequence[float]]] = None,
        labels: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        bboxes = list(bboxes) if bboxes is not None else []
        labels = list(labels) if labels is not None else []
        last_result: Optional[Dict[str, Any]] = None

        for _ in range(self.max_retries):
            result = self.transform(image=image, bboxes=bboxes, labels=labels)
            last_result = result
            if len(result["bboxes"]) > 0 or len(bboxes) == 0:
                return result

        # 재시도 후에도 전멸했다면 마지막 결과를 그대로 반환합니다.
        return last_result  # type: ignore[return-value]


def denormalize(
    tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
):
    """정규화된 (C, H, W) 텐서를 시각화용 (H, W, C) uint8 배열로 되돌립니다."""
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("denormalize 는 torch.Tensor 입력을 기대합니다.")

    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)

    image = tensor.detach().cpu() * std_t + mean_t
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return (image * 255).round().astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
#  Part 4. [B] 오프라인 기하 증강 — cv2 + numpy (02_baseline 이식)
# ═══════════════════════════════════════════════════════════════════════════
#
#  박스를 **중심 좌표 (cx, cy, w, h)** 로 다룹니다. 회전·스케일 계산이 간단해집니다.
#  저장할 때만 좌상단 기준으로 되돌립니다.

Box = Tuple[float, float, float, float, int]   # (cx, cy, w, h, class_index)


@dataclass
class Sample:
    """증강 파이프라인이 주고받는 단위."""

    image: np.ndarray                              # BGR uint8
    boxes: List[Box] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "Sample":
        return Sample(self.image.copy(), list(self.boxes), dict(self.meta))

    @property
    def hw(self) -> Tuple[int, int]:
        return self.image.shape[:2]

    def boxes_xywh(self) -> List[Tuple[float, float, float, float, int]]:
        """좌상단 기준 [(x, y, w, h, cls)] — 저장용."""
        return [(cx - w / 2, cy - h / 2, w, h, c) for cx, cy, w, h, c in self.boxes]

    @staticmethod
    def from_xywh(image, boxes_xywh, meta=None) -> "Sample":
        return Sample(
            image,
            [(x + w / 2, y + h / 2, w, h, int(c)) for x, y, w, h, c in boxes_xywh],
            dict(meta or {}),
        )


class Transform:
    """모든 오프라인 변환의 기반. p 확률로 apply 를 수행합니다."""

    def __init__(self, p: float = 1.0):
        self.p = float(p)

    def apply(self, s: Sample, rng: random.Random) -> Sample:
        raise NotImplementedError

    def __call__(self, s: Sample, rng: Optional[random.Random] = None) -> Sample:
        rng = rng or random
        if self.p >= 1.0 or rng.random() < self.p:
            return self.apply(s, rng)
        return s

    def __repr__(self):
        args = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        inner = ", ".join(f"{k}={v}" for k, v in args.items())
        return f"{type(self).__name__}({inner})"


class Compose(Transform):
    """변환을 순서대로 적용."""

    def __init__(self, transforms: Sequence[Transform]):
        super().__init__(p=1.0)
        self.transforms = list(transforms)

    def apply(self, s, rng):
        for t in self.transforms:
            s = t(s, rng)
        return s

    def __repr__(self):
        body = "\n".join("  " + repr(t) for t in self.transforms)
        return f"Compose([\n{body}\n])"


class OneOf(Transform):
    """후보 중 하나만 적용 (색상 변형이 누적되지 않게)."""

    def __init__(self, transforms: Sequence[Transform], p: float = 1.0):
        super().__init__(p)
        self.transforms = list(transforms)

    def apply(self, s, rng):
        return rng.choice(self.transforms).apply(s, rng)


class PhotometricTransform(Transform):
    """화소값만 바꾸는 변환 — 박스는 그대로."""

    def apply(self, s, rng):
        s.image = self.apply_image(s.image, rng)
        return s

    def apply_image(self, img, rng):
        raise NotImplementedError


# ─────────────────────────────────── 화소값 변환
class BrightnessContrast(PhotometricTransform):
    def __init__(self, b_lim=0.30, c_lim=0.30, p=0.7):
        super().__init__(p)
        self.b_lim, self.c_lim = b_lim, c_lim

    def apply_image(self, img, rng):
        a = 1.0 + rng.uniform(-self.c_lim, self.c_lim)
        beta = rng.uniform(-self.b_lim, self.b_lim) * 255.0
        return cv2.convertScaleAbs(img, alpha=a, beta=beta)


class Gamma(PhotometricTransform):
    def __init__(self, lo=60, hi=140, p=0.5):
        super().__init__(p)
        self.lo, self.hi = lo, hi

    def apply_image(self, img, rng):
        g = rng.uniform(self.lo, self.hi) / 100.0
        lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / g)) * 255.0, 0, 255).astype(np.uint8)
        return cv2.LUT(img, lut)


class HSVJitter(PhotometricTransform):
    """★ h_lim 을 작게 유지합니다 — 색(color1)은 클래스 정보입니다."""

    def __init__(self, h_lim=HSV_H_LIMIT, s_lim=HSV_S_LIMIT, v_lim=HSV_V_LIMIT, p=0.7):
        super().__init__(p)
        self.h_lim, self.s_lim, self.v_lim = h_lim, s_lim, v_lim

    def apply_image(self, img, rng):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        if self.h_lim > 0:
            hsv[..., 0] = (hsv[..., 0] + rng.randint(-self.h_lim, self.h_lim)) % 180
        if self.s_lim > 0:
            hsv[..., 1] = np.clip(hsv[..., 1] + rng.randint(-self.s_lim, self.s_lim), 0, 255)
        if self.v_lim > 0:
            hsv[..., 2] = np.clip(hsv[..., 2] + rng.randint(-self.v_lim, self.v_lim), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


class ToneCurve(PhotometricTransform):
    def __init__(self, scale=0.3, p=P_TONE):
        super().__init__(p)
        self.scale = scale

    def apply_image(self, img, rng):
        ly = float(np.clip(rng.gauss(0.25, self.scale), 0.01, 0.99))
        hy = float(np.clip(rng.gauss(0.75, self.scale), 0.01, 0.99))
        if hy < ly:
            ly, hy = hy, ly
        lut = np.clip(
            np.interp(np.arange(256) / 255.0, [0.0, 0.25, 0.75, 1.0], [0.0, ly, hy, 1.0]) * 255.0,
            0, 255,
        ).astype(np.uint8)
        return cv2.LUT(img, lut)


class ISONoise(PhotometricTransform):
    def __init__(self, color=(0.01, 0.05), inten=(0.1, 0.5), p=P_NOISE):
        super().__init__(p)
        self.color, self.inten = color, inten

    def apply_image(self, img, rng):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
        hls[..., 1] = np.clip(
            hls[..., 1] + np.random.normal(0, rng.uniform(*self.inten) * 12.75, hls.shape[:2]),
            0, 255,
        )
        hls[..., 0] = np.clip(
            hls[..., 0] + np.random.normal(0, rng.uniform(*self.color) * 180, hls.shape[:2]),
            0, 179,
        )
        return cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2BGR)


class MotionBlur(PhotometricTransform):
    """★ 확률을 낮게 유지합니다 — 각인이 뭉개지면 복구되지 않습니다."""

    def __init__(self, ksize=(3, 5), p=P_BLUR):
        super().__init__(p)
        self.ksize = ksize

    def apply_image(self, img, rng):
        k = rng.choice(list(self.ksize))
        ker = np.zeros((k, k), np.float32)
        c = (k - 1) / 2.0
        ang = rng.uniform(0, 180)
        dx, dy = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        for i in range(k):
            t = i - c
            x, y = int(round(c + dx * t)), int(round(c + dy * t))
            ker[np.clip(y, 0, k - 1), np.clip(x, 0, k - 1)] = 1.0
        return cv2.filter2D(img, -1, ker / ker.sum())


class RandomShadow(PhotometricTransform):
    """★ '그림자는 알약과 무관하게 붙었다 떨어졌다 한다'는 신호를 줍니다."""

    def __init__(self, n=(1, 3), n_vert=5, alpha=(0.35, 0.65), p=0.5):
        super().__init__(p)
        self.n, self.n_vert, self.alpha = n, n_vert, alpha

    def apply_image(self, img, rng):
        H, W = img.shape[:2]
        out = img.astype(np.float32)
        for _ in range(rng.randint(*self.n)):
            pts = np.array(
                [[rng.randint(0, W - 1), rng.randint(0, H - 1)] for _ in range(self.n_vert)],
                np.int32,
            )
            m = np.zeros((H, W), np.float32)
            cv2.fillPoly(m, [cv2.convexHull(pts)], 1.0)
            k = (max(3, (min(H, W) // 40) | 1),) * 2
            m = cv2.GaussianBlur(m, k, 0)
            out *= (1.0 - rng.uniform(*self.alpha) * m[..., None])
        return np.clip(out, 0, 255).astype(np.uint8)


# ─────────────────────────────────── 기하 변환
class RotateScale(Transform):
    """★ 박스는 내접 타원을 회전시켜 계산합니다.

    회전한 박스를 축정렬 외접 사각형으로 감싸면 길쭉한 알약을 45도 돌렸을 때
    박스 면적이 2.7배로 부풉니다. 대신 박스에 내접한 타원을 회전시켜
    그 외접 사각형을 씁니다. 알약은 대부분 타원·원형이라 이 근사가 정확합니다.

        반너비 = sqrt((a·cosθ)² + (b·sinθ)²)      a = w/2
        반높이 = sqrt((a·sinθ)² + (b·cosθ)²)      b = h/2
    """

    def __init__(self, limit=ROT_LIMIT, scale=SCALE_RANGE, p=ROT_PROB):
        super().__init__(p)
        self.limit, self.scale = limit, scale

    def apply(self, s, rng):
        H, W = s.hw
        ang = rng.uniform(-self.limit, self.limit)
        sc = rng.uniform(*self.scale)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, sc)
        s.image = cv2.warpAffine(
            s.image, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

        t = np.deg2rad(ang)
        ct, st_ = np.cos(t), np.sin(t)
        nb = []
        for cx, cy, w, h, cid in s.boxes:
            a, b = w * sc / 2.0, h * sc / 2.0
            hw = float(np.hypot(a * ct, b * st_))
            hh = float(np.hypot(a * st_, b * ct))
            dx, dy = cx - W / 2, cy - H / 2
            nb.append(
                (
                    W / 2 + (dx * ct + dy * st_) * sc,
                    H / 2 + (-dx * st_ + dy * ct) * sc,
                    hw * 2,
                    hh * 2,
                    cid,
                )
            )
        s.boxes = nb
        return s


class Flip(Transform):
    """⚠️ 각인이 뒤집히므로 알약에는 기본적으로 쓰지 않습니다."""

    def __init__(self, horizontal=True, p=0.5):
        super().__init__(p)
        self.horizontal = horizontal

    def apply(self, s, rng):
        H, W = s.hw
        if self.horizontal:
            s.image = cv2.flip(s.image, 1)
            s.boxes = [(W - cx, cy, w, h, c) for cx, cy, w, h, c in s.boxes]
        else:
            s.image = cv2.flip(s.image, 0)
            s.boxes = [(cx, H - cy, w, h, c) for cx, cy, w, h, c in s.boxes]
        return s


class ClipBoxes(Transform):
    """이미지 밖으로 나간 박스를 자르고, 너무 많이 잘린 박스는 삭제합니다."""

    def __init__(self, min_visibility=MIN_VISIBILITY, min_side=4.0):
        super().__init__(p=1.0)
        self.min_visibility, self.min_side = min_visibility, min_side

    def apply(self, s, rng):
        H, W = s.hw
        keep = []
        for cx, cy, w, h, cid in s.boxes:
            x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
            x2, y2 = min(float(W), cx + w / 2), min(float(H), cy + h / 2)
            nw, nh = x2 - x1, y2 - y1
            if nw < self.min_side or nh < self.min_side:
                continue
            if nw * nh < self.min_visibility * w * h:
                continue
            keep.append((x1 + nw / 2, y1 + nh / 2, nw, nh, cid))
        s.boxes = keep
        return s


class GeometricAugmentor:
    """train 원본을 `multiplier` 배로 늘립니다.

    Args:
        multiplier: 최종 배수. 3 이면 원본 1장 + 증강 2장.
        use_flip:   좌우/상하 반전 사용 여부 (각인 때문에 기본 False)
    """

    def __init__(self, multiplier: int = DEFAULT_GEOM_MULT, use_flip: bool = USE_FLIP):
        if multiplier < 1:
            raise ValueError("multiplier 는 1 이상이어야 합니다.")
        self.multiplier = int(multiplier)
        self.use_flip = bool(use_flip)
        self.pipeline = self.build()

    def build(self) -> Compose:
        ts: List[Transform] = [
            RandomShadow(p=0.5),
            BrightnessContrast(p=0.7),
            Gamma(p=0.5),
            HSVJitter(h_lim=HSV_H_LIMIT, s_lim=HSV_S_LIMIT, v_lim=HSV_V_LIMIT, p=0.7),
            ToneCurve(p=P_TONE),
            ISONoise(p=P_NOISE),
            MotionBlur(p=P_BLUR),
        ]
        if self.use_flip:
            ts += [Flip(horizontal=True, p=0.5), Flip(horizontal=False, p=0.5)]
        ts += [
            RotateScale(limit=ROT_LIMIT, scale=SCALE_RANGE, p=ROT_PROB),
            ClipBoxes(MIN_VISIBILITY),
        ]
        return Compose(ts)

    @property
    def n_extra(self) -> int:
        """이미지 1장당 추가로 만들 증강본 수."""
        return self.multiplier - 1

    def generate(self, base: Sample, n: int, rng: random.Random) -> List[Sample]:
        """base 로부터 증강본 n 장을 만듭니다. 박스가 전멸한 결과는 버립니다."""
        out = []
        for _ in range(n):
            s = self.pipeline(base.clone(), rng)
            if s.boxes:
                out.append(s)
        return out


# ═══════════════════════════════════════════════════════════════════════════
#  Part 5. Copy & Paste — 그림자 상관관계 절단 (02_baseline 이식)
# ═══════════════════════════════════════════════════════════════════════════
#
#  1. 알약을 마스크로 컷아웃          → 그림자가 물리적으로 제거됨
#  2. 새 배경에 랜덤 위치·회전으로 붙임
#  3. 그림자를 별도 레이어로 합성      → 방향·길이·농도·흐림 전부 랜덤
#
#  ★ 그림자는 배경과 색은 같고 밝기만 어둡습니다. 밝기로 가르면 그림자가
#    알약으로 오인됩니다. Lab 의 (a, b) 색상 거리로 판별하면 뒤집힙니다.


def _largest_center_component(m: np.ndarray, h: int, w: int, shrink: int):
    """가장 크고 중앙에 가까운 연결 성분만 남기고 외곽선을 채웁니다."""
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    cy, cx = h / 2, w / 2
    best, bs = None, -1e18
    for i in range(1, n):
        a_ = st[i, cv2.CC_STAT_AREA]
        if a_ < 0.05 * h * w:
            continue
        ys, xs = np.where(lbl == i)
        s = a_ - np.hypot(ys.mean() - cy, xs.mean() - cx) * max(h, w)
        if s > bs:
            best, bs = i, s
    if best is None:
        return None
    mask = (lbl == best).astype(np.uint8)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.006 * cv2.arcLength(c, True), True)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [c], -1, 1, cv2.FILLED)

    if shrink > 0:
        filled = cv2.erode(filled, np.ones((3, 3), np.uint8), iterations=shrink)

    if not (0.06 * h * w < filled.sum() < 0.99 * h * w):
        return None
    return filled


def cutout_chroma(
    crop: np.ndarray,
    chroma_k: float = CUT_CHROMA_K,
    min_chroma: float = CUT_MIN_CHROMA,
    l_k: float = CUT_L_K,
    shrink: int = CUT_SHRINK,
):
    """Lab 색상거리 기반 컷아웃. 그림자를 명시적으로 배제. 실패하면 None."""
    h, w = crop.shape[:2]
    if h < 12 or w < 12:
        return None

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, ab = lab[..., 0], lab[..., 1:]

    # 테두리 링을 배경 표본으로 사용
    b = max(2, min(h, w) // 10)
    ring_ab = np.concatenate(
        [ab[:b].reshape(-1, 2), ab[-b:].reshape(-1, 2),
         ab[:, :b].reshape(-1, 2), ab[:, -b:].reshape(-1, 2)]
    )
    ring_L = np.concatenate([L[:b].ravel(), L[-b:].ravel(), L[:, :b].ravel(), L[:, -b:].ravel()])
    bg_ab = np.median(ring_ab, axis=0)
    bg_L = float(np.median(ring_L))
    sd_L = float(np.std(ring_L)) + 1e-6

    d = cv2.medianBlur(np.linalg.norm(ab - bg_ab, axis=2).astype(np.float32), 5)
    dL = cv2.medianBlur((L - bg_L).astype(np.float32), 5)

    noise = float(np.median(np.linalg.norm(ring_ab - bg_ab, axis=1)))
    thr_c = max(min_chroma, noise * chroma_k)
    thr_l = sd_L * l_k

    shadow = (dL < -thr_l) & (d < thr_c)          # 어둡고 + 색은 배경과 같음
    fg = ((d > thr_c) | (dL > thr_l)) & ~shadow

    m = fg.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return _largest_center_component(m, h, w, shrink)


def cutout_grabcut(crop: np.ndarray, margin=0.08, iters=4, shrink=CUT_SHRINK):
    """색상 대비가 없는 흰 알약용 fallback. 실패하면 None."""
    h, w = crop.shape[:2]
    if h < 24 or w < 24:
        return None
    mx, my = int(w * margin), int(h * margin)
    rect = (mx, my, max(1, w - 2 * mx), max(1, h - 2 * my))
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return _largest_center_component(m, h, w, shrink)


def cutout_pill(crop: np.ndarray):
    """색상 기반 → 실패 시 GrabCut. 둘 다 실패하면 None."""
    m = cutout_chroma(crop)
    if m is None and CUT_USE_GRABCUT_FALLBACK:
        m = cutout_grabcut(crop)
    return m


# ---------------------------------------------------------------------------
#  cropped_pills_review 폴더명 ↔ data.yaml 클래스명 매칭 헬퍼
# ---------------------------------------------------------------------------

def _norm_key(s: str) -> str:
    """비교용 정규화 — 공백·기호 제거 후 소문자."""
    return re.sub(r"[\s_\-\.\(\)\[\]/,]", "", str(s)).lower()


def _key_candidates(s: str) -> List[str]:
    """'3351_일양하이트린정 2mg' → ['3351일양하이트린정2mg', '일양하이트린정2mg', '3351']"""
    s = str(s).strip()
    cands = [s]
    if "_" in s:
        head, tail = s.split("_", 1)
        cands.append(tail)
        if head.strip().isdigit():
            cands.append(str(int(head)))
    nums = re.findall(r"\d+", s)
    if nums:
        longest = max(nums, key=len)
        cands.append(str(int(longest)))
    out, seen = [], set()
    for c in cands:
        k = _norm_key(c)
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _build_name_lookup(names: Sequence[str]) -> Dict[str, int]:
    """클래스명 리스트 → {정규화키: class_id}"""
    lut: Dict[str, int] = {}
    for i, n in enumerate(names):
        for k in _key_candidates(n):
            lut.setdefault(k, i)
    return lut


def _match_class_id(folder_name: str, lut: Dict[str, int]) -> Optional[int]:
    """폴더명을 클래스 id 로 변환. 못 찾으면 None."""
    for k in _key_candidates(folder_name):
        if k in lut:
            return lut[k]
    return None


class PillCropLibrary:
    """YOLO train 이미지의 GT 박스에서 알약을 컷아웃해 클래스별로 보관합니다.

    ★ `crops_dir`(= CROPPED_PILLS_DIR) 이 지정되면 컷아웃 대신
      이미 잘려 있는 `cropped_pills_review` 폴더의 이미지를 재료로 씁니다.

    `02_baseline.ipynb` 는 `01_eda` 가 만든 `crop_pool.json` 을 읽었지만,
    여기서는 **train 라벨에서 직접** 크롭하므로 별도 준비물이 없습니다.
    캐시는 RGBA png 로 저장하므로 두 번째 실행부터는 컷아웃을 건너뜁니다.
    """

    def __init__(
        self,
        cache_dir: PathLike,
        max_per_class: int = MAX_CROPS_PER_CLASS,
        margin: float = 0.12,
        crops_dir: Optional[PathLike] = None,   # ★ cropped_pills_review 경로
    ):
        self.cache_dir = Path(cache_dir)
        self.max_per_class = int(max_per_class)
        self.margin = float(margin)
        # ★ 인자 > 전역 CROPPED_PILLS_DIR 순으로 결정
        _cd = crops_dir if crops_dir is not None else CROPPED_PILLS_DIR
        self.crops_dir: Optional[Path] = Path(_cd) if _cd else None
        self.unmatched_folders: List[str] = []
        self.items: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
        self.box_stats: Dict[int, Dict[str, float]] = {}
        self.global_box: Dict[str, float] = {"w_med": 120.0, "h_med": 120.0}
        self.stats = Counter()
        self.fail_by_class = Counter()

    # ---------- 캐시 ----------
    def _cache_paths(self, cid) -> List[Path]:
        d = self.cache_dir / str(cid)
        return sorted(d.glob("*.png")) if d.is_dir() else []

    def _save_cache(self, cid, idx, bgr, mask):
        d = self.cache_dir / str(cid)
        d.mkdir(parents=True, exist_ok=True)
        rgba = np.dstack([bgr, (mask * 255).astype(np.uint8)])
        imwrite_unicode(d / f"{idx:04d}.png", rgba)

    @staticmethod
    def _tight(bgr, mask):
        """마스크의 최소 외접 사각형으로 잘라 냅니다."""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        return bgr[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()

    # ---------- 구축 ----------
    def build(
        self,
        records: Sequence[Dict[str, Any]],
        rebuild: bool = False,
        verbose: bool = True,
        names: Optional[Sequence[str]] = None,   # ★ 폴더명 매칭용 클래스명
    ) -> "PillCropLibrary":
        """records = [{"src": 이미지경로, "boxes": [(x, y, w, h, cls), ...]}, ...]"""
        if rebuild and self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # ---- 박스 크기 통계 (붙여넣을 때 목표 크기로 사용) ----
        ws, hs = defaultdict(list), defaultdict(list)
        by_class: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]] = defaultdict(list)
        for r in records:
            for x, y, w, h, cid in r["boxes"]:
                ws[int(cid)].append(w)
                hs[int(cid)].append(h)
                by_class[int(cid)].append((r["src"], (x, y, w, h)))
        for cid in ws:
            self.box_stats[cid] = {
                "w_med": float(np.median(ws[cid])),
                "h_med": float(np.median(hs[cid])),
            }
        if ws:
            self.global_box = {
                "w_med": float(np.median([v for a in ws.values() for v in a])),
                "h_med": float(np.median([v for a in hs.values() for v in a])),
            }

        # ---- ★ 크롭 폴더가 지정되면 그쪽에서 재료를 읽습니다 ----
        if self.crops_dir is not None:
            return self._build_from_crops_dir(names=names, rebuild=rebuild, verbose=verbose)

        # ---- 클래스별 컷아웃 ----
        classes = sorted(by_class)
        for n_done, cid in enumerate(classes, 1):
            cached = self._cache_paths(cid)
            if cached and not rebuild:
                loaded = []
                for p in cached[: self.max_per_class]:
                    rgba = imread_unicode_unchanged(p)
                    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
                        continue
                    loaded.append((rgba[..., :3].copy(), (rgba[..., 3] > 127).astype(np.uint8)))
                if loaded:
                    self.items[cid] = loaded
                    self.stats["캐시 재사용"] += len(loaded)
                    continue

            out, idx = [], 0
            for src, (x, y, w, h) in by_class[cid][: self.max_per_class * 3]:
                if len(out) >= self.max_per_class:
                    break
                img = imread_unicode(src)
                if img is None:
                    self.stats["읽기 실패"] += 1
                    continue
                H, W = img.shape[:2]
                mx, my = w * self.margin, h * self.margin
                x0 = int(max(0, round(x - mx)))
                y0 = int(max(0, round(y - my)))
                x1 = int(min(W, round(x + w + mx)))
                y1 = int(min(H, round(y + h + my)))
                if x1 - x0 < 16 or y1 - y0 < 16:
                    self.stats["너무 작음"] += 1
                    continue
                crop = img[y0:y1, x0:x1]

                m = cutout_pill(crop)
                if m is None:
                    self.stats["컷아웃 실패"] += 1
                    self.fail_by_class[cid] += 1
                    continue
                t = self._tight(crop, m)
                if t is None:
                    self.stats["컷아웃 실패"] += 1
                    continue
                bgr, mask = t
                if min(bgr.shape[:2]) < 16:
                    self.stats["너무 작음"] += 1
                    continue
                self._save_cache(cid, idx, bgr, mask)
                idx += 1
                out.append((bgr, mask))
                self.stats["신규 컷아웃"] += 1

            if out:
                self.items[cid] = out
            if verbose and n_done % 20 == 0:
                print(f"    크롭 라이브러리 {n_done}/{len(classes)} 클래스")

        return self

    # ---------- ★ cropped_pills_review 폴더에서 구축 ----------
    def _build_from_crops_dir(
        self,
        names: Optional[Sequence[str]],
        rebuild: bool = False,
        verbose: bool = True,
    ) -> "PillCropLibrary":
        """CROPPED_PILLS_DIR 의 클래스별 하위 폴더에서 크롭 이미지를 읽어 옵니다."""
        root = self.crops_dir
        if root is None or not root.is_dir():
            raise FileNotFoundError(
                f"크롭 폴더를 찾을 수 없습니다: {root}\n"
                f"→ pill_transforms.CROPPED_PILLS_DIR 경로를 확인하세요."
            )
        if not names:
            raise ValueError(
                "크롭 폴더를 쓰려면 클래스명(names)이 필요합니다. "
                "build(records, names=names) 로 전달하세요."
            )

        lut = _build_name_lookup(names)
        subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
        if verbose:
            print(f"    크롭 폴더 {root}  (하위 폴더 {len(subdirs)}개)")

        for n_done, d in enumerate(subdirs, 1):
            cid = _match_class_id(d.name, lut)
            if cid is None:
                self.unmatched_folders.append(d.name)
                self.stats["폴더 매칭 실패"] += 1
                continue

            # 캐시 재사용
            cached = self._cache_paths(cid)
            if cached and not rebuild:
                loaded = []
                for p in cached[: self.max_per_class]:
                    rgba = imread_unicode_unchanged(p)
                    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
                        continue
                    loaded.append((rgba[..., :3].copy(), (rgba[..., 3] > 127).astype(np.uint8)))
                if loaded:
                    self.items[cid] = loaded
                    self.stats["캐시 재사용"] += len(loaded)
                    continue

            files = sorted(
                p for p in d.iterdir()
                if p.is_file() and p.suffix.lower() in CROPPED_PILLS_EXTS
            )
            out, idx = [], 0
            for p in files[: self.max_per_class * 3]:
                if len(out) >= self.max_per_class:
                    break
                raw = imread_unicode_unchanged(p)
                if raw is None:
                    self.stats["읽기 실패"] += 1
                    continue
                if raw.ndim == 2:
                    raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

                # 알파 채널이 있으면 그대로 마스크로 사용
                m = None
                if CROPPED_USE_ALPHA and raw.ndim == 3 and raw.shape[2] == 4:
                    a = raw[..., 3]
                    crop = raw[..., :3].copy()
                    if a.min() < 250:                      # 실제 투명 영역 존재
                        m = (a > 127).astype(np.uint8)
                        self.stats["알파 사용"] += 1
                else:
                    crop = raw[..., :3].copy() if raw.ndim == 3 else raw

                if min(crop.shape[:2]) < 16:
                    self.stats["너무 작음"] += 1
                    continue

                # 알파가 없으면 기존 컷아웃으로 배경 제거
                if m is None:
                    m = cutout_pill(crop)
                if m is None:
                    self.stats["컷아웃 실패"] += 1
                    self.fail_by_class[cid] += 1
                    continue

                t = self._tight(crop, m)
                if t is None:
                    self.stats["컷아웃 실패"] += 1
                    continue
                bgr, mask = t
                if min(bgr.shape[:2]) < 16:
                    self.stats["너무 작음"] += 1
                    continue

                self._save_cache(cid, idx, bgr, mask)
                idx += 1
                out.append((bgr, mask))
                self.stats["신규 크롭"] += 1

            if out:
                self.items[cid] = out
            if verbose and n_done % 20 == 0:
                print(f"    크롭 라이브러리 {n_done}/{len(subdirs)} 폴더")

        if verbose and self.unmatched_folders:
            head = ", ".join(self.unmatched_folders[:5])
            print(f"    ⚠️ 클래스명 매칭 실패 폴더 {len(self.unmatched_folders)}개: {head}"
                  f"{' ...' if len(self.unmatched_folders) > 5 else ''}")
        return self

    # ---------- 조회 ----------
    @property
    def classes(self) -> List[int]:
        return sorted(self.items)

    def target_size(self, cid, rng) -> Tuple[float, float]:
        """★ 원본 GT 박스 통계에 맞춘 목표 크기 — 붙여넣기 크기 분포가 유지됩니다."""
        st = self.box_stats.get(int(cid))
        w = st["w_med"] if st else self.global_box["w_med"]
        h = st["h_med"] if st else self.global_box["h_med"]
        j = rng.uniform(*CP_SCALE_JIT)
        return w * j, h * j

    def patch(self, cid, rng, rotate: bool = True):
        """붙여넣을 (bgr, mask) 를 목표 크기로 리사이즈 + 회전해 반환합니다."""
        pool = self.items.get(int(cid))
        if not pool:
            return None
        bgr, mask = pool[rng.randrange(len(pool))]

        # 1) 목표 크기로 리사이즈 (종횡비는 크롭 원본을 따름)
        tw, th = self.target_size(cid, rng)
        h0, w0 = mask.shape
        s = min(tw / max(w0, 1), th / max(h0, 1))
        s = float(np.clip(s, 0.05, 8.0))
        nw, nh = max(8, int(round(w0 * s))), max(8, int(round(h0 * s)))
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        # 2) 회전 (알약은 방향이 무의미하므로 0~360)
        if rotate:
            diag = int(np.ceil(np.hypot(nh, nw)))
            ci = np.zeros((diag, diag, 3), np.uint8)
            cm = np.zeros((diag, diag), np.uint8)
            oy, ox = (diag - nh) // 2, (diag - nw) // 2
            ci[oy:oy + nh, ox:ox + nw] = bgr
            cm[oy:oy + nh, ox:ox + nw] = mask
            M = cv2.getRotationMatrix2D((diag / 2, diag / 2), rng.uniform(0, 360), 1.0)
            bgr = cv2.warpAffine(ci, M, (diag, diag), flags=cv2.INTER_CUBIC)
            mask = cv2.warpAffine(cm, M, (diag, diag), flags=cv2.INTER_NEAREST)
            t = self._tight(bgr, mask)
            if t is None:
                return None
            bgr, mask = t
        return bgr, mask


class CopyPasteAugmentor:
    """크롭 라이브러리의 알약을 새 위치에 붙여 합성 이미지를 만듭니다.

    Args:
        library:     PillCropLibrary
        mode:        "synth" | "onto_train" | "mix"
        class_freq:  {class_index: 원본 등장 횟수} — 주면 **희소 클래스를 더 자주**
                     추출합니다 (역빈도 가중). None 이면 균등 추출.
    """

    def __init__(
        self,
        library: PillCropLibrary,
        mode: str = CP_MODE,
        class_freq: Optional[Dict[int, int]] = None,
        pills_range: Tuple[int, int] = PILLS_RANGE,
        extra_range: Tuple[int, int] = CP_EXTRA_RANGE,
        overlap: float = CP_OVERLAP,
        feather: int = CP_FEATHER,
        synth_ratio: float = CP_SYNTH_RATIO,
    ):
        self.lib = library
        self.mode = mode
        self.pills_range = pills_range
        self.extra_range = extra_range
        self.overlap = overlap
        self.feather = feather
        self.synth_ratio = synth_ratio

        self.pool = library.classes
        if not self.pool:
            raise RuntimeError(
                "크롭 라이브러리가 비어 있습니다.\n"
                "→ 컷아웃 파라미터(CUT_MIN_CHROMA, CUT_CHROMA_K)를 확인하세요."
            )

        # ★ 희소 클래스일수록 큰 가중치 (1/sqrt(빈도))
        if class_freq:
            self.weights = [
                1.0 / math.sqrt(max(class_freq.get(c, 1), 1)) for c in self.pool
            ]
        else:
            self.weights = [1.0] * len(self.pool)
        self.stats = Counter()

    # ---------- 배경 ----------
    @staticmethod
    def blank_canvas(H, W, rng):
        """단색 + 가우시안 노이즈 + 완만한 밝기 기울기."""
        base = np.array([rng.randint(110, 245) for _ in range(3)], np.float32)
        canvas = np.ones((H, W, 3), np.float32) * base
        gy = np.linspace(rng.uniform(-14, 14), rng.uniform(-14, 14), H)[:, None]
        gx = np.linspace(rng.uniform(-14, 14), rng.uniform(-14, 14), W)[None, :]
        canvas += (gy + gx)[..., None]
        canvas += np.random.normal(0, 5, (H, W, 3))
        return np.clip(canvas, 0, 255).astype(np.uint8)

    # ---------- 그림자 ----------
    def cast_shadow(self, canvas, mask, y, x, rng):
        """마스크를 눕혀 그림자 레이어로 합성합니다.

        ★ 패치 사각형 안에서 warp 하면 눕힌 그림자가 경계에서 잘려
          '사각형 자국'이 남습니다. 여유 패딩을 준 뒤 잘라 붙입니다.
        """
        ph, pw = mask.shape
        pad = max(8, int(max(ph, pw) * 0.5))
        big = np.zeros((ph + 2 * pad, pw + 2 * pad), np.float32)
        big[pad:pad + ph, pad:pad + pw] = mask.astype(np.float32)
        H2, W2 = big.shape
        cy2 = H2 / 2.0

        shear = rng.uniform(-0.6, 0.6)
        stretch = rng.uniform(0.8, 1.6)
        M = np.float32([[1.0, shear, -shear * cy2],
                        [0.0, stretch, cy2 * (1.0 - stretch)]])
        sh = cv2.warpAffine(big, M, (W2, H2), flags=cv2.INTER_LINEAR, borderValue=0.0)

        k = rng.randint(*SHADOW_BLUR) | 1
        sh = cv2.GaussianBlur(sh, (k, k), 0)
        alpha = rng.uniform(*SHADOW_ALPHA)

        oy = y - pad + rng.randint(-4, 10)
        ox = x - pad + rng.randint(-4, 10)
        y0, x0 = max(0, oy), max(0, ox)
        y1 = min(canvas.shape[0], oy + H2)
        x1 = min(canvas.shape[1], ox + W2)
        if y1 <= y0 or x1 <= x0:
            return
        sub = sh[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        roi = canvas[y0:y1, x0:x1].astype(np.float32)
        canvas[y0:y1, x0:x1] = np.clip(
            roi * (1.0 - alpha * sub[..., None]), 0, 255
        ).astype(np.uint8)

    # ---------- 알파 합성 ----------
    def blend(self, canvas, patch, mask, y, x):
        ph, pw = mask.shape
        a = mask.astype(np.float32)
        if self.feather > 0:
            k = self.feather * 2 + 1
            a = cv2.GaussianBlur(a, (k, k), 0)
        a = a[..., None]
        roi = canvas[y:y + ph, x:x + pw].astype(np.float32)
        canvas[y:y + ph, x:x + pw] = np.clip(
            roi * (1 - a) + patch.astype(np.float32) * a, 0, 255
        ).astype(np.uint8)

    # ---------- 알약 1개 붙이기 ----------
    def paste_one(self, canvas, occ, rng, tries: int = 40):
        cid = rng.choices(self.pool, weights=self.weights, k=1)[0]
        got = self.lib.patch(cid, rng)
        if got is None:
            self.stats["패치 생성 실패"] += 1
            return None
        patch, mask = got
        ph, pw = mask.shape
        H, W = canvas.shape[:2]
        if ph >= H or pw >= W:
            self.stats["크기 초과"] += 1
            return None

        reg = mask > 0
        area = int(reg.sum())
        if area <= 0:
            return None

        for _ in range(tries):
            yy = rng.randint(0, H - ph)
            xx = rng.randint(0, W - pw)
            sub = occ[yy:yy + ph, xx:xx + pw]
            if int(np.count_nonzero(sub[reg])) <= self.overlap * area:
                self.cast_shadow(canvas, mask, yy, xx, rng)   # ★ 그림자 먼저
                self.blend(canvas, patch, mask, yy, xx)       # ★ 알약을 그 위에
                occ[yy:yy + ph, xx:xx + pw][reg] = 1
                return (xx + pw / 2.0, yy + ph / 2.0, float(pw), float(ph), int(cid))
        self.stats["배치 실패"] += 1
        return None

    # ---------- 합성 이미지 1장 ----------
    def synth_one(self, H, W, rng) -> Optional[Sample]:
        canvas = self.blank_canvas(H, W, rng)
        occ = np.zeros((H, W), np.uint8)
        boxes = []
        for _ in range(rng.randint(*self.pills_range)):
            b = self.paste_one(canvas, occ, rng)
            if b:
                boxes.append(b)
        if not boxes:
            return None
        return Sample(canvas, boxes, {"kind": "cp_synth"})

    def paste_onto(self, base: Sample, rng) -> Optional[Sample]:
        """원본 train 이미지 위에 알약을 추가로 붙입니다."""
        s = base.clone()
        H, W = s.hw
        occ = np.zeros((H, W), np.uint8)
        for cx, cy, w, h, _ in s.boxes:            # 기존 알약 자리는 점유 처리
            x1, y1 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
            x2, y2 = int(min(W, cx + w / 2)), int(min(H, cy + h / 2))
            occ[y1:y2, x1:x2] = 1
        added = 0
        for _ in range(rng.randint(*self.extra_range)):
            b = self.paste_one(s.image, occ, rng)
            if b:
                s.boxes.append(b)
                added += 1
        if added == 0:
            return None
        s.meta = {"kind": "cp_onto", "n_added": added}
        return s

    # ---------- 배치 생성 ----------
    def generate(
        self,
        n: int,
        train_records: Sequence[Dict[str, Any]],
        rng: random.Random,
        canvas_hw: Tuple[int, int],
        progress_every: int = 200,
    ) -> List[Sample]:
        out: List[Sample] = []
        H0, W0 = canvas_hw
        for k in range(n):
            if self.mode == "synth":
                use_synth = True
            elif self.mode == "onto_train":
                use_synth = False
            else:
                use_synth = rng.random() < self.synth_ratio

            if use_synth or not train_records:
                s = self.synth_one(H0, W0, rng)
                use_synth = True
            else:
                r = rng.choice(list(train_records))
                img = imread_unicode(r["src"])
                if img is None:
                    continue
                s = self.paste_onto(Sample.from_xywh(img, r["boxes"]), rng)

            if s is not None:
                s.meta["index"] = k
                out.append(s)
                self.stats["synth" if use_synth else "onto_train"] += 1
            if progress_every and (k + 1) % progress_every == 0:
                print(f"    Copy&Paste {k + 1:,}/{n:,}")
        return out


# ═══════════════════════════════════════════════════════════════════════════
#  Part 6. YOLO 데이터셋 빌더  (yolo11s_baseline.ipynb 가 호출)
# ═══════════════════════════════════════════════════════════════════════════

def _read_yolo_label(path: Path, W: int, H: int) -> List[Tuple[float, float, float, float, int]]:
    """YOLO txt → [(x, y, w, h, cls)] 좌상단 절대 픽셀."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        bw, bh = w * W, h * H
        out.append((cx * W - bw / 2, cy * H - bh / 2, bw, bh, c))
    return out


def _write_yolo_label(path: Path, boxes, W: int, H: int) -> int:
    """[(x, y, w, h, cls)] → YOLO txt. 저장한 박스 수를 반환."""
    lines = []
    for x, y, w, h, c in boxes:
        if w <= 0 or h <= 0:
            continue
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        nw, nh = min(w / W, 1.0), min(h / H, 1.0)
        lines.append(
            f"{int(c)} {np.clip(cx, 0, 1):.6f} {np.clip(cy, 0, 1):.6f} {nw:.6f} {nh:.6f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return len(lines)


def _collect_split(src_root: Path, split: str) -> List[Dict[str, Any]]:
    """images/{split} + labels/{split} 를 읽어 records 로 만듭니다."""
    img_dir = src_root / "images" / split
    lbl_dir = src_root / "labels" / split
    if not img_dir.is_dir():
        return []

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    records = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        img = imread_unicode(p)
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = _read_yolo_label(lbl_dir / f"{p.stem}.txt", W, H)
        records.append(
            {"src": str(p), "file_name": p.name, "stem": p.stem,
             "width": W, "height": H, "boxes": boxes}
        )
    return records


def build_augmented_yolo_dataset(
    src_root: PathLike,
    dst_root: Optional[PathLike] = None,
    geom_mult: Optional[int] = None,
    n_synth: Optional[int] = None,
    *,
    use_flip: Optional[bool] = None,
    cp_mode: Optional[str] = None,
    cp_weighted: bool = True,
    max_crops_per_class: Optional[int] = None,
    crops_dir: Optional[PathLike] = None,   # ★ cropped_pills_review 경로
    rebuild_crop_cache: bool = False,
    preprocess_val_test: bool = True,
    seed: Optional[int] = None,
    overwrite: bool = True,
    verbose: bool = True,
) -> str:
    """YOLO 데이터셋을 읽어 **증강된 새 YOLO 데이터셋**을 만들고 data.yaml 경로를 반환합니다.

    Args:
        src_root: `pill_detection_dataset.ipynb` 가 만든 폴더.
                  images/{train,val,test}, labels/{train,val,test}, data.yaml 필요.
        dst_root: 결과 폴더. None 이면 `<src_root>_aug`.
        geom_mult: ★ train 이미지 1장당 최종 장수. 3 이면 원본 1 + 증강 2.
                   1 이면 기하 증강 없음(원본만).
        n_synth:   ★ Copy & Paste 합성 이미지 수. 0 이면 끔.
        crops_dir: ★ `cropped_pills_review` 폴더 경로. 지정하면 Copy&Paste 재료를
                   train 이미지에서 컷아웃하지 않고 이 폴더의 크롭 이미지로 씁니다.
                   None 이면 전역 `CROPPED_PILLS_DIR` 을 따릅니다.
        cp_weighted: True 면 희소 클래스를 더 자주 붙여 불균형을 완화합니다.
        preprocess_val_test: val/test 에도 동일 전처리(WB+CLAHE)를 적용할지.
                   ★ train 에만 걸면 분포가 어긋나므로 기본 True 를 권장합니다.

    Returns:
        생성된 `data.yaml` 의 절대 경로 문자열.

    주의:
        val / test 에는 **증강을 적용하지 않습니다.** 전처리만 동일하게 겁니다.
        증강본은 train 원본에서만 파생되므로 데이터 누수가 원천 차단됩니다.
    """
    # ★ 인자를 안 넘기면 "지금 이 순간의" 전역 설정값을 씁니다.
    #   (파일 상단 CONFIG 블록을 바꾸거나 노트북에서 pt.DEFAULT_GEOM_MULT = 5 처럼
    #    바꾼 뒤 이 함수를 호출하면 그 값이 반영됩니다)
    if geom_mult is None:
        geom_mult = DEFAULT_GEOM_MULT
    if n_synth is None:
        n_synth = DEFAULT_N_SYNTH
    if use_flip is None:
        use_flip = USE_FLIP
    if cp_mode is None:
        cp_mode = CP_MODE
    if max_crops_per_class is None:
        max_crops_per_class = MAX_CROPS_PER_CLASS
    if seed is None:
        seed = SEED

    t0 = time.time()
    src_root = Path(src_root).resolve()
    dst_root = Path(dst_root).resolve() if dst_root else src_root.parent / f"{src_root.name}_aug"

    if geom_mult < 1:
        raise ValueError("geom_mult 는 1 이상이어야 합니다. (1 = 증강 없음)")
    if n_synth < 0:
        raise ValueError("n_synth 는 0 이상이어야 합니다. (0 = Copy&Paste 끔)")

    # ---------- 0. 입력 확인 ----------
    if not (src_root / "images" / "train").is_dir():
        raise FileNotFoundError(
            f"{src_root}/images/train 이 없습니다.\n"
            "→ pill_detection_dataset.ipynb 의 YOLO 내보내기 셀을 먼저 실행하세요."
        )

    src_yaml = src_root / "data.yaml"
    names: List[str] = []
    if src_yaml.exists():
        try:
            import yaml

            cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8")) or {}
            raw = cfg.get("names", [])
            if isinstance(raw, dict):
                names = [raw[k] for k in sorted(raw, key=lambda v: int(v))]
            else:
                names = list(raw)
        except Exception as e:  # pragma: no cover
            print(f"⚠️ data.yaml 을 읽지 못했습니다({e}). 클래스 이름을 인덱스로 대체합니다.")

    rng = random.Random(seed)
    np.random.seed(seed)

    # ★ 현재 전역값(USE_WHITE_BALANCE/USE_CLAHE/CLAHE_CLIP/CLAHE_GRID)을 반영한
    #   전처리 인스턴스. 노트북에서 pt.CLAHE_CLIP 등을 바꾼 뒤 이 함수를 부르면
    #   그 값이 그대로 적용됩니다 (import 시점에 고정되는 모듈 전역 PREPROCESS 와 다름).
    _pp = Preprocess(
        white_balance=USE_WHITE_BALANCE, clahe=USE_CLAHE,
        clip=CLAHE_CLIP, grid=CLAHE_GRID,
    )

    if verbose:
        print("═" * 66)
        print(f"  증강 데이터셋 생성   기하 ×{geom_mult}  |  Copy&Paste {n_synth}장")
        print(f"  입력 {src_root}")
        print(f"  출력 {dst_root}")
        print("═" * 66)

    # ---------- 1. 원본 수집 ----------
    train_recs = _collect_split(src_root, "train")
    val_recs = _collect_split(src_root, "val")
    test_recs = _collect_split(src_root, "test")
    if not train_recs:
        raise RuntimeError("train 이미지를 하나도 읽지 못했습니다.")

    seen_cls = {c for r in train_recs for *_, c in r["boxes"]}
    if not names:
        names = [str(i) for i in range((max(seen_cls) + 1) if seen_cls else 1)]

    if verbose:
        print(f"[1/5] 원본  train {len(train_recs):,} / val {len(val_recs):,} / "
              f"test {len(test_recs):,}  |  클래스 {len(names)}종")

    # ---------- 2. 출력 폴더 준비 ----------
    if overwrite and dst_root.exists():
        shutil.rmtree(dst_root)
    for split in ("train", "val", "test"):
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = Counter()
    box_counts = Counter()

    def emit(sample: Sample, fn: str, kind: str) -> None:
        """전처리 → 이미지 저장 → 라벨 저장. ★ 전처리는 언제나 마지막."""
        img = _pp.apply_image(sample.image)
        h, w = img.shape[:2]
        imwrite_unicode(dst_root / "images" / "train" / fn, img)
        n = _write_yolo_label(
            dst_root / "labels" / "train" / f"{Path(fn).stem}.txt",
            sample.boxes_xywh(), w, h,
        )
        counts[kind] += 1
        box_counts[kind] += n

    # ---------- 3. train 원본 + 기하 증강 ----------
    geom = GeometricAugmentor(multiplier=geom_mult, use_flip=use_flip)
    if verbose:
        print(f"[2/5] train 원본 저장 + 기하 증강 (이미지당 +{geom.n_extra}장)")

    for n_done, r in enumerate(train_recs, 1):
        img = imread_unicode(r["src"])
        if img is None:
            counts["읽기 실패"] += 1
            continue
        boxes = [b for b in r["boxes"] if b[2] > 0 and b[3] > 0]
        base = Sample.from_xywh(img, boxes)

        emit(base, r["file_name"], "orig")

        if geom.n_extra > 0 and boxes:
            for k, s in enumerate(geom.generate(base, geom.n_extra, rng)):
                emit(s, f"aug_{r['stem']}_{k}.png", "aug")

        if verbose and (n_done % 100 == 0 or n_done == len(train_recs)):
            print(f"    {n_done:,}/{len(train_recs):,} 원본 처리")

    # ---------- 4. Copy & Paste ----------
    if n_synth > 0:
        if verbose:
            print(f"[3/5] 크롭 라이브러리 구축 (클래스당 최대 {max_crops_per_class}장)")
        _crops_dir = crops_dir if crops_dir is not None else CROPPED_PILLS_DIR
        if verbose and _crops_dir:
            print(f"    ★ 재료 소스: cropped_pills_review → {_crops_dir}")
        lib = PillCropLibrary(
            cache_dir=dst_root / "_crop_cache",
            max_per_class=max_crops_per_class,
            crops_dir=_crops_dir,
        ).build(train_recs, rebuild=rebuild_crop_cache, verbose=verbose, names=names)

        n_items = sum(len(v) for v in lib.items.values())
        if verbose:
            print(f"    사용 가능 클래스 {len(lib.classes)}종 / 전체 {len(names)}종")
            print(f"    컷아웃 알약 {n_items:,}개  {dict(lib.stats)}")

        if not lib.classes:
            print("⚠️ 컷아웃에 전부 실패해 Copy&Paste 를 건너뜁니다.")
            print("   → CUT_MIN_CHROMA 를 5.0 으로, CUT_CHROMA_K 를 1.6 으로 낮춰 보세요.")
            if _crops_dir:
                print(f"   → 크롭 폴더 경로도 확인하세요: {_crops_dir}")
                if lib.unmatched_folders:
                    print(f"   → 폴더명이 data.yaml 클래스명과 매칭되지 않았습니다 "
                          f"({len(lib.unmatched_folders)}개). 예: {lib.unmatched_folders[:3]}")
        else:
            freq = Counter(c for r in train_recs for *_, c in r["boxes"])
            cp = CopyPasteAugmentor(
                lib, mode=cp_mode, class_freq=dict(freq) if cp_weighted else None,
                pills_range=PILLS_RANGE, extra_range=CP_EXTRA_RANGE,
                overlap=CP_OVERLAP, feather=CP_FEATHER, synth_ratio=CP_SYNTH_RATIO,
            )
            hs = [r["height"] for r in train_recs]
            wsz = [r["width"] for r in train_recs]
            canvas_hw = (int(np.median(hs)), int(np.median(wsz)))

            if verbose:
                print(f"[4/5] Copy&Paste {n_synth:,}장 생성 "
                      f"(모드 {cp_mode}, 캔버스 {canvas_hw[1]}x{canvas_hw[0]})")

            for k, s in enumerate(cp.generate(n_synth, train_recs, rng, canvas_hw)):
                emit(s, f"cp_{k:06d}.png", "cp")

            if verbose and cp.stats:
                print(f"    Copy&Paste 내부 통계: {dict(cp.stats)}")
    elif verbose:
        print("[3-4/5] Copy&Paste 건너뜀 (n_synth=0)")

    # ---------- 5. val / test — 증강 없이 전처리만 ----------
    if verbose:
        print("[5/5] val / test 복사 (증강 없음, 전처리만)")

    for split, recs in (("val", val_recs), ("test", test_recs)):
        for r in recs:
            dst_img = dst_root / "images" / split / r["file_name"]
            if preprocess_val_test:
                img = imread_unicode(r["src"])
                if img is None:
                    continue
                imwrite_unicode(dst_img, _pp.apply_image(img))
            else:
                shutil.copy2(r["src"], dst_img)
            _write_yolo_label(
                dst_root / "labels" / split / f"{r['stem']}.txt",
                r["boxes"], r["width"], r["height"],
            )
            counts[split] += 1

    # ---------- 6. data.yaml + 기록 ----------
    yaml_path = dst_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {dst_root}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write(f"nc: {len(names)}\nnames:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")

    info = {
        "source": str(src_root),
        "geom_mult": geom_mult,
        "n_synth": n_synth,
        "cp_mode": cp_mode,
        "crops_dir": str(crops_dir if crops_dir is not None else CROPPED_PILLS_DIR or ""),
        "cp_weighted": cp_weighted,
        "use_flip": use_flip,
        "preprocess": {
            "white_balance": USE_WHITE_BALANCE,
            "clahe": USE_CLAHE,
            "clahe_clip": CLAHE_CLIP,
            "applied_to": "train/val/test 전부",
        },
        "counts": dict(counts),
        "box_counts": dict(box_counts),
        "seed": seed,
    }
    (dst_root / "augment_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------- 7. 요약 + 검산 ----------
    n_train = counts["orig"] + counts["aug"] + counts["cp"]
    target = len(train_recs) * geom_mult
    if verbose:
        print("\n■ 생성 결과")
        print(f"  train  원본 {counts['orig']:,} + 증강 {counts['aug']:,} "
              f"+ Copy&Paste {counts['cp']:,} = {n_train:,}장")
        print(f"  val    {counts['val']:,}장 / test {counts['test']:,}장")
        ok = (counts["orig"] + counts["aug"]) == target
        print(f"\n  ★ 기하 증강 검산 {counts['orig'] + counts['aug']:,} / 목표 {target:,}장 "
              f"{'✅' if ok else '⚠️ 박스 전멸로 일부 손실'}")
        if n_synth:
            print(f"  ★ Copy&Paste  {counts['cp']:,} / 목표 {n_synth:,}장 "
                  f"{'✅' if counts['cp'] == n_synth else '⚠️ 배치 실패분 손실'}")
        print(f"\n  전처리(WB={USE_WHITE_BALANCE}, CLAHE={USE_CLAHE}) → "
              f"train/val/test 전부 적용 ✅")
        print(f"  소요 시간 {time.time() - t0:.1f}초")
        print(f"\n★ data.yaml = {yaml_path}")

    return str(yaml_path)


# ═══════════════════════════════════════════════════════════════════════════
#  Part 7. 검수용 시각화 (선택)
# ═══════════════════════════════════════════════════════════════════════════

def preview_augmented(
    dst_root: PathLike,
    out_dir: Optional[PathLike] = None,
    per_kind: int = 3,
    names: Optional[Sequence[str]] = None,
) -> List[str]:
    """생성된 train 폴더에서 유형별(orig/aug/cp) 표본에 박스를 그려 저장합니다.

    ★ 학습 전에 반드시 눈으로 확인하세요.
        박스 위치 : 알약에 딱 맞는가 (밀렸으면 RotateScale 확인)
        박스 크기 : 그림자까지 포함하지 않았는가 (MIN_VISIBILITY 확인)
        각인      : 글자가 읽히는가 (뭉개졌으면 P_BLUR 를 0 으로)
        Copy&Paste: 경계에 후광이 없는가 (CP_FEATHER 조절)
    """
    dst_root = Path(dst_root)
    out_dir = Path(out_dir) if out_dir else dst_root / "_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir = dst_root / "images" / "train"
    lbl_dir = dst_root / "labels" / "train"
    palette = [(255, 89, 94), (56, 176, 0), (25, 130, 196), (255, 202, 58),
               (138, 80, 220), (0, 187, 249), (241, 91, 181), (155, 200, 60)]

    seen, saved = Counter(), []
    for p in sorted(img_dir.iterdir()):
        kind = "cp" if p.name.startswith("cp_") else ("aug" if p.name.startswith("aug_") else "orig")
        if seen[kind] >= per_kind:
            continue
        img = imread_unicode(p)
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = _read_yolo_label(lbl_dir / f"{p.stem}.txt", W, H)
        if not boxes:
            continue
        vis = img.copy()
        for x, y, w, h, c in boxes:
            col = palette[int(c) % len(palette)]
            cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), col, 3)
            label = str(names[int(c)]) if names and int(c) < len(names) else str(int(c))
            cv2.putText(vis, label[:12], (int(x), max(14, int(y) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        out_p = out_dir / f"{kind}_{seen[kind]}_{p.stem}.png"
        imwrite_unicode(out_p, vis)
        saved.append(str(out_p))
        seen[kind] += 1

    print(f"검수 이미지 {len(saved)}장 → {out_dir}   유형별 {dict(seen)}")
    return saved


# ═══════════════════════════════════════════════════════════════════════════
#  셀프 테스트 — import 할 때는 실행되지 않습니다.
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile

    print("■ 셀프 테스트 (더미 데이터)")
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "processed"

    # 더미 YOLO 데이터셋 생성
    for split, n in (("train", 6), ("val", 2), ("test", 2)):
        (src / "images" / split).mkdir(parents=True, exist_ok=True)
        (src / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            H, W = 640, 480
            img = np.full((H, W, 3), 200, np.uint8)
            lines = []
            for j in range(3):
                cx, cy = random.uniform(0.25, 0.75), random.uniform(0.25, 0.75)
                bw, bh = 0.16, 0.12
                cv2.ellipse(
                    img,
                    (int(cx * W), int(cy * H)),
                    (int(bw * W / 2), int(bh * H / 2)),
                    0, 0, 360,
                    (60 + 40 * j, 90, 200 - 30 * j), -1,
                )
                lines.append(f"{j} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            imwrite_unicode(src / "images" / split / f"{split}_{i}.png", img)
            (src / "labels" / split / f"{split}_{i}.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )
    (src / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "nc: 3\nnames:\n  0: A\n  1: B\n  2: C\n",
        encoding="utf-8",
    )

    out_yaml = build_augmented_yolo_dataset(src, geom_mult=3, n_synth=5, verbose=True)

    dst = Path(out_yaml).parent
    n_tr = len(list((dst / "images" / "train").iterdir()))
    print(f"\n검증: train 이미지 {n_tr}장 (기대 6×3 + 5 = 23)")
    assert n_tr == 6 * 3 + 5, n_tr

    preview_augmented(dst, per_kind=1, names=["A", "B", "C"])

    if HAS_ALBUMENTATIONS:
        tf = get_train_transform(image_size=320, to_tensor=False)
        r = tf(image=np.random.randint(0, 255, (1280, 976, 3), np.uint8),
               bboxes=[[100, 150, 300, 400]], labels=[1])
        print(f"\nAlbumentations 파이프라인 OK — image {r['image'].shape}, "
              f"bboxes {len(r['bboxes'])}")
    else:
        print("\n(albumentations 미설치 — 온라인 transform 테스트는 건너뜀)")

    shutil.rmtree(tmp)
    print("\n■ 모든 테스트 통과")
