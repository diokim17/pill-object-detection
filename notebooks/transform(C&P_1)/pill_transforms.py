"""
pill_transforms.py
==================

경구약제(알약) Object Detection 프로젝트 공용 전처리 / 증강 모듈입니다.
팀 노트북 4개가 **이 파일 하나만** import 하면 됩니다.

    ┌──────────────────────────────────────┬────────────────────────────────────┐
    │ 노트북                               │ 이 모듈에서 쓰는 것                │
    ├──────────────────────────────────────┼────────────────────────────────────┤
    │ pill_detection_dataset.ipynb         │ get_train_transforms()             │
    │ yolo11s_baseline_cp.ipynb            │ configure() / preflight() /        │
    │                                      │ build_augmented_yolo_dataset()     │
    │ base_model_faster-rcnn_train.ipynb   │ get_frcnn_train_transform()        │
    │                                      │ get_frcnn_eval_transform()         │
    │ base_model_faster-rcnn_predict.ipynb │ get_frcnn_eval_transform()         │
    │                                      │ prepare_image_for_inference()      │
    │                                      │ undo_letterbox_boxes()             │
    └──────────────────────────────────────┴────────────────────────────────────┘

구조 (AI Hub 팀 공유 가이드와 동일한 실행 흐름)
------------------------------------------------
가이드의 `--preflight → --execute` 2단계 원칙을 그대로 따릅니다.

    Part 0  설정          환경 자동 감지 · 고정 배경색 · 경로(configure)
    Part 1  I/O           한글 경로 안전 read/write
    Part 2  전처리        Shades-of-Gray WB + CLAHE (train/val/test/추론 공통)
    Part 3  온라인 transform (Albumentations)  — Faster R-CNN 계열
    Part 3.5 Faster R-CNN 어댑터 · 추론 letterbox 복원
    Part 4  오프라인 기하 증강 (cv2 only)
    Part 5  컷아웃 · 크롭 라이브러리
    Part 6  Copy & Paste · YOLO 증강 데이터셋 빌더
    Part 7  검수 시각화
    Part 8  ★ preflight / execute / CLI

★ 배경색 고정
--------------
Copy & Paste 합성 배경은 더 이상 무작위로 만들지 않습니다.
팀 crop 산출물(`K-003351-...png`)의 알약 바깥 픽셀에서 실측한 값을
`PILL_BG_*` 상수로 **고정**했습니다.

    RGB (105, 110, 128) / BGR (128, 110, 105) / HEX #696E80
    HSV(OpenCV)  H=113  S=44  V=128
    채널 노이즈 std ≈ 5.0,  밝기 기울기 ≈ ±6 (V p5~p95 = 119~134)

letterbox 패딩도 같은 색으로 채웁니다(`USE_BG_PAD=True`). 검정 패딩을 쓰면
학습 이미지에 실제 촬영에 없는 색이 들어가기 때문입니다.

Colab 빠른 사용법
------------------
    from google.colab import drive; drive.mount("/content/drive")

    import sys; sys.path.insert(0, "/content/drive/MyDrive/.../pill-object-detection/src")
    import pill_transforms as pt

    pt.configure()                 # 경로 자동 감지 + 출력
    pt.preflight()                 # 1단계: 사전검증 (파일을 만들지 않음)
    aug_yaml = pt.build_augmented_yolo_dataset()   # 2단계: 실제 생성

터미널에서도 동일합니다.

    python pill_transforms.py --preflight
    python pill_transforms.py --execute --geom-mult 3 --n-synth 600
    python pill_transforms.py --bg-check /path/to/crop.png
    python pill_transforms.py --selftest

기본값
------
    DEFAULT_GEOM_MULT = 3   train 1장 → 최종 3장 (원본 1 + 증강 2)
    DEFAULT_N_SYNTH   = 600 Copy & Paste 합성 이미지 수 (200 단위로 조절)
    CLAHE_CLIP        = 5.0 전 이미지 항상 적용 (학습·평가·추론 동일)
    CP_BG_MODE        = "fixed"  ★ 고정 배경색 사용
    USE_FLIP          = False    각인이 뒤집히면 클래스 정보 손상

반영한 EDA / Dataset 특성
--------------------------
1. 원본 해상도 976x1280 동일 → 비율을 왜곡하지 않는 LongestMaxSize + Pad(letterbox)
2. bbox 가 정사각형에 가깝고 평균 area_ratio 약 5.6% → erosion_rate 를 낮게
3. 배경·조명이 전 샘플 동일 → 색상/명암 증강 필수, 배경색은 고정값으로 재현
4. 촬영 각도 70/75/90도 → 온라인은 소각도, 오프라인은 ±180도
5. 클래스 불균형 51배 → Copy & Paste 로 희소 클래스 표본을 증가
6. 각인(print_front/back)이 클래스 정보 → flip 기본 off, 블러 최소
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
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


# ═══════════════════════════════════════════════════════════════════════════
#  0-1. 실행 환경 자동 감지  (Colab · 로컬 · Windows 공용)
# ═══════════════════════════════════════════════════════════════════════════
#  ★ 예전 버전은 D:\PillData\... 같은 특정 팀원의 로컬 경로가 하드코딩돼
#    다른 팀원 환경에서 곧바로 실패했습니다. 이제는 configure() 가
#    실행 환경을 감지해 경로를 자동으로 채웁니다.
# ═══════════════════════════════════════════════════════════════════════════

IN_COLAB: bool = ("google.colab" in sys.modules) or Path("/content").is_dir()

#  Colab 에서 팀이 공유하는 Google Drive 프로젝트 폴더
COLAB_PROJECT_ROOT = (
    "/content/drive/MyDrive/코드잇 AI 13기/AI 13기 프로젝트/pill-object-detection"
)

#  환경변수로도 지정할 수 있습니다:  export PILL_PROJECT_ROOT=/path/to/project
ENV_PROJECT_KEY = "PILL_PROJECT_ROOT"


# ═══════════════════════════════════════════════════════════════════════════
#  0-2. ★★★ 고정 배경색 — 팀 crop 산출물에서 실측한 값 ★★★
# ═══════════════════════════════════════════════════════════════════════════
#  측정 대상 : K-003351-003832-016232_0_2_0_2_90_000_200.png
#              (crop_additional_dataset_shared.py 가 만든 3351_일양하이트린정 2mg)
#  측정 방법 : 상단 라벨바·파란 테두리를 제외한 뒤, 알약 외접 타원 바깥
#              (= 촬영 배경) 픽셀만 모아 채널별 median 을 취했습니다.
#
#      배경  BGR (128, 110, 105)  /  RGB (105, 110, 128)  /  HEX #696E80
#      HSV(OpenCV 0~179)  H = 112.8   S = 44.0   V = 128.0
#      채널 노이즈 std ≈ 4.4 ~ 5.5   |   V 분포 p5~p95 = 119 ~ 134
#
#  ▸ Copy & Paste 합성 배경(CP_BG_MODE="fixed")
#  ▸ letterbox 패딩(USE_BG_PAD=True)
#  두 곳이 모두 이 값을 씁니다. 값을 바꾸려면 여기 숫자만 고치면 됩니다.
# ═══════════════════════════════════════════════════════════════════════════

PILL_BG_BGR: Tuple[int, int, int] = (128, 110, 105)   # cv2 기본 채널 순서
PILL_BG_RGB: Tuple[int, int, int] = (105, 110, 128)   # albumentations / PIL 순서
PILL_BG_HEX: str = "#696E80"
PILL_BG_HSV: Tuple[float, float, float] = (113.0, 44.0, 128.0)  # H 0~179

PILL_BG_NOISE_STD: float = 5.0   # 실측 채널 노이즈 (0 이면 완전 균일한 단색)
PILL_BG_GRAD: float = 6.0        # 실측 밝기 기울기 진폭 (0 이면 기울기 없음)
PILL_BG_V_JITTER: float = 0.0    # 장마다 전체 밝기를 흔들고 싶을 때만 > 0

#  참고용 — crop 검수 이미지의 상단 라벨바 / 테두리 색 (증강에는 쓰지 않습니다)
CROP_LABEL_BAR_RGB: Tuple[int, int, int] = (24, 45, 75)
CROP_FRAME_RGB: Tuple[int, int, int] = (30, 120, 220)

#  letterbox 패딩을 배경색으로 채울지 (False 면 예전처럼 검정 0 패딩)
USE_BG_PAD: bool = True


# ═══════════════════════════════════════════════════════════════════════════
#  0-3. 노트북에서 자주 바꾸는 두 값
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_GEOM_MULT = 3     # train 원본 1장 → 최종 3장 (원본 1 + 증강 2)
DEFAULT_N_SYNTH = 600     # Copy & Paste 합성 이미지 수 (200 단위로 조절)


# ---------- 전처리 (학습·평가·추론 전부 동일하게 적용) ----------
USE_WHITE_BALANCE = True   # Shades-of-Gray 화이트밸런스
USE_CLAHE = True           # Lab 의 L 채널 CLAHE — 항상 켬(배경/객체 대비)
CLAHE_CLIP = 5.0           # 대비를 세게. 노이즈가 뜨면 4.0 으로
CLAHE_GRID = 8

# ---------- 오프라인 기하 증강 ----------
ROT_LIMIT = 180            # 알약은 회전 불변이므로 크게
ROT_PROB = 0.8
SCALE_RANGE = (0.95, 1.05)
MIN_VISIBILITY = 0.2       # 잘린 뒤 남은 면적이 이보다 작으면 박스 삭제
USE_FLIP = False           # 각인이 뒤집히므로 기본 off

P_BLUR = 0.20
P_NOISE = 0.30
P_TONE = 0.30
HSV_H_LIMIT = 8            # 색(color1)이 클래스 정보라 H 는 작게
HSV_S_LIMIT = 25
HSV_V_LIMIT = 30

# ---------- Copy & Paste ----------
PILLS_RANGE = (3, 4)       # 합성 이미지 1장에 붙일 알약 수 (원본 분포와 동일)
CP_EXTRA_RANGE = (1, 1)    # onto_train 모드에서 추가로 붙일 알약 수
CP_MODE = "mix"            # "synth" | "onto_train" | "mix"
CP_SYNTH_RATIO = 0.5       # mix 일 때 인공 배경 비율
CP_OVERLAP = 0.10          # 허용 겹침 비율
CP_SCALE_JIT = (0.95, 1.05)
CP_FEATHER = 2             # 경계 페더링(px)
SHADOW_ALPHA = (0.20, 0.55)
SHADOW_BLUR = (3, 12)
MAX_CROPS_PER_CLASS = 40   # 크롭 라이브러리 클래스당 최대 장수

# ---------- Copy & Paste 인공 배경 모드 ----------
#   "fixed"  ★ 기본 — 위에서 실측한 PILL_BG_* 고정색 (재현성 최고)
#   "crops"  크롭 이미지 배경의 색조(H,S)를 표본에서 뽑아 씀 (예전 동작)
#   "random" 완전 무작위 단색 (권장하지 않음)
CP_BG_MODE = "fixed"

CP_BG_FROM_CROPS = True        # "crops" 모드에서만 의미가 있습니다
CP_BG_V_RANGE = (160.0, 240.0) # "crops" 모드에서 새로 뽑는 밝기(V) 범위
CP_BG_S_SCALE = (0.95, 1.05)   # "crops" 모드 채도 지터
CP_BG_HUE_JITTER = 2.0         # "crops" 모드 색상 지터 (H 0~179)
CP_BG_GRAD = 14.0              # "crops"/"random" 모드 밝기 기울기
CP_BG_NOISE = 2.0              # "crops" 모드 노이즈 표준편차


# ═══════════════════════════════════════════════════════════════════════════
#  0-4. ★ 경로 — configure() 가 실행 환경에 맞춰 자동으로 채웁니다
# ═══════════════════════════════════════════════════════════════════════════
#  기대하는 프로젝트 구조 (팀 공용)
#
#      pill-object-detection/
#      ├── src/pill_transforms.py           ← 이 파일
#      ├── notebooks/*.ipynb
#      ├── data/
#      │   ├── processed/                   YOLO 데이터셋 (images/labels/data.yaml)
#      │   ├── processed_aug/               ← 이 모듈이 만드는 증강 데이터셋
#      │   ├── cropped_pills_review/        Copy & Paste 재료 (클래스별 폴더)
#      │   └── team_work/cropped_output/    AI Hub 추가 데이터 crop 결과
#      └── outputs/cutcheck/                컷아웃 검수 산출물
#
#  노트북에서 직접 지정하려면:
#      pt.configure(project_root="/content/drive/MyDrive/.../pill-object-detection")
#      pt.configure(crops_dir="/content/drive/MyDrive/.../cropped_output")
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT: Optional[str] = None
DATASET_DIR: Optional[str] = None        # 원본 YOLO 데이터셋 (src_root)
AUG_DATASET_DIR: Optional[str] = None    # 증강 결과 (dst_root)
CROPPED_PILLS_DIR: Optional[str] = None  # Copy & Paste 재료
TEAM_WORK_DIR: Optional[str] = None      # AI Hub 팀 작업 폴더
CUTOUT_CHECK_DIR: Optional[str] = None   # 컷아웃 검수 산출물

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
#  Part 0-5. 환경 설정 함수 — 모든 노트북의 첫 셀에서 호출하세요
# ═══════════════════════════════════════════════════════════════════════════

def mount_drive(mountpoint: str = "/content/drive") -> bool:
    """Colab 이면 Google Drive 를 마운트합니다. 이미 마운트돼 있으면 그냥 True."""
    if not IN_COLAB:
        print("Colab 환경이 아닙니다 — 드라이브 마운트를 건너뜁니다.")
        return False
    if Path(mountpoint, "MyDrive").is_dir():
        print(f"이미 마운트됨: {mountpoint}")
        return True
    try:
        from google.colab import drive  # type: ignore

        drive.mount(mountpoint)
        return True
    except Exception as e:  # pragma: no cover
        print(f"드라이브 마운트 실패: {e}")
        return False


def find_project_root(start: Optional[PathLike] = None) -> Path:
    """프로젝트 루트를 자동으로 찾습니다.

    우선순위
        1. 환경변수 PILL_PROJECT_ROOT
        2. Colab 의 공유 드라이브 경로 (COLAB_PROJECT_ROOT)
        3. 현재 폴더에서 위로 올라가며 'pill-object-detection' 또는
           data/ + src/ 를 모두 가진 폴더
        4. 현재 작업 폴더
    """
    env = os.environ.get(ENV_PROJECT_KEY)
    if env and Path(env).is_dir():
        return Path(env).resolve()

    if IN_COLAB and Path(COLAB_PROJECT_ROOT).is_dir():
        return Path(COLAB_PROJECT_ROOT).resolve()

    here = Path(start).resolve() if start else Path.cwd().resolve()
    for cand in [here, *here.parents]:
        if cand.name == "pill-object-detection":
            return cand
        if (cand / "data").is_dir() and (cand / "src").is_dir():
            return cand
    return here


def _first_existing(*cands: Optional[PathLike]) -> Optional[Path]:
    for c in cands:
        if c and Path(c).is_dir():
            return Path(c).resolve()
    return None


def configure(
    project_root: Optional[PathLike] = None,
    dataset_dir: Optional[PathLike] = None,
    aug_dataset_dir: Optional[PathLike] = None,
    crops_dir: Optional[PathLike] = None,
    team_work_dir: Optional[PathLike] = None,
    cutout_check_dir: Optional[PathLike] = None,
    verbose: bool = True,
) -> Dict[str, Optional[str]]:
    """★ 모든 노트북의 첫 셀에서 한 번 호출하세요.

    인자를 주지 않으면 실행 환경(Colab / 로컬)을 감지해 프로젝트 표준 구조로
    경로를 채웁니다. 개별 인자만 덮어써도 됩니다.

        import pill_transforms as pt
        pt.configure()                                   # 자동
        pt.configure(project_root="/content/drive/MyDrive/.../pill-object-detection")
        pt.configure(crops_dir=".../team_work/cropped_output")

    Returns:
        설정된 경로 dict (존재하지 않는 항목은 그대로 두고 경고만 합니다).
    """
    global PROJECT_ROOT, DATASET_DIR, AUG_DATASET_DIR
    global CROPPED_PILLS_DIR, TEAM_WORK_DIR, CUTOUT_CHECK_DIR

    root = Path(project_root).resolve() if project_root else find_project_root()
    PROJECT_ROOT = str(root)

    DATASET_DIR = str(Path(dataset_dir).resolve()) if dataset_dir else str(
        root / "data" / "processed"
    )
    AUG_DATASET_DIR = str(Path(aug_dataset_dir).resolve()) if aug_dataset_dir else str(
        root / "data" / "processed_aug"
    )
    TEAM_WORK_DIR = str(Path(team_work_dir).resolve()) if team_work_dir else str(
        root / "data" / "team_work"
    )
    CUTOUT_CHECK_DIR = str(Path(cutout_check_dir).resolve()) if cutout_check_dir else str(
        root / "outputs" / "cutcheck"
    )

    if crops_dir:
        CROPPED_PILLS_DIR = str(Path(crops_dir).resolve())
    else:
        # 표준 위치 → AI Hub 팀 작업 결과 → 없으면 None(= train 박스에서 직접 컷아웃)
        found = _first_existing(
            root / "data" / "cropped_pills_review",
            Path(TEAM_WORK_DIR) / "cropped_output",
            root / "data" / "team_work" / "cropped_output",
        )
        CROPPED_PILLS_DIR = str(found) if found else None

    paths = current_paths()
    if verbose:
        print("═" * 66)
        print(f"  환경: {'Colab' if IN_COLAB else '로컬'}   |   "
              f"배경색 고정 {PILL_BG_HEX} RGB{PILL_BG_RGB}")
        print("═" * 66)
        check_paths(verbose=True)
    return paths


def current_paths() -> Dict[str, Optional[str]]:
    """현재 설정된 경로를 dict 로 돌려줍니다."""
    return {
        "PROJECT_ROOT": PROJECT_ROOT,
        "DATASET_DIR": DATASET_DIR,
        "AUG_DATASET_DIR": AUG_DATASET_DIR,
        "CROPPED_PILLS_DIR": CROPPED_PILLS_DIR,
        "TEAM_WORK_DIR": TEAM_WORK_DIR,
        "CUTOUT_CHECK_DIR": CUTOUT_CHECK_DIR,
    }


def ensure_deps(albumentations: bool = True, quiet: bool = True) -> Dict[str, bool]:
    """Colab 에서 부족한 패키지를 설치합니다 (노트북 첫 셀용, 선택).

        pt.ensure_deps()
    """
    out: Dict[str, bool] = {}
    want = ["opencv-python-headless"] if IN_COLAB else []
    if albumentations and not HAS_ALBUMENTATIONS:
        want.append("albumentations")
    for pkg in want:
        rc = os.system(f"{sys.executable} -m pip install {'-q ' if quiet else ''}{pkg}")
        out[pkg] = rc == 0
    if not want:
        print("필요한 패키지가 이미 설치돼 있습니다.")
    return out


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


def pad_fill_value() -> Union[int, Tuple[int, int, int]]:
    """★ letterbox 패딩 색 — 기본은 실측 고정 배경색(RGB).

    albumentations 파이프라인은 RGB 이미지를 받으므로 PILL_BG_RGB 를 씁니다.
    검정 패딩으로 되돌리려면 `pt.USE_BG_PAD = False`.
    """
    return tuple(PILL_BG_RGB) if USE_BG_PAD else 0


def _pad_transform(image_size: int):
    """PadIfNeeded 의 인자명이 albumentations 버전마다 달라 둘 다 시도합니다."""
    fill = pad_fill_value()
    common = dict(
        min_height=image_size,
        min_width=image_size,
        border_mode=cv2.BORDER_CONSTANT,
    )
    try:
        return A.PadIfNeeded(**common, fill=fill)       # 1.4.21+
    except TypeError:
        return A.PadIfNeeded(**common, value=fill)      # 구버전


def _rotate_transform(limit: float, p: float):
    fill = pad_fill_value()
    common = dict(limit=limit, border_mode=cv2.BORDER_CONSTANT, p=p)
    try:
        return A.Rotate(**common, fill=fill)
    except TypeError:
        return A.Rotate(**common, value=fill)


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
    2. LongestMaxSize                 : 976x1280 비율을 왜곡 없이 축소
    3. ★ CLAHE(p=1.0)                : **모든 이미지에 항상 적용** (패딩 앞)
    4. PadIfNeeded(letterbox)         : 여백을 고정 배경색 PILL_BG_RGB 로 채움
    5. HorizontalFlip / VerticalFlip / Rotate(소각도)
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

        # ★ CLAHE — 전 이미지 적용 (전처리이므로 확률 없음)
        #   ★ 반드시 패딩 **앞**에 둡니다. 패딩 뒤에 걸면 고정 배경색으로 채운
        #     여백이 타일 히스토그램에 섞여 가장자리 대비가 왜곡되고,
        #     패딩 색도 미세하게 달라집니다.
        A.CLAHE(clip_limit=CLAHE_CLIP, tile_grid_size=(CLAHE_GRID, CLAHE_GRID), p=1.0),

        _pad_transform(image_size),

        # 상하좌우 반전 — 각인 때문에 기본 비활성
        A.HorizontalFlip(p=hflip_p),
        A.VerticalFlip(p=vflip_p),

        # 회전 (소각도) — 여백은 고정 배경색으로 채웁니다
        _rotate_transform(limit=15, p=0.4),

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
        # ★ 학습과 동일한 CLAHE (패딩 앞)
        A.CLAHE(clip_limit=CLAHE_CLIP, tile_grid_size=(CLAHE_GRID, CLAHE_GRID), p=1.0),
        _pad_transform(image_size),
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


# ═══════════════════════════════════════════════════════════════════════════
#  Part 3.5. Faster R-CNN 어댑터 + 추론 letterbox 복원
#            (train.ipynb / predict.ipynb 가 같은 코드를 쓰도록 모듈로 이동)
# ═══════════════════════════════════════════════════════════════════════════

class FasterRCNNTransform:
    """Albumentations 결과를 Faster R-CNN 입력 형식으로 바꾸는 Adapter.

    ★ 예전에는 이 클래스를 train 노트북 안에 직접 정의했습니다. 그래서
      predict 노트북이 같은 전처리를 재현하지 못할 위험이 있었습니다.
      이제 두 노트북 모두 이 모듈에서 가져다 씁니다.

    출력 이미지: torch.Tensor [C, H, W] float32, 0~1
    (Normalize 는 걸지 않습니다 — torchvision detection 모델이 내부에서 수행)
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, bboxes, labels):
        transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
        img = transformed["image"]

        if isinstance(img, np.ndarray):
            import torch  # 지연 import — torch 없이도 모듈이 로드되도록

            t = torch.from_numpy(np.ascontiguousarray(img))
            if t.ndim == 3:
                t = t.permute(2, 0, 1)
            transformed["image"] = t.float() / 255.0
        return transformed

    def __repr__(self) -> str:  # 노트북 출력용
        return f"FasterRCNNTransform({self.transform})"


def get_frcnn_train_transform(image_size: int = 640, augment: bool = False,
                              **kwargs) -> FasterRCNNTransform:
    """Faster R-CNN 학습용 transform.

    Args:
        augment: False(기본) 면 baseline 과 동일하게 증강 없이
                 letterbox + CLAHE 만 적용합니다. True 면 온라인 증강을 켭니다.
    """
    base = (get_train_transforms(image_size=image_size, to_tensor=False, **kwargs)
            if augment else
            get_valid_transforms(image_size=image_size, to_tensor=False, **kwargs))
    return FasterRCNNTransform(base)


def get_frcnn_eval_transform(image_size: int = 640, **kwargs) -> FasterRCNNTransform:
    """Faster R-CNN 검증 / 추론용 transform (학습과 동일한 전처리)."""
    return FasterRCNNTransform(
        get_valid_transforms(image_size=image_size, to_tensor=False, **kwargs)
    )


def letterbox_meta(orig_h: int, orig_w: int, image_size: int = 640) -> Dict[str, float]:
    """LongestMaxSize + PadIfNeeded(center) 가 만든 좌표 변환 정보.

    `get_valid_transforms` 와 **동일한 규칙**으로 계산하므로, 추론 결과 박스를
    원본 이미지 좌표로 되돌릴 때 그대로 쓸 수 있습니다.
    """
    scale = float(image_size) / float(max(orig_h, orig_w))
    new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))
    return {
        "scale": scale,
        "pad_top": (image_size - new_h) // 2,
        "pad_left": (image_size - new_w) // 2,
        "new_h": new_h, "new_w": new_w,
        "orig_h": int(orig_h), "orig_w": int(orig_w),
        "image_size": int(image_size),
    }


def undo_letterbox_boxes(boxes_xyxy, meta: Dict[str, float]) -> np.ndarray:
    """★ 추론 박스([x1,y1,x2,y2], 640 기준)를 원본 해상도 좌표로 되돌립니다.

        pred = model([img_t])[0]
        boxes = pt.undo_letterbox_boxes(pred["boxes"].cpu().numpy(), meta)

    제출 파일은 원본 좌표를 요구하므로 이 단계를 빠뜨리면 mAP 가 0 이 됩니다.
    """
    b = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).copy()
    if b.size == 0:
        return b
    b[:, [0, 2]] -= float(meta["pad_left"])
    b[:, [1, 3]] -= float(meta["pad_top"])
    b /= float(meta["scale"])
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, meta["orig_w"] - 1)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, meta["orig_h"] - 1)
    return b


def prepare_image_for_inference(
    image: Union[PathLike, np.ndarray],
    image_size: int = 640,
    to_tensor: bool = True,
):
    """추론 1장 전처리. `(tensor 또는 ndarray, meta)` 를 돌려줍니다.

        img_t, meta = pt.prepare_image_for_inference("test_images/K-000001.png")
        with torch.no_grad():
            pred = model([img_t.to(device)])[0]
        boxes = pt.undo_letterbox_boxes(pred["boxes"].cpu().numpy(), meta)

    학습 때와 **완전히 같은** letterbox + CLAHE 를 적용합니다.
    """
    if isinstance(image, np.ndarray):
        rgb = image if image.shape[-1] == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        bgr = imread_unicode(image)
        if bgr is None:
            raise FileNotFoundError(f"이미지를 읽지 못했습니다: {image}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    meta = letterbox_meta(h, w, image_size=image_size)

    tf = get_valid_transforms(image_size=image_size, to_tensor=False)
    out = tf(image=rgb, bboxes=[], labels=[])["image"]

    if not to_tensor:
        return out, meta

    import torch  # 지연 import

    t = torch.from_numpy(np.ascontiguousarray(out)).permute(2, 0, 1).float() / 255.0
    return t, meta


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


def sample_bg_tone(
    crop_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Optional[Tuple[float, float]]:
    """★ 크롭 이미지의 **배경 색조만** 뽑습니다 → (H, S), 0~179 / 0~255.

    - mask 가 있으면 알약 바깥(mask==0) 픽셀을, 없으면 테두리 링을 배경으로 봅니다.
    - 밝기(V)는 **일부러 버립니다.** Copy&Paste 인공 배경은 이 색조에
      새로 뽑은 밝기를 입혀 만듭니다. (배경 색감은 유지 + 밝기는 다양화)
    - 색상(H)은 원형 값이라 산술평균이 아니라 벡터 평균으로 구합니다.
    """
    if crop_bgr is None or crop_bgr.ndim != 3:
        return None
    h, w = crop_bgr.shape[:2]
    if h < 8 or w < 8:
        return None

    hsv = cv2.cvtColor(crop_bgr[..., :3], cv2.COLOR_BGR2HSV)
    if mask is not None and mask.shape[:2] == (h, w) and int((mask == 0).sum()) >= 50:
        bg = mask == 0
    else:
        bg = np.zeros((h, w), bool)
        b = max(2, min(h, w) // 10)
        bg[:b] = bg[-b:] = True
        bg[:, :b] = bg[:, -b:] = True

    hh = hsv[..., 0][bg].astype(np.float32)
    ss = hsv[..., 1][bg].astype(np.float32)
    if hh.size < 20:
        return None

    ang = hh * (np.pi / 90.0)               # 0~179 → 0~2π
    hbar = math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean()))
    if hbar < 0:
        hbar += 2 * math.pi
    return (float(hbar * 90.0 / np.pi), float(np.median(ss)))


def _checkerboard(h: int, w: int, cell: int = 12) -> np.ndarray:
    """투명 영역 확인용 체커보드 배경."""
    yy, xx = np.mgrid[0:h, 0:w]
    tile = (((yy // cell) + (xx // cell)) % 2).astype(np.uint8)
    return np.dstack([np.where(tile == 0, 235, 200).astype(np.uint8)] * 3)


def cutout_panel(crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """검수용 3분할 패널: 원본 | 마스크 경계 | 체커보드 위 컷아웃."""
    h, w = mask.shape[:2]
    gap = np.full((h, 6, 3), 255, np.uint8)

    edge = crop_bgr.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edge, cnts, -1, (0, 0, 255), 2)

    a = mask.astype(np.float32)[..., None]
    comp = np.clip(_checkerboard(h, w).astype(np.float32) * (1 - a)
                   + crop_bgr.astype(np.float32) * a, 0, 255).astype(np.uint8)
    return np.hstack([crop_bgr, gap, edge, gap, comp])


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
        check_dir: Optional[PathLike] = None,   # ★ 컷아웃 검수 저장 폴더(cutcheck)
    ):
        self.cache_dir = Path(cache_dir)
        self.max_per_class = int(max_per_class)
        self.margin = float(margin)
        self.check_dir: Optional[Path] = Path(check_dir) if check_dir else None
        # ★ 인자 > 전역 CROPPED_PILLS_DIR 순으로 결정
        _cd = crops_dir if crops_dir is not None else CROPPED_PILLS_DIR
        self.crops_dir: Optional[Path] = Path(_cd) if _cd else None
        self.unmatched_folders: List[str] = []
        # ★ 크롭 이미지 배경에서 뽑은 색조 (H, S) 표본 — 인공 배경 생성에 사용
        self.bg_tones: List[Tuple[float, float]] = []
        self.class_names: Dict[int, str] = {}
        self.items: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
        self.box_stats: Dict[int, Dict[str, float]] = {}
        self.global_box: Dict[str, float] = {"w_med": 120.0, "h_med": 120.0}
        self.stats = Counter()
        self.fail_by_class = Counter()

    # ---------- 캐시 ----------
    def _cache_paths(self, cid) -> List[Path]:
        d = self.cache_dir / str(cid)
        return sorted(d.glob("*.png")) if d.is_dir() else []

    @property
    def _tone_json(self) -> Path:
        return self.cache_dir / "_bg_tones.json"

    def _save_tones(self) -> None:
        """★ 배경 색조 표본을 캐시에 남깁니다 (두 번째 실행에서 컷아웃을 건너뛰어도
        인공 배경 색감이 그대로 재현되도록)."""
        if not self.bg_tones:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._tone_json.write_text(
                json.dumps([[round(h, 3), round(s, 3)] for h, s in self.bg_tones]),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _load_tones(self) -> None:
        if self.bg_tones or not self._tone_json.exists():
            return
        try:
            self.bg_tones = [(float(a), float(b))
                             for a, b in json.loads(self._tone_json.read_text(encoding="utf-8"))]
        except Exception:
            self.bg_tones = []

    def _save_cache(self, cid, idx, bgr, mask):
        d = self.cache_dir / str(cid)
        d.mkdir(parents=True, exist_ok=True)
        rgba = np.dstack([bgr, (mask * 255).astype(np.uint8)])
        imwrite_unicode(d / f"{idx:04d}.png", rgba)

    def _save_check(self, cid, stem: str, crop, m, bgr, mask) -> None:
        """★ 컷아웃 검수물 저장 — cut/ (RGBA) + panel/ (원본|경계|투명배경)."""
        if self.check_dir is None:
            return
        try:
            name = self.class_names.get(int(cid)) if self.class_names else None
            folder = f"{int(cid)}_{name}" if name else str(int(cid))
            folder = re.sub(r'[\\/:*?"<>|]', "_", folder)[:80]
            base = self.check_dir / folder
            rgba = np.dstack([bgr, (mask * 255).astype(np.uint8)])
            imwrite_unicode(base / "cut" / f"{stem}.png", rgba)
            imwrite_unicode(base / "panel" / f"{stem}.png", cutout_panel(crop, m))
        except Exception:
            self.stats["검수 저장 실패"] += 1

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
        if names:
            self.class_names = {i: str(n) for i, n in enumerate(names)}
        if not rebuild:
            self._load_tones()          # ★ 캐시 재사용 시 배경 색조도 복원

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

                tone = sample_bg_tone(crop, m)          # ★ 배경 색조만 수집
                if tone is not None:
                    self.bg_tones.append(tone)

                self._save_cache(cid, idx, bgr, mask)
                self._save_check(cid, f"{Path(src).stem}_{idx:03d}", crop, m, bgr, mask)
                idx += 1
                out.append((bgr, mask))
                self.stats["신규 컷아웃"] += 1

            if out:
                self.items[cid] = out
            if verbose and n_done % 20 == 0:
                print(f"    크롭 라이브러리 {n_done}/{len(classes)} 클래스")

        self._save_tones()
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

                # ★ 이 크롭 이미지의 배경 색조(H,S)만 저장 → 인공 배경 재료
                tone = sample_bg_tone(crop, m)
                if tone is not None:
                    self.bg_tones.append(tone)

                self._save_cache(cid, idx, bgr, mask)
                self._save_check(cid, p.stem, crop, m, bgr, mask)
                idx += 1
                out.append((bgr, mask))
                self.stats["신규 크롭"] += 1

            if out:
                self.items[cid] = out
            if verbose and n_done % 20 == 0:
                print(f"    크롭 라이브러리 {n_done}/{len(subdirs)} 폴더")

        self._save_tones()
        if verbose:
            print(f"    배경 색조 표본 {len(self.bg_tones):,}개 수집 "
                  f"(Copy&Paste 인공 배경에 사용)")
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


def export_cutout_check(
    crops_dir: Optional[PathLike] = None,
    out_dir: Optional[PathLike] = None,
    names: Optional[Sequence[str]] = None,
    *,
    max_per_class: int = 0,          # 0 = 전부
    save_panel: bool = True,
    save_failed: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """★ `cropped_pills_review` 전체를 컷아웃해 검수 폴더에 정리합니다.

    기본 출력 = `CUTOUT_CHECK_DIR` (= D:/PillData/cutcheck)

        cutcheck/
        ├── {category_id}_{클래스명}/
        │   ├── cut/     ← 배경 투명(RGBA) 컷아웃 — Copy&Paste 에 실제로 붙는 그림
        │   └── panel/   ← 원본 | 경계 | 체커보드 합성 (눈으로 검수)
        ├── _failed/     ← 컷아웃 실패 원본 (파라미터 조정 대상)
        ├── cutout_report.csv    파일 단위 결과
        └── cutout_summary.csv   클래스 단위 성공률

    Returns:
        {"n_ok", "n_fail", "n_class", "out_dir", "report_csv", "summary_csv"}
    """
    crops_dir = Path(crops_dir or CROPPED_PILLS_DIR or "")
    out_dir = Path(out_dir or CUTOUT_CHECK_DIR or "")
    if not crops_dir.is_dir():
        raise FileNotFoundError(
            f"크롭 폴더를 찾을 수 없습니다: {crops_dir}\n"
            f"→ CROPPED_PILLS_DIR 경로를 확인하세요."
        )
    if str(out_dir) in ("", "."):
        raise ValueError("out_dir(CUTOUT_CHECK_DIR)를 지정하세요.")

    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lut = _build_name_lookup(names) if names else {}
    subdirs = sorted([d for d in crops_dir.iterdir() if d.is_dir()])
    if not subdirs:                                   # 하위 폴더가 없으면 평면 구조
        subdirs = [crops_dir]

    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    per_class = defaultdict(lambda: {"ok": 0, "fail": 0})
    tones: List[Tuple[float, float]] = []
    stats = Counter()

    for n_done, d in enumerate(subdirs, 1):
        cid = _match_class_id(d.name, lut) if lut else None
        label = f"{cid}_{names[cid]}" if (cid is not None and names) else d.name
        folder = re.sub(r'[\\/:*?"<>|]', "_", str(label))[:80]
        dst = out_dir / folder

        files = sorted(p for p in d.iterdir()
                       if p.is_file() and p.suffix.lower() in CROPPED_PILLS_EXTS)
        if max_per_class:
            files = files[:max_per_class]

        for p in files:
            raw = imread_unicode_unchanged(p)
            if raw is None:
                stats["읽기 실패"] += 1
                rows.append({"folder": d.name, "file": p.name, "class_id": cid,
                             "status": "read_fail"})
                per_class[label]["fail"] += 1
                continue
            if raw.ndim == 2:
                raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

            m, how = None, "chroma/grabcut"
            if CROPPED_USE_ALPHA and raw.ndim == 3 and raw.shape[2] == 4:
                crop = raw[..., :3].copy()
                if raw[..., 3].min() < 250:
                    m, how = (raw[..., 3] > 127).astype(np.uint8), "alpha"
            else:
                crop = raw[..., :3].copy() if raw.ndim == 3 else raw
            if m is None:
                m = cutout_pill(crop)

            if m is None or min(crop.shape[:2]) < 16:
                stats["컷아웃 실패"] += 1
                per_class[label]["fail"] += 1
                rows.append({"folder": d.name, "file": p.name, "class_id": cid,
                             "status": "cutout_fail"})
                if save_failed:
                    imwrite_unicode(out_dir / "_failed" / folder / p.name, crop)
                continue

            ys, xs = np.where(m > 0)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            bgr, mask = crop[y0:y1, x0:x1].copy(), m[y0:y1, x0:x1].copy()

            rgba = np.dstack([bgr, (mask * 255).astype(np.uint8)])
            imwrite_unicode(dst / "cut" / f"{p.stem}.png", rgba)
            if save_panel:
                imwrite_unicode(dst / "panel" / f"{p.stem}.png", cutout_panel(crop, m))

            tone = sample_bg_tone(crop, m)
            if tone:
                tones.append(tone)

            stats["성공"] += 1
            per_class[label]["ok"] += 1
            rows.append({
                "folder": d.name, "file": p.name, "class_id": cid, "status": "ok",
                "method": how,
                "src_w": crop.shape[1], "src_h": crop.shape[0],
                "cut_w": bgr.shape[1], "cut_h": bgr.shape[0],
                "fill_ratio": round(float(mask.mean()), 4),
                "bg_hue": round(tone[0], 2) if tone else "",
                "bg_sat": round(tone[1], 2) if tone else "",
            })

        if verbose and (n_done % 10 == 0 or n_done == len(subdirs)):
            print(f"    컷아웃 검수 {n_done}/{len(subdirs)} 폴더  "
                  f"(성공 {stats['성공']:,} / 실패 {stats['컷아웃 실패']:,})")

    # ---------- 리포트 ----------
    report_csv = out_dir / "cutout_report.csv"
    cols = ["folder", "file", "class_id", "status", "method",
            "src_w", "src_h", "cut_w", "cut_h", "fill_ratio", "bg_hue", "bg_sat"]
    import csv as _csv
    with open(report_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    summary_csv = out_dir / "cutout_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.writer(f)
        w.writerow(["class_folder", "성공", "실패", "성공률"])
        for k in sorted(per_class):
            ok, ng = per_class[k]["ok"], per_class[k]["fail"]
            w.writerow([k, ok, ng, round(ok / max(ok + ng, 1), 3)])

    # 배경 색조 표본도 남겨 둡니다 (Copy&Paste 배경 재현용)
    if tones:
        (out_dir / "bg_tones.json").write_text(
            json.dumps([[round(h, 3), round(s, 3)] for h, s in tones]), encoding="utf-8"
        )

    n_ok, n_fail = stats["성공"], stats["컷아웃 실패"] + stats["읽기 실패"]
    if verbose:
        print("\n■ 컷아웃 검수 결과")
        print(f"  성공 {n_ok:,}장 / 실패 {n_fail:,}장  "
              f"(성공률 {n_ok / max(n_ok + n_fail, 1):.1%})")
        print(f"  클래스 폴더 {len(per_class):,}개")
        print(f"  배경 색조 표본 {len(tones):,}개 → {out_dir / 'bg_tones.json'}")
        print(f"  저장 위치 {out_dir}")
        print(f"  리포트 {report_csv.name} / {summary_csv.name}")
        print(f"  소요 {time.time() - t0:.1f}초")
        if n_fail:
            print("  ※ 실패분은 _failed 폴더에 있습니다. "
                  "CUT_MIN_CHROMA 를 5.0, CUT_CHROMA_K 를 1.6 으로 낮추면 대개 줄어듭니다.")

    return {"n_ok": n_ok, "n_fail": n_fail, "n_class": len(per_class),
            "out_dir": str(out_dir), "report_csv": str(report_csv),
            "summary_csv": str(summary_csv), "bg_tones": tones}


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

        # ★ 크롭 이미지에서 모은 배경 색조 표본 (없으면 랜덤 배경으로 fallback)
        self.bg_tones: List[Tuple[float, float]] = list(getattr(library, "bg_tones", []) or [])

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
    def random_canvas(H, W, rng):
        """예전 방식 — 완전 랜덤 단색 + 노이즈 + 밝기 기울기 (fallback)."""
        base = np.array([rng.randint(110, 245) for _ in range(3)], np.float32)
        canvas = np.ones((H, W, 3), np.float32) * base
        gy = np.linspace(rng.uniform(-14, 14), rng.uniform(-14, 14), H)[:, None]
        gx = np.linspace(rng.uniform(-14, 14), rng.uniform(-14, 14), W)[None, :]
        canvas += (gy + gx)[..., None]
        canvas += np.random.normal(0, 5, (H, W, 3))
        return np.clip(canvas, 0, 255).astype(np.uint8)

    @staticmethod
    def fixed_canvas(H, W, rng):
        """★ 기본 배경 — 팀 crop 에서 실측한 PILL_BG_BGR 고정색으로 채웁니다.

        숫자는 모두 Part 0-2 에 상수로 박혀 있어 누가 언제 실행해도 같은
        배경이 나옵니다. 완전한 단색이 되지 않도록 실측한 만큼의
        밝기 기울기(PILL_BG_GRAD)와 노이즈(PILL_BG_NOISE_STD)만 얹습니다.
        (둘 다 0 으로 두면 완벽한 단색 배경이 됩니다)
        """
        canvas = np.empty((H, W, 3), np.float32)
        canvas[..., 0] = float(PILL_BG_BGR[0])
        canvas[..., 1] = float(PILL_BG_BGR[1])
        canvas[..., 2] = float(PILL_BG_BGR[2])

        if PILL_BG_V_JITTER:
            canvas += rng.uniform(-PILL_BG_V_JITTER, PILL_BG_V_JITTER)
        if PILL_BG_GRAD:
            g = PILL_BG_GRAD
            gy = np.linspace(rng.uniform(-g, g), rng.uniform(-g, g), H)[:, None]
            gx = np.linspace(rng.uniform(-g, g), rng.uniform(-g, g), W)[None, :]
            canvas += (gy + gx)[..., None]
        if PILL_BG_NOISE_STD:
            canvas += np.random.normal(0.0, PILL_BG_NOISE_STD, (H, W, 3))
        return np.clip(canvas, 0, 255).astype(np.uint8)

    def blank_canvas(self, H, W, rng):
        """인공 배경을 만듭니다. `CP_BG_MODE` 로 방식을 고릅니다.

            "fixed"  ★ 기본 — 실측 고정 배경색 (재현성 최고)
            "crops"  크롭 이미지 배경의 색조(H,S)만 표본에서 뽑아 씀
            "random" 완전 무작위 단색
        """
        mode = str(CP_BG_MODE).lower()

        if mode == "fixed":
            self.stats["고정 배경"] += 1
            return self.fixed_canvas(H, W, rng)
        if mode == "random":
            self.stats["랜덤 배경"] += 1
            return self.random_canvas(H, W, rng)

        # mode == "crops" — 표본이 없으면 고정 배경으로 되돌아갑니다.
        if not (CP_BG_FROM_CROPS and self.bg_tones):
            self.stats["고정 배경(대체)"] += 1
            return self.fixed_canvas(H, W, rng)

        hue, sat = self.bg_tones[rng.randrange(len(self.bg_tones))]
        hue = (hue + rng.uniform(-CP_BG_HUE_JITTER, CP_BG_HUE_JITTER)) % 180.0
        sat = float(np.clip(sat * rng.uniform(*CP_BG_S_SCALE), 0.0, 255.0))
        val = float(rng.uniform(*CP_BG_V_RANGE))          # ★ 밝기는 새로 뽑음

        g = CP_BG_GRAD
        gy = np.linspace(rng.uniform(-g, g), rng.uniform(-g, g), H)[:, None]
        gx = np.linspace(rng.uniform(-g, g), rng.uniform(-g, g), W)[None, :]

        hsv = np.empty((H, W, 3), np.float32)
        hsv[..., 0] = hue
        hsv[..., 1] = sat
        hsv[..., 2] = np.clip(
            val + gy + gx + np.random.normal(0, CP_BG_NOISE, (H, W)), 0, 255
        )
        canvas = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        canvas += np.random.normal(0, CP_BG_NOISE, (H, W, 3))
        self.stats["색조 배경"] += 1
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
    src_root: Optional[PathLike] = None,
    dst_root: Optional[PathLike] = None,
    geom_mult: Optional[int] = None,
    n_synth: Optional[int] = None,
    *,
    use_flip: Optional[bool] = None,
    cp_mode: Optional[str] = None,
    cp_weighted: bool = True,
    max_crops_per_class: Optional[int] = None,
    crops_dir: Optional[PathLike] = None,   # ★ cropped_pills_review 경로
    cutout_check_dir: Optional[PathLike] = None,   # ★ 컷아웃 검수 저장(cutcheck)
    rebuild_crop_cache: bool = False,
    preprocess_val_test: bool = True,
    force: bool = False,          # ★ True 면 사전검증 실패해도 강행
    seed: Optional[int] = None,
    overwrite: bool = True,
    verbose: bool = True,
) -> str:
    """YOLO 데이터셋을 읽어 **증강된 새 YOLO 데이터셋**을 만들고 data.yaml 경로를 반환합니다.

    Args:
        src_root: `pill_detection_dataset.ipynb` 가 만든 폴더.
                  images/train, labels/train, data.yaml 이 필수이고
                  val/test 는 있으면 쓰고 없으면 건너뜁니다.
        dst_root: 결과 폴더. None 이면 `<src_root>_aug`.
        geom_mult: ★ train 이미지 1장당 최종 장수. 3 이면 원본 1 + 증강 2.
                   1 이면 기하 증강 없음(원본만).
        n_synth:   ★ Copy & Paste 합성 이미지 수. 0 이면 끔.
        crops_dir: ★ `cropped_pills_review` 폴더 경로. 지정하면 Copy&Paste 재료를
                   train 이미지에서 컷아웃하지 않고 이 폴더의 크롭 이미지로 씁니다.
                   None 이면 전역 `CROPPED_PILLS_DIR` 을 따릅니다.
        cp_weighted: True 면 희소 클래스를 더 자주 붙여 불균형을 완화합니다.
        preprocess_val_test: train 이외 split 에도 동일 전처리(WB+CLAHE)를 적용할지.
                   ★ train 에만 걸면 분포가 어긋나므로 기본 True 를 권장합니다.

    Returns:
        생성된 `data.yaml` 의 절대 경로 문자열.

    주의:
        train 이외 split 에는 **증강을 적용하지 않습니다.** 전처리만 동일하게 겁니다.
        ★ val 폴더가 비어 있으면(= 홀드아웃 없이 전체를 train 으로 쓰는 구성)
          data.yaml 의 `val:` 이 자동으로 `images/train` 을 가리킵니다.
          Ultralytics 가 val 경로를 요구하기 때문이며, 이때 val 지표는
          학습 데이터에 대한 점수이므로 일반화 성능이 아닙니다.
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

    # ★ 경로를 안 넘기면 configure() 가 설정한 팀 표준 경로를 씁니다.
    if src_root is None:
        if DATASET_DIR is None:
            configure(verbose=False)
        src_root = DATASET_DIR
    if dst_root is None and AUG_DATASET_DIR:
        dst_root = AUG_DATASET_DIR

    src_root = Path(src_root).resolve()
    dst_root = Path(dst_root).resolve() if dst_root else src_root.parent / f"{src_root.name}_aug"

    if geom_mult < 1:
        raise ValueError("geom_mult 는 1 이상이어야 합니다. (1 = 증강 없음)")
    if n_synth < 0:
        raise ValueError("n_synth 는 0 이상이어야 합니다. (0 = Copy&Paste 끔)")

    # ---------- ★ 사전검증 → 실행 (팀 공유 가이드와 동일한 2단계) ----------
    if not force:
        report = preflight(
            src_root=src_root, dst_root=dst_root, crops_dir=crops_dir,
            geom_mult=geom_mult, n_synth=n_synth, verbose=verbose,
        )
        if not report["ok"]:
            raise RuntimeError(
                "사전검증(preflight) 실패 — 실제 생성을 중단했습니다.\n"
                + "\n".join(f"  · {m}" for m in report["errors"])
                + "\n→ 위 항목을 고친 뒤 다시 실행하세요. "
                  "(검증을 건너뛰려면 force=True)"
            )

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
        _check_dir = cutout_check_dir if cutout_check_dir is not None else CUTOUT_CHECK_DIR
        lib = PillCropLibrary(
            cache_dir=dst_root / "_crop_cache",
            max_per_class=max_crops_per_class,
            crops_dir=_crops_dir,
            check_dir=_check_dir,
        ).build(train_recs, rebuild=rebuild_crop_cache, verbose=verbose, names=names)
        if verbose and _check_dir:
            print(f"    ★ 컷아웃 검수 저장 → {_check_dir}")

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

    # ---------- 5. train 이외 split — 증강 없이 전처리만 ----------
    #  ★ images/ 아래에 train 이 아닌 폴더가 있으면 전부 처리합니다.
    #     (val, test 외에 02 가 만드는 'all' = 원본 전체 split 포함)
    extra_splits = []
    _img_root = src_root / "images"
    if _img_root.is_dir():
        extra_splits = [d.name for d in sorted(_img_root.iterdir())
                        if d.is_dir() and d.name not in ("train", "val", "test")]

    split_recs = [("val", val_recs), ("test", test_recs)]
    for sp in extra_splits:
        (dst_root / "images" / sp).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / sp).mkdir(parents=True, exist_ok=True)
        split_recs.append((sp, _collect_split(src_root, sp)))

    if verbose:
        _named = [s_ for s_, r_ in split_recs if r_]
        print("[5/5] " + (" / ".join(_named) + " 복사 (증강 없음, 전처리만)"
                          if _named else "train 이외 split 없음 — 건너뜀"))

    for split, recs in split_recs:
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
    # ★ 홀드아웃(val/test)이 없는 구성 — 원본 전부를 train 으로 쓰는 경우
    #   Ultralytics 는 val 경로가 반드시 있어야 하므로 train 을 그대로 가리킵니다.
    #   이때 val 지표는 "학습 데이터에 대한 점수"이므로 낙관적입니다 (일반화 성능 아님).
    val_from_train = counts["val"] == 0
    if val_from_train and verbose:
        print("\n⚠️  val 이미지가 없습니다 — data.yaml 의 val 을 images/train 으로 지정합니다.")
        print("    (전체를 train 으로 쓰는 구성. val mAP 는 학습 데이터 점수이니 참고용으로만 보세요)")

    yaml_path = dst_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {dst_root}\n")
        f.write("train: images/train\n")
        f.write("val: images/train\n" if val_from_train else "val: images/val\n")
        if counts["test"]:
            f.write("test: images/test\n")
        for sp in extra_splits:
            f.write(f"{sp}: images/{sp}\n")
        f.write(f"nc: {len(names)}\nnames:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")

    info = {
        "source": str(src_root),
        "geom_mult": geom_mult,
        "n_synth": n_synth,
        "cp_mode": cp_mode,
        "crops_dir": str(crops_dir if crops_dir is not None else CROPPED_PILLS_DIR or ""),
        "cutout_check_dir": str(cutout_check_dir if cutout_check_dir is not None
                                else CUTOUT_CHECK_DIR or ""),
        "cp_bg_mode": CP_BG_MODE,
        "cp_bg_fixed_rgb": list(PILL_BG_RGB),
        "cp_bg_fixed_hex": PILL_BG_HEX,
        "pad_fill_rgb": list(PILL_BG_RGB) if USE_BG_PAD else [0, 0, 0],
        "cp_bg_from_crops": CP_BG_FROM_CROPS,
        "cp_weighted": cp_weighted,
        "use_flip": use_flip,
        "preprocess": {
            "white_balance": USE_WHITE_BALANCE,
            "clahe": USE_CLAHE,
            "clahe_clip": CLAHE_CLIP,
            "clahe_grid": CLAHE_GRID,
            "applied_to": "train/val/test 전부",
        },
        "counts": dict(counts),
        "box_counts": dict(box_counts),
        "val_from_train": val_from_train,   # ★ 홀드아웃 없이 전체를 train 으로 쓴 구성
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
        if val_from_train:
            print("  val    없음 → data.yaml 의 val 이 images/train 을 가리킵니다")
        else:
            print(f"  val    {counts['val']:,}장 / test {counts['test']:,}장")
        for sp in extra_splits:
            print(f"  {sp:<6} {counts[sp]:,}장  (04·05 가 쓰는 원본 전체 split)")
        ok = (counts["orig"] + counts["aug"]) == target
        print(f"\n  ★ 기하 증강 검산 {counts['orig'] + counts['aug']:,} / 목표 {target:,}장 "
              f"{'✅' if ok else '⚠️ 박스 전멸로 일부 손실'}")
        if n_synth:
            print(f"  ★ Copy&Paste  {counts['cp']:,} / 목표 {n_synth:,}장 "
                  f"{'✅' if counts['cp'] == n_synth else '⚠️ 배치 실패분 손실'}")
        print(f"\n  전처리(WB={USE_WHITE_BALANCE}, CLAHE={USE_CLAHE}) → 모든 split 에 적용 ✅")
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
#  셀프 테스트 — 주피터에서도 `pt.selftest()` 로 바로 돌릴 수 있습니다.
# ═══════════════════════════════════════════════════════════════════════════

def check_paths(verbose: bool = True) -> Dict[str, bool]:
    """★ 노트북 첫 셀에서 경로가 실제로 존재하는지 확인합니다.

        import pill_transforms as pt
        pt.configure()      # 내부에서 이 함수를 호출합니다
        pt.check_paths()
    """
    if PROJECT_ROOT is None:
        configure(verbose=False)

    required = {"DATASET_DIR"}
    optional_note = {
        "CUTOUT_CHECK_DIR": "실행할 때 자동으로 만들어집니다",
        "AUG_DATASET_DIR": "증강 실행 시 만들어집니다",
        "TEAM_WORK_DIR": "AI Hub 추가 데이터를 쓸 때만 필요합니다",
        "CROPPED_PILLS_DIR": "없으면 train 라벨 박스에서 직접 컷아웃합니다",
    }

    out: Dict[str, bool] = {}
    for k, v in current_paths().items():
        exists = bool(v) and Path(v).is_dir()
        out[k] = exists
        if verbose:
            if exists:
                mark = "✅"
            elif k in required:
                mark = "❌ 없음(필수)"
            else:
                mark = "— " + optional_note.get(k, "")
            print(f"  {k:<18} {v}  {mark}")
    if verbose:
        print(f"  {'배경색(고정)':<18} {PILL_BG_HEX}  RGB{PILL_BG_RGB} / "
              f"BGR{PILL_BG_BGR}  mode={CP_BG_MODE}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Part 8. ★ 사전검증(preflight) → 실행(execute)
#            AI Hub 팀 공유 가이드와 동일한 2단계 흐름입니다.
# ═══════════════════════════════════════════════════════════════════════════

def _disp_width(text: str) -> int:
    """한글·전각 문자를 2칸으로 계산합니다 (표 정렬용)."""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int, right: bool = False) -> str:
    gap = " " * max(0, width - _disp_width(text))
    return gap + text if right else text + gap


def _fmt_row(label: str, value: Any, criterion: str = "") -> str:
    return f"  {_pad(label, 26)}{_pad(str(value), 12, right=True)}   {criterion}"


def preflight(
    src_root: Optional[PathLike] = None,
    dst_root: Optional[PathLike] = None,
    crops_dir: Optional[PathLike] = None,
    geom_mult: Optional[int] = None,
    n_synth: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """★ 1단계 — 파일을 하나도 만들지 않고 입력만 검증합니다.

        import pill_transforms as pt
        pt.configure()
        rep = pt.preflight()
        rep["ok"]        # True 면 build_augmented_yolo_dataset() 실행 가능

    반드시 0 이어야 하는 항목
        · bbox 오류 수
        · 이미지 읽기 오류 수
        · 클래스 ID 범위 오류 수
        · 라벨 파일 없는 이미지 수
    허용되는 항목
        · 크롭 폴더 미매칭(그만큼 Copy&Paste 재료가 줄어듭니다)
        · 기존 출력 폴더 존재(overwrite=True 로 덮어씁니다)
    """
    if src_root is None:
        if DATASET_DIR is None:
            configure(verbose=False)
        src_root = DATASET_DIR
    if dst_root is None and AUG_DATASET_DIR:
        dst_root = AUG_DATASET_DIR
    if geom_mult is None:
        geom_mult = DEFAULT_GEOM_MULT
    if n_synth is None:
        n_synth = DEFAULT_N_SYNTH

    src_root = Path(src_root)
    dst_root = Path(dst_root) if dst_root else src_root.parent / f"{src_root.name}_aug"
    _cd = crops_dir if crops_dir is not None else CROPPED_PILLS_DIR
    crops_path = Path(_cd) if _cd else None

    rep: Dict[str, Any] = {
        "src_root": str(src_root), "dst_root": str(dst_root),
        "crops_dir": str(crops_path) if crops_path else None,
        "geom_mult": int(geom_mult), "n_synth": int(n_synth),
        "errors": [], "warnings": [],
    }

    if verbose:
        print("═" * 66)
        print("  사전검증 (preflight) — 파일을 만들지 않습니다")
        print(f"  입력 {src_root}")
        print(f"  출력 {dst_root}")
        print("═" * 66)

    # ---------- 1. 필수 폴더 ----------
    img_dir, lbl_dir = src_root / "images" / "train", src_root / "labels" / "train"
    if not img_dir.is_dir():
        rep["errors"].append(
            f"{img_dir} 가 없습니다 → pill_detection_dataset.ipynb 의 "
            "YOLO 내보내기 셀을 먼저 실행하세요."
        )
    if not lbl_dir.is_dir():
        rep["errors"].append(f"{lbl_dir} 가 없습니다.")
    if rep["errors"]:
        rep["ok"] = False
        if verbose:
            for m in rep["errors"]:
                print("  ❌ " + m)
        return rep

    # ---------- 2. data.yaml / 클래스 ----------
    names: List[str] = []
    yaml_path = src_root / "data.yaml"
    if yaml_path.exists():
        try:
            import yaml

            cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            raw = cfg.get("names", [])
            names = ([raw[k] for k in sorted(raw, key=lambda v: int(v))]
                     if isinstance(raw, dict) else list(raw))
        except Exception as e:
            rep["warnings"].append(f"data.yaml 을 읽지 못했습니다: {e}")
    else:
        rep["warnings"].append("data.yaml 이 없습니다 (클래스 이름 없이 진행)")
    rep["num_classes_yaml"] = len(names)

    # ---------- 3. 이미지 · 라벨 검증 ----------
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    imgs = [q for q in sorted(img_dir.iterdir()) if q.suffix.lower() in exts]
    n_img = len(imgs)
    n_read_err = n_missing_lbl = n_empty_lbl = n_bbox_err = n_cls_err = 0
    n_box = 0
    seen_cls: set = set()
    sizes: Counter = Counter()

    for q in imgs:
        im = imread_unicode(q)
        if im is None:
            n_read_err += 1
            continue
        H, W = im.shape[:2]
        sizes[(W, H)] += 1
        lp = lbl_dir / f"{q.stem}.txt"
        if not lp.exists():
            n_missing_lbl += 1
            continue
        raw_lines = [ln for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not raw_lines:
            n_empty_lbl += 1
            continue
        for ln in raw_lines:
            parts = ln.split()
            if len(parts) < 5:
                n_bbox_err += 1
                continue
            try:
                c = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
            except ValueError:
                n_bbox_err += 1
                continue
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0) or bw <= 0 or bh <= 0 \
                    or bw > 1.0 or bh > 1.0:
                n_bbox_err += 1
                continue
            if c < 0 or (names and c >= len(names)):
                n_cls_err += 1
                continue
            seen_cls.add(c)
            n_box += 1

    rep.update({
        "train_images": n_img, "train_boxes": n_box,
        "image_read_errors": n_read_err, "missing_labels": n_missing_lbl,
        "empty_labels": n_empty_lbl, "bbox_errors": n_bbox_err,
        "class_id_errors": n_cls_err, "classes_in_train": len(seen_cls),
        "image_sizes": {f"{w}x{h}": c for (w, h), c in sizes.most_common(5)},
    })

    if n_img == 0:
        rep["errors"].append("train 이미지가 0장입니다.")
    if n_read_err:
        rep["errors"].append(f"이미지 읽기 오류 {n_read_err}장")
    if n_bbox_err:
        rep["errors"].append(f"bbox 오류 {n_bbox_err}건 (좌표가 0~1 정규화값인지 확인)")
    if n_cls_err:
        rep["errors"].append(f"클래스 ID 범위 오류 {n_cls_err}건 (data.yaml 의 nc 와 불일치)")
    if n_missing_lbl:
        rep["errors"].append(f"라벨 파일이 없는 이미지 {n_missing_lbl}장")
    if n_empty_lbl:
        rep["warnings"].append(f"박스가 0개인 라벨 {n_empty_lbl}장 (배경 이미지로 처리)")

    # ---------- 4. val / test ----------
    for sp in ("val", "test"):
        d = src_root / "images" / sp
        rep[f"{sp}_images"] = len([q for q in d.iterdir() if q.suffix.lower() in exts]) \
            if d.is_dir() else 0
    if rep["val_images"] == 0:
        rep["warnings"].append(
            "val 이미지가 없습니다 → data.yaml 의 val 이 images/train 을 가리킵니다 "
            "(val 지표는 학습 데이터 점수이므로 일반화 성능이 아닙니다)"
        )

    # ---------- 5. Copy & Paste 재료 ----------
    matched = unmatched = n_crop_files = 0
    unmatched_names: List[str] = []
    if n_synth and crops_path is not None:
        if not crops_path.is_dir():
            rep["warnings"].append(
                f"크롭 폴더가 없습니다: {crops_path} → train 라벨 박스에서 직접 컷아웃합니다"
            )
        else:
            lut = _build_name_lookup(names) if names else {}
            for d in sorted(x for x in crops_path.iterdir() if x.is_dir()):
                files = [f for f in d.iterdir()
                         if f.suffix.lower() in CROPPED_PILLS_EXTS]
                if not files:
                    continue
                cid = _match_class_id(d.name, lut) if lut else None
                if cid is None:
                    unmatched += 1
                    if len(unmatched_names) < 5:
                        unmatched_names.append(d.name)
                else:
                    matched += 1
                    n_crop_files += len(files)
            if matched == 0:
                rep["warnings"].append(
                    "크롭 폴더의 클래스를 하나도 매칭하지 못했습니다 "
                    "(폴더명이 '{category_id}_{category_name}' 형식인지 확인)"
                )
            if unmatched:
                rep["warnings"].append(
                    f"매칭 실패 폴더 {unmatched}개 예: {', '.join(unmatched_names)}"
                )
    rep.update({"crops_classes_matched": matched, "crops_classes_unmatched": unmatched,
                "crops_files": n_crop_files})

    # ---------- 6. 출력 충돌 ----------
    existing = 0
    d = dst_root / "images" / "train"
    if d.is_dir():
        existing = len([q for q in d.iterdir() if q.suffix.lower() in exts])
        if existing:
            rep["warnings"].append(
                f"기존 출력 {existing}장이 있습니다 → overwrite=True 면 폴더를 비우고 다시 만듭니다"
            )
    rep["existing_output_images"] = existing

    # ---------- 7. 예상 산출량 ----------
    rep["expected_train_images"] = n_img * int(geom_mult) + int(n_synth)
    rep["ok"] = not rep["errors"]

    if verbose:
        print(_fmt_row("train 이미지 수", f"{n_img:,}", ""))
        print(_fmt_row("train bbox 수", f"{n_box:,}", ""))
        print(_fmt_row("data.yaml 클래스 수", len(names), "≥ 1"))
        print(_fmt_row("train 등장 클래스 수", len(seen_cls), "data.yaml 이하"))
        print(_fmt_row("이미지 읽기 오류", n_read_err, "0 이어야 실행 가능"))
        print(_fmt_row("라벨 없는 이미지", n_missing_lbl, "0 이어야 실행 가능"))
        print(_fmt_row("bbox 오류", n_bbox_err, "0 이어야 실행 가능"))
        print(_fmt_row("클래스 ID 오류", n_cls_err, "0 이어야 실행 가능"))
        print(_fmt_row("val / test 이미지", f"{rep['val_images']:,} / {rep['test_images']:,}",
                       "0 이어도 됨"))
        print(_fmt_row("크롭 클래스 매칭", f"{matched} / {matched + unmatched}",
                       "미매칭은 허용(재료만 감소)"))
        print(_fmt_row("크롭 이미지 수", f"{n_crop_files:,}", ""))
        print(_fmt_row("기존 출력 이미지", f"{existing:,}", "overwrite=True 면 재생성"))
        print(_fmt_row("배경 모드", CP_BG_MODE, f"{PILL_BG_HEX} RGB{PILL_BG_RGB}"))
        print(_fmt_row("예상 생성 train 수",
                       f"{rep['expected_train_images']:,}",
                       f"{n_img:,}×{geom_mult} + CP {n_synth:,}"))
        for m in rep["warnings"]:
            print("  ⚠️  " + m)
        for m in rep["errors"]:
            print("  ❌ " + m)
        print("─" * 66)
        print("  ✅ 사전검증 통과 — 실제 생성을 실행할 수 있습니다."
              if rep["ok"] else
              "  ❌ 사전검증 실패 — 위 항목을 고친 뒤 다시 실행하세요.")
    return rep


def bg_check(image_path: PathLike, verbose: bool = True) -> Dict[str, Any]:
    """★ 고정 배경색이 내 crop 이미지와 실제로 맞는지 확인합니다.

        python pill_transforms.py --bg-check "/path/to/3351_.../K-0033....png"

    상단 라벨바와 테두리를 제외한 뒤 알약 외접 타원 바깥 픽셀의 median 을
    구해 `PILL_BG_BGR` 과 비교합니다. 채널 차이가 12 이내면 통과로 봅니다.
    """
    img = imread_unicode(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽지 못했습니다: {image_path}")

    h, w = img.shape[:2]
    # 상단 라벨바(어두운 단색 띠) 제거
    rows = np.where(np.abs(img.reshape(h, -1, 3).mean(axis=1)
                           - np.array(CROP_LABEL_BAR_RGB[::-1])).max(axis=1) < 30)[0]
    y0 = int(rows.max()) + 1 if rows.size and rows.min() == 0 else 0
    # 파란 테두리 여유분 제거
    m = 4
    inner = img[y0 + m:h - m, m:w - m]
    if inner.size == 0 or min(inner.shape[:2]) < 16:
        inner = img
    ih, iw = inner.shape[:2]
    yy, xx = np.mgrid[0:ih, 0:iw]
    rr = np.sqrt(((yy - (ih - 1) / 2) / (ih / 2)) ** 2 + ((xx - (iw - 1) / 2) / (iw / 2)) ** 2)
    px = inner[rr > 1.06]
    if px.shape[0] < 50:
        px = inner.reshape(-1, 3)

    med = np.median(px, axis=0)
    diff = np.abs(med - np.array(PILL_BG_BGR, dtype=float))
    ok = bool(diff.max() <= 12)
    out = {
        "measured_bgr": [int(v) for v in med],
        "measured_rgb": [int(v) for v in med[::-1]],
        "fixed_bgr": list(PILL_BG_BGR),
        "max_channel_diff": float(diff.max()),
        "ok": ok,
        "n_pixels": int(px.shape[0]),
    }
    if verbose:
        print(f"  파일          {Path(image_path).name}")
        print(f"  측정 배경 RGB {tuple(out['measured_rgb'])}")
        print(f"  고정 배경 RGB {PILL_BG_RGB}  {PILL_BG_HEX}")
        print(f"  최대 채널 차이 {diff.max():.1f}  "
              + ("✅ 일치" if ok else "⚠️ 차이가 큽니다 — PILL_BG_* 를 재측정하세요"))
    return out


def selftest(verbose: bool = True) -> bool:
    """더미 데이터로 전체 파이프라인을 한 번 돌려 봅니다.

    노트북에서:
        import pill_transforms as pt
        pt.selftest()

    실제 데이터를 건드리지 않고 임시 폴더에서만 동작하며,
    끝나면 임시 폴더를 지웁니다. 오류 없이 True 가 나오면
    이 환경에서 모듈이 정상 동작한다는 뜻입니다.
    """
    import tempfile

    global CROPPED_PILLS_DIR, CUTOUT_CHECK_DIR
    _keep = (CROPPED_PILLS_DIR, CUTOUT_CHECK_DIR)

    print("■ 셀프 테스트 (더미 데이터)")
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "processed"

    # 실제 경로를 건드리지 않도록 임시 크롭 폴더를 만들어 씁니다.
    _crops = tmp / "cropped_pills_review"
    for j, nm in enumerate(["A", "B", "C"]):
        d = _crops / nm
        d.mkdir(parents=True, exist_ok=True)
        for k in range(3):
            c_ = np.full((120, 120, 3), (190, 205, 215), np.uint8)
            cv2.ellipse(c_, (60, 60), (38, 28), 30 * k, 0, 360,
                        (60 + 40 * j, 90, 200 - 30 * j), -1)
            imwrite_unicode(d / f"crop_{k}.png", c_)
    CROPPED_PILLS_DIR = str(_crops)
    CUTOUT_CHECK_DIR = str(tmp / "cutcheck")

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

    # ---- ★ 고정 배경색 검증 ----
    canvas = CopyPasteAugmentor.fixed_canvas(160, 160, random.Random(0))
    med = np.median(canvas.reshape(-1, 3), axis=0)
    dif = float(np.abs(med - np.array(PILL_BG_BGR, float)).max())
    print(f"\n고정 배경색 {PILL_BG_HEX} RGB{PILL_BG_RGB} — "
          f"생성 캔버스 median BGR {tuple(int(v) for v in med)} (차이 {dif:.1f})")
    assert dif <= 3.0, f"고정 배경색이 상수와 다릅니다: {med}"

    # ---- ★ letterbox 좌표 복원 검증 (추론에서 제출 좌표를 만드는 경로) ----
    meta = letterbox_meta(1280, 976, image_size=640)
    sc, pl, pt_ = meta["scale"], meta["pad_left"], meta["pad_top"]
    orig = np.array([[100.0, 150.0, 300.0, 400.0]])
    boxed = orig * sc + np.array([pl, pt_, pl, pt_], float)
    back = undo_letterbox_boxes(boxed, meta)
    err = float(np.abs(back - orig).max())
    print(f"letterbox 복원 오차 {err:.4f}px  (scale {sc:.4f}, pad {pl}/{pt_})")
    assert err < 1e-3, f"letterbox 복원이 틀렸습니다: {back}"

    # ---- ★ preflight 가 잘못된 라벨을 잡아내는지 ----
    bad = src / "labels" / "train" / "train_0.txt"
    keep = bad.read_text(encoding="utf-8")
    bad.write_text("0 1.7 0.5 0.2 0.2", encoding="utf-8")
    rep_bad = preflight(src, dst_root=tmp / "never", verbose=False)
    bad.write_text(keep, encoding="utf-8")
    print(f"preflight 오류 감지 테스트 — ok={rep_bad['ok']} "
          f"(bbox 오류 {rep_bad['bbox_errors']}건)")
    assert rep_bad["ok"] is False and rep_bad["bbox_errors"] >= 1

    if HAS_ALBUMENTATIONS:
        tf = get_train_transform(image_size=320, to_tensor=False)
        r = tf(image=np.random.randint(0, 255, (1280, 976, 3), np.uint8),
               bboxes=[[100, 150, 300, 400]], labels=[1])
        print(f"\nAlbumentations 파이프라인 OK — image {r['image'].shape}, "
              f"bboxes {len(r['bboxes'])}  |  패딩 RGB {pad_fill_value()}")
    else:
        print("\n(albumentations 미설치 — 온라인 transform 테스트는 건너뜀)")

    n_cut = len(list(Path(CUTOUT_CHECK_DIR).glob("*/cut/*.png")))
    print(f"\n컷아웃 검수 산출물 {n_cut}장 (임시 폴더)")

    shutil.rmtree(tmp, ignore_errors=True)
    CROPPED_PILLS_DIR, CUTOUT_CHECK_DIR = _keep
    print("\n✅ 셀프 테스트 통과 — 이 환경에서 pill_transforms 가 정상 동작합니다.")
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  CLI — 팀 공유 가이드와 동일하게 "먼저 --preflight, 성공하면 --execute"
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pill_transforms.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="알약 검출 전처리 / 증강 모듈 (사전검증 → 실행)",
        epilog=(
            "예시\n"
            "  1) 사전검증 (파일을 만들지 않습니다)\n"
            "     python pill_transforms.py --preflight\n"
            "  2) 실제 생성\n"
            "     python pill_transforms.py --execute --geom-mult 3 --n-synth 600\n"
            "  3) 고정 배경색 확인\n"
            "     python pill_transforms.py --bg-check /path/to/crop.png\n"
            "  4) 환경 점검\n"
            "     python pill_transforms.py --selftest\n"
        ),
    )
    ap.add_argument("--project-root", help="프로젝트 루트 (미지정 시 자동 감지)")
    ap.add_argument("--src-root", help="원본 YOLO 데이터셋 (기본 data/processed)")
    ap.add_argument("--dst-root", help="증강 결과 폴더 (기본 data/processed_aug)")
    ap.add_argument("--crops-dir", help="Copy&Paste 재료 폴더 (cropped_pills_review)")
    ap.add_argument("--cutout-check-dir", help="컷아웃 검수 저장 폴더")
    ap.add_argument("--geom-mult", type=int, default=None,
                    help=f"train 1장당 최종 장수 (기본 {DEFAULT_GEOM_MULT})")
    ap.add_argument("--n-synth", type=int, default=None,
                    help=f"Copy&Paste 합성 장수 (기본 {DEFAULT_N_SYNTH})")
    ap.add_argument("--cp-bg-mode", choices=["fixed", "crops", "random"], default=None,
                    help="합성 배경 방식 (기본 fixed = 실측 고정색)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--preflight", action="store_true", help="사전검증만 (기본 동작)")
    ap.add_argument("--execute", action="store_true", help="사전검증 후 실제 생성")
    ap.add_argument("--force", action="store_true", help="사전검증 실패해도 강행")
    ap.add_argument("--preview", action="store_true", help="생성 후 검수 이미지 저장")
    ap.add_argument("--bg-check", metavar="IMAGE", help="crop 이미지로 고정 배경색 검증")
    ap.add_argument("--selftest", action="store_true", help="더미 데이터로 환경 점검")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    global CP_BG_MODE

    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    if args.bg_check:
        return 0 if bg_check(args.bg_check)["ok"] else 1

    if args.selftest:
        return 0 if selftest() else 1

    if args.cp_bg_mode:
        CP_BG_MODE = args.cp_bg_mode

    configure(
        project_root=args.project_root,
        dataset_dir=args.src_root,
        aug_dataset_dir=args.dst_root,
        crops_dir=args.crops_dir,
        cutout_check_dir=args.cutout_check_dir,
        verbose=True,
    )

    rep = preflight(geom_mult=args.geom_mult, n_synth=args.n_synth, verbose=True)

    if not args.execute:
        print("\n※ 검증 전용 모드입니다. 실제로 만들려면 --execute 를 붙이세요.")
        return 0 if rep["ok"] else 1

    if not rep["ok"] and not args.force:
        return 1

    yaml_path = build_augmented_yolo_dataset(
        geom_mult=args.geom_mult, n_synth=args.n_synth,
        cutout_check_dir=args.cutout_check_dir,
        seed=args.seed, force=True, verbose=True,
    )
    if args.preview:
        preview_augmented(Path(yaml_path).parent, per_kind=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())