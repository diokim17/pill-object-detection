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

★ 학습 하이퍼파라미터도 이 파일이 단일 출처입니다
---------------------------------------------------
    import pill_transforms as pt

    model.train(
        data=DATA_YAML,
        imgsz=pt.DEFAULT_IMGSZ,        # ★ 1024 (Colab 실험 기준)
        batch=pt.DEFAULT_BATCH,        # 16
        **pt.get_online_aug("off"),    # ★ hsv_s=0.3 만 켜고 나머지 전부 0
    )

- `DEFAULT_IMGSZ = 1024`
- `ONLINE_AUG_PRESETS["off"]` → **hsv_s=0.3 만 켜짐**,
  scale · translate · shear · perspective · fliplr · flipud ·
  mosaic · mixup · copy_paste · erasing · degrees · hsv_h · hsv_v 는 전부 0.
  오프라인에서 이미 증강을 끝냈으므로 내장 증강을 겹치면 이중 증강이 됩니다.

기본값 (요청 반영)
------------------
- `DEFAULT_GEOM_MULT = 3`   → train 이미지 1장당 최종 3장 (원본 1 + 증강 2)
- `DEFAULT_N_SYNTH   = 600` → Copy & Paste 합성 이미지 600장
- `CLAHE_CLIP = 5.0`        → ★ 3.0 에서 5.0 으로 상향
- `CROPPED_PILLS_DIR`       → D:/PillData/pilldata/cropped_pills_review
- `CUTOUT_CHECK_DIR`        → D:/PillData/cutcheck (컷아웃 검수 산출물)
- `CP_BG_FROM_CROPS = True` → ★ 인공 배경은 크롭 이미지 배경의 **색조(H,S)만** 사용
- **CLAHE 는 확률이 아니라 전 이미지에 항상 적용됩니다.**
  (train / 평가 / 추론까지 동일 — 전처리를 학습에만 걸면 분포가 어긋납니다)
- ★ **train / val / test 분할을 쓰지 않습니다.** 원본 이미지 전부(224장)를 train 으로
  쓰고, val 폴더가 비면 data.yaml 의 `val:` 이 `images/train` 을 가리킵니다.
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
import sys
import time
from collections import Counter, defaultdict
from itertools import zip_longest
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
DEFAULT_CP_WEIGHTED = True   # True = 희소 클래스를 더 자주 붙임 (역빈도 가중)
# 200장 단위로 조절해주세요. (200->400->600)

# ═══════════════════════════════════════════════════════════════════════════
#  ★ 학습 하이퍼파라미터 — 02(스모크) · 03(본학습) · 04 · 05 가 전부 여기를 읽습니다
# ═══════════════════════════════════════════════════════════════════════════
#
#  ★ IMGSZ  — Colab 실험 기준값
#     각인(RE20, NR800 …)이 클래스 정보인 데이터라 해상도가 성능에 직결됩니다.
#     640 에서는 print_sensitive 클래스의 박스 짧은 변이 60px 미만으로 떨어져
#     각인이 안 읽힙니다. 1024 가 Colab T4/L4 에서 batch 16 으로 도는 상한선입니다.
#
#     ⚠️ VRAM 이 부족하면 IMGSZ 를 내리지 말고 **BATCH 를 먼저** 내리세요.
#        (해상도를 내리면 각인이 사라져 오분류가 늘어납니다)
DEFAULT_IMGSZ = 1024
DEFAULT_BATCH = 16

# ---------- ★ 학습 예산 (02 스모크 · 03 본학습이 여기만 읽습니다) ----------
#  ⚠️ 아래 값들은 **잠겨 있습니다**. 노트북에서 `pt.DEFAULT_EPOCHS = 50` 처럼
#     덮어쓰면 AttributeError 가 납니다. 바꾸려면 이 파일을 직접 고치세요.
#     (팀원마다 다른 epochs 로 돌려서 실험 비교가 무의미해지는 것을 막습니다)
DEFAULT_MODEL = "yolo11s.pt"      # 시작 가중치
DEFAULT_EPOCHS = 100              # ★ 본 학습(03) 상한
DEFAULT_PATIENCE = 15             # ★ 조기 종료 (val=train 이라 거의 안 걸립니다)
DEFAULT_SMOKE_EPOCHS = 10         # ★ 02 스모크 테스트 전용
DEFAULT_SMOKE_PATIENCE = 50       # 스모크는 조기 종료를 사실상 끕니다
DEFAULT_WORKERS = 0               # Windows 는 0 이 안전
DEFAULT_AMP = True
DEFAULT_DETERMINISTIC = True

# ---------- ★ 추론/평가 기본값 (04 · 05 가 읽습니다) ----------
DEFAULT_CONF = 0.001              # mAP 계산용이라 매우 낮게
DEFAULT_IOU_NMS = 0.7
DEFAULT_MAX_DET = 100
DEFAULT_EVAL_CONF = 0.25          # 오류 분해 · 시각화용 임계값

# ---------- ★ 온라인 증강 (Ultralytics 내장) ----------
#
#  이 프로젝트는 **오프라인에서 이미 증강을 끝냅니다**
#  (pill_transforms 가 회전 ±180° · HSV · 톤커브 · 노이즈 · Copy&Paste 를 디스크에 씀).
#  여기서 YOLO 내장 증강까지 세게 걸면 **이중 증강**이 되어
#  color1(색상)·각인이 원본에서 크게 벗어납니다.
#
#  그래서 기본 프리셋 "off" 는 **hsv_s 만 남기고 전부 0** 입니다.
#
#  | 인자          | 값  | 이유                                                |
#  |---------------|-----|-----------------------------------------------------|
#  | hsv_s         | 0.3 | ★ 조명·채도 변화 대응. 유일하게 켜 두는 항목        |
#  | hsv_h         | 0.0 | 색상(color1)이 클래스 정보 — 건드리면 안 됨         |
#  | hsv_v         | 0.0 | 오프라인에서 이미 밝기·감마 적용                    |
#  | degrees       | 0.0 | 오프라인에서 ±180° 적용 완료                        |
#  | scale         | 0.0 | ★ 무효화. 오프라인 SCALE_RANGE 로 이미 처리         |
#  | translate     | 0.0 | ★ 무효화                                            |
#  | shear         | 0.0 | ★ 무효화                                            |
#  | perspective   | 0.0 | ★ 무효화. 알약이 찌그러지면 모양(shape)이 왜곡됨    |
#  | fliplr/flipud | 0.0 | ★ 무효화. 각인 글자가 뒤집히면 클래스 단서가 파괴됨 |
#  | mosaic        | 0.0 | ★ 무효화                                            |
#  | mixup         | 0.0 | ★ 무효화. 알약이 반투명하게 겹치면 각인이 사라짐    |
#  | copy_paste    | 0.0 | ★ 무효화. 오프라인 Copy&Paste 와 중복               |
#  | erasing       | 0.0 | ★ 무효화. 각인이 통째로 지워질 수 있음              |
#
#  ⚠️ hsv_s=0.3 은 채도를 ±30% 흔듭니다. color1(색상)이 클래스 정보인 데이터라
#     "연한 노랑 ↔ 흰색" 처럼 인접한 색끼리 헷갈릴 여지가 생깁니다.
#     04 의 color_sensitive 그룹 AP 가 떨어지면 0.2 → 0.15 로 낮춰 보세요.

#  ⚠️⚠️ 위 표와 실제 "off" 값이 다릅니다 — 확인하고 하나로 맞추세요 ⚠️⚠️
#     표 설명:            hsv_h 0, translate 0, shear 0
#     실제 "off" 값:      hsv_h 0.015, translate 0.1, shear 2.0  ← 켜져 있습니다
#     의도대로 "hsv_s 만 켜기" 를 원하면 ONLINE_AUG = "strict_off" 를 쓰세요.
#     (기존 실험과의 연속성 때문에 "off" 의 숫자는 그대로 두었습니다)

ONLINE_AUG_PRESETS: Dict[str, Dict[str, float]] = {
    # ★ 기존 기본값 — 02·03 이 지금까지 실제로 써 온 값
    "off": dict(
        hsv_h=0.015, hsv_s=0.3, hsv_v=0.0,          # ★ hsv_s 만 켬
        degrees=0.0, translate=0.1, scale=0.0, shear=2.0, perspective=0.0,
        fliplr=0.0, flipud=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
    ),
    # ★ 문서(표)와 정확히 같은 값 — hsv_s 만 켜고 나머지 전부 0
    "strict_off": dict(
        hsv_h=0.0, hsv_s=0.3, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        fliplr=0.0, flipud=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
    ),
    # 증강량이 부족해 보일 때 — 소각도 회전만 추가
    "light": dict(
        hsv_h=0.015, hsv_s=0.3, hsv_v=0.0,
        degrees=15.0, translate=0.1, scale=0.1, shear=0.0, perspective=0.0,
        fliplr=0.0, flipud=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
    ),
    # 증강 없는 data/pill_raw 로 학습할 때만 (증강 효과 정량화 실험)
    "full": dict(
        hsv_h=0.015, hsv_s=0.3, hsv_v=0.0,
        degrees=180.0, translate=0.1, scale=0.1, shear=0.0, perspective=0.0,
        fliplr=0.0, flipud=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
    ),
}
DEFAULT_ONLINE_AUG = "off"


def get_online_aug(preset: Optional[str] = None, **overrides) -> Dict[str, float]:
    """Ultralytics `model.train()` 에 그대로 넘길 증강 인자 dict 를 돌려줍니다.

        from pill_transforms import get_online_aug
        model.train(data=..., imgsz=pt.DEFAULT_IMGSZ, **pt.get_online_aug())

        # 일부만 바꾸고 싶을 때
        pt.get_online_aug("off", hsv_s=0.15)

    ★ fliplr / flipud 는 어느 프리셋에서도 0 입니다.
      각인(RE20, NR800)이 뒤집히면 클래스 단서 자체가 파괴됩니다.
    """
    if preset is None:
        preset = DEFAULT_ONLINE_AUG
    if preset not in ONLINE_AUG_PRESETS:
        raise ValueError(
            f"모르는 프리셋입니다: {preset!r}  "
            f"(가능: {list(ONLINE_AUG_PRESETS)})"
        )
    aug = dict(ONLINE_AUG_PRESETS[preset])
    unknown = set(overrides) - set(aug)
    if unknown:
        raise ValueError(f"증강 인자가 아닙니다: {sorted(unknown)}")
    aug.update(overrides)
    return aug


def describe_online_aug(preset: Optional[str] = None) -> str:
    """프리셋을 사람이 읽는 표로. 노트북 설정 셀에서 출력용."""
    aug = get_online_aug(preset)
    on = {k: v for k, v in aug.items() if v}
    off = sorted(k for k, v in aug.items() if not v)
    lines = [f"온라인 증강 프리셋 '{preset or DEFAULT_ONLINE_AUG}'"]
    lines.append(f"  켜짐  {on if on else '없음'}")
    lines.append(f"  무효화(0)  {', '.join(off)}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  ★ 학습 인자 단일 출처 — 02(스모크) · 03(본학습)이 이 함수만 부릅니다
# ═══════════════════════════════════════════════════════════════════════════

def get_train_kwargs(
    preset: Optional[str] = None,
    *,
    smoke: bool = False,
    rect: bool = False,
    resume: bool = False,
) -> Dict[str, Any]:
    """`model.train(data=..., project=..., name=..., **pt.get_train_kwargs())`

    epochs · patience · imgsz · batch · workers · 온라인 증강까지 **전부** 이
    함수가 채웁니다. 노트북에서 숫자를 직접 쓰지 마세요.

        smoke=True   → 02 의 스모크 테스트 예산 (DEFAULT_SMOKE_EPOCHS)
        smoke=False  → 03 의 본 학습 예산 (DEFAULT_EPOCHS)

    ⚠️ Ultralytics 는 안 넘긴 증강 인자에 자체 기본값(translate 0.1,
       erasing 0.4, mosaic 1.0)을 씁니다. 그래서 항상 전부 넘깁니다.
    """
    aug = get_online_aug(preset)
    if rect:
        # Ultralytics 는 rect=True 면 mosaic 을 못 씁니다
        aug = dict(aug, mosaic=0.0)

    kw: Dict[str, Any] = dict(
        epochs=DEFAULT_SMOKE_EPOCHS if smoke else DEFAULT_EPOCHS,
        patience=DEFAULT_SMOKE_PATIENCE if smoke else DEFAULT_PATIENCE,
        imgsz=DEFAULT_IMGSZ,
        batch=DEFAULT_BATCH,
        workers=DEFAULT_WORKERS,
        amp=DEFAULT_AMP,
        seed=SEED,
        deterministic=DEFAULT_DETERMINISTIC,
        rect=bool(rect),
        resume=bool(resume),
    )
    kw.update(aug)
    return kw


def describe_hparams(preset: Optional[str] = None) -> str:
    """지금 잠겨 있는 학습 하이퍼파라미터를 사람이 읽는 표로."""
    lines = ["■ 학습 하이퍼파라미터 (pill_transforms.py 단일 출처 · 잠김)",
             f"   모델        {DEFAULT_MODEL}",
             f"   imgsz       {DEFAULT_IMGSZ}      batch  {DEFAULT_BATCH}",
             f"   epochs      본학습 {DEFAULT_EPOCHS} / 스모크 {DEFAULT_SMOKE_EPOCHS}",
             f"   patience    본학습 {DEFAULT_PATIENCE} / 스모크 {DEFAULT_SMOKE_PATIENCE}",
             f"   workers     {DEFAULT_WORKERS}      seed   {SEED}",
             f"   오프라인    기하 ×{DEFAULT_GEOM_MULT} / Copy&Paste {DEFAULT_N_SYNTH}장"
             f" / another 얹기 {ANOTHER_ONTO_N}장"
             f"  (★ CP+another얹기 합이 기하증강 증가분과 1:1 목표"
             f" — 실제 비율은 02 의 4-검산 셀에서 확인)",
             f"   전처리      WB={USE_WHITE_BALANCE} / CLAHE={USE_CLAHE}"
             f" (clip {CLAHE_CLIP}, grid {CLAHE_GRID})",
             f"   추론        conf {DEFAULT_CONF} / iou {DEFAULT_IOU_NMS}"
             f" / max_det {DEFAULT_MAX_DET}",
             "",
             describe_online_aug(preset)]
    return "\n".join(lines)


# ---------- 전처리 (학습·평가·추론 전부 동일하게 적용) ----------
USE_WHITE_BALANCE = True   # Shades-of-Gray 화이트밸런스
USE_CLAHE = True           # ★ Lab 의 L 채널 CLAHE — 항상 켬(배경, 객체 구분)
CLAHE_CLIP = 5.0           # ★ 3.0 → 5.0 (대비를 더 세게. 노이즈가 뜨면 4.0 으로)
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
PILLS_RANGE = (3, 4)       # 합성 이미지 1장에 붙일 알약 수 (원본 분포와 동일)
CP_EXTRA_RANGE = (1, 1)    # onto_train 모드에서 추가로 붙일 알약 수
CP_MODE = "mix"            # "synth" | "onto_train" | "mix"
CP_SYNTH_RATIO = 0.5       # mix 일 때 인공 배경 비율
CP_OVERLAP = 0.10          # 허용 겹침 비율
CP_SCALE_JIT = (0.95, 1.05)
CP_FEATHER = 2             # 경계 페더링(px)
SHADOW_ALPHA = (0.30, 0.40)
SHADOW_BLUR = (3, 12)
MAX_CROPS_PER_CLASS = 40   # 크롭 라이브러리 클래스당 최대 장수

# ---------- ★ Copy&Paste 인공 배경 — 크롭 이미지 배경의 "색조만" 사용 ----------
#  crop 이미지의 알약 바깥(배경) 픽셀에서 HSV 의 H(색상)·S(채도)만 가져오고,
#  V(밝기)는 가져오지 않고 아래 범위에서 새로 뽑습니다.
#  → 원본 촬영 배경의 색감은 유지하되 밝기 분포는 다양해집니다.
CP_BG_FROM_CROPS = True       # False 면 예전처럼 완전 랜덤 단색 배경
CP_BG_V_RANGE = (180.0, 240.0)   # 새로 뽑는 밝기(V) 범위 0~255
CP_BG_S_SCALE = (0.95, 1.05)     # 가져온 채도에 곱할 지터
CP_BG_HUE_JITTER = 2.0           # 색상(H, 0~179) 지터 — 작게 유지
CP_BG_GRAD = 14.0                # 완만한 밝기 기울기 진폭
CP_BG_NOISE = 2.0                # 가우시안 노이즈 표준편차

# ═══════════════════════════════════════════════════════════════════════════
#  ★★★ 합성 배경 — train_images 와 "같은 색으로 고정" ★★★
# ═══════════════════════════════════════════════════════════════════════════
#  이전 동작의 문제
#    · 배경 색조(H,S)를 **크롭마다 다르게** 가져오고 밝기(V)는 CP_BG_V_RANGE 에서
#      새로 뽑았습니다. train_images 실측이 V≈131 인데 범위가 (150~240)이라
#      합성본이 **항상 원본보다 밝고 진한 보라색**으로 나왔습니다.
#  지금 동작
#    · BG_MODE="fixed"  → 아래 한 개의 HSV 값으로 **모든 합성 배경을 고정**합니다.
#      (train_images 를 실측해서 채우려면 measure_train_bg_hsv() 를 쓰세요)
BG_MODE = "fixed"                 # "fixed"(권장) | "from_crop"(옛 동작) | "random"
BG_HSV: Optional[Tuple[float, float, float]] = None   # None → BG_HSV_DEFAULT
BG_HSV_DEFAULT = (117.0, 62.0, 131.0)   # ★ train_images 실측 중앙값 (OpenCV HSV)
#  ★ 배경색은 **모든 합성본에서 완전히 같은 한 색**입니다.
#    (0, 0, 0) 이면 장마다 색이 전혀 흔들리지 않습니다. 촬영 질감만 남기려고
#    아주 약한 밝기 기울기(BG_GRAD)와 노이즈(BG_NOISE)는 유지합니다.
BG_JITTER = (0.0, 0.0, 0.0)       # (H, S, V) 흔들림 — 0 = 색 완전 통일
#  ★ True 면 BG_JITTER 값과 상관없이 **무조건** 흔들림을 0 으로 강제합니다.
#    (노트북이 옛 BG_JITTER 를 주입해도 배경색이 갈라지지 않게 하는 안전장치)
BG_STRICT_UNIFORM = True
BG_GRAD = 5.0                     # 완만한 밝기 기울기 진폭 (색조는 그대로)
BG_NOISE = 2.0                    # 가우시안 노이즈 표준편차

# ---------- ★ 한 이미지 안의 클래스 규칙 ----------
UNIQUE_CLASS_PER_IMAGE = True     # ★ 한 장에 같은 종류의 알약을 두 번 넣지 않습니다

# ═══════════════════════════════════════════════════════════════════════════
#  ★★★ 균일 배치 — 원본 사진처럼 알약을 화면에 고르게 흩뿌립니다 ★★★
# ═══════════════════════════════════════════════════════════════════════════
#  원본 사진(train_images)은 알약이 한쪽에 몰리지 않고 2x2 / 삼각형 구도로
#  넓게 퍼져 있습니다. 무작위 좌표를 뽑으면 뭉치거나 한쪽으로 쏠립니다.
#  → 캔버스를 격자로 나눠 **서로 다른 칸에 하나씩** 놓고, 칸 안에서만 흔듭니다.
UNIFORM_LAYOUT = True             # False 면 옛 방식(완전 랜덤 좌표)
UNIFORM_SLOT_JITTER = 0.34        # 칸 크기 대비 흔들림 (0 = 칸 정중앙, 0.5 = 칸 가장자리까지)
UNIFORM_EDGE_PAD = 0.05           # 화면 가장자리 여백 (짧은 변 대비)
UNIFORM_SLOT_TRIES = 24           # 칸 안에서 자리 찾기 재시도 횟수

# ═══════════════════════════════════════════════════════════════════════════
#  ★★★ train_images(알약 3개) + train_another 알약 1개 = 4개 합성 ★★★
# ═══════════════════════════════════════════════════════════════════════════
#  원본 사진 위에 train_another 의 알약을 하나 더 얹어 "알약 4개" 이미지를 만듭니다.
#  배경·조명·그림자가 진짜 사진 그대로라서 캔버스 합성보다 도메인 갭이 작습니다.
#
#  ★ 200 이라는 숫자의 근거 — "합성 비율 ≈ 기하 증강 비율" 1:1 목표
#     기하 증강이 늘리는 장수 = (train_images + train_another) × (GEOM_MULT - 1)
#     합성이 늘리는 장수     = N_SYNTH(Copy&Paste) + ANOTHER_ONTO_N(이 값)
#     GEOM_MULT=3, N_SYNTH=600 기준으로 "기하 증강 늘어난 분" 과
#     "Copy&Paste + another얹기" 를 비슷한 크기로 맞추려고 200 을 더했습니다.
#     (원본 200여 장 규모를 가정한 값입니다 — 데이터가 크게 달라지면
#      02 의 검산 셀이 실제 비율을 계산해 보여 주니 거기서 다시 맞추세요)
ANOTHER_ONTO_N = 200              # ★ 만들 장수 (0 = 끔)
ANOTHER_ONTO_BASE_PILLS = 3       # 재료로 쓸 원본 이미지의 알약 수
ANOTHER_ONTO_ADD = 1              # 얹을 알약 수 → 최종 3+1 = 4개
ANOTHER_ONTO_MAX_PER_CLASS = 0    # 0 = 제한 없음 (train_another 크롭 사용 상한)
ANOTHER_ONTO_SHADOW = True        # 얹은 알약 밑에 그림자
ANOTHER_ONTO_SIZE_CLAMP = (0.5, 2.0)  # 원본 알약 중앙값 대비 허용 크기. 벗어나면 리사이즈
ANOTHER_ONTO_ROTATE = True        # 얹기 전에 임의 회전 (각인 방향 다양화)


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

#  ★ 기본값을 팀 공용 경로로 지정했습니다 (노트북에서 덮어쓸 수 있습니다).
#  ★ None 으로 두면 import 할 때 자동 탐색합니다 (Colab · Windows 공용).
#    직접 지정하려면 pt.setup(pill_root=...) 를 쓰거나 여기를 고치세요.
CROPPED_PILLS_DIR: Optional[str] = None

#  ★ AI Hub 추가 데이터 팀 작업 폴더 (공유 가이드 기준)
#     team_work/cropped_output 이 있으면 Copy&Paste 재료로 추가할 수 있습니다.
TEAM_WORK_DIR: Optional[str] = None

#  ★ 컷아웃 검수 결과를 모아 둘 폴더
CUTOUT_CHECK_DIR: Optional[str] = None

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


# ---------------------------------------------------------------------------
#  ★ 고정 배경 — train_images 와 같은 색으로 합성 캔버스를 만듭니다
# ---------------------------------------------------------------------------

def measure_train_bg_hsv(
    images_dir: PathLike,
    labels_dir: Optional[PathLike] = None,
    n: int = 60,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """★ train 원본 이미지의 **배경 색(HSV 중앙값)** 을 실측합니다.

    YOLO 라벨이 있으면 박스 안(알약)을 빼고 배경 픽셀만 봅니다.
    결과를 `BG_HSV` 에 넣으면 모든 합성 배경이 원본과 같은 색이 됩니다.

        pt.BG_HSV = pt.measure_train_bg_hsv(f"{RAW_DIR}/images/train",
                                            f"{RAW_DIR}/labels/train")
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir) if labels_dir else None
    paths = sorted(p for p in images_dir.glob("*")
                   if p.suffix.lower() in CROPPED_PILLS_EXTS)
    if not paths:
        if verbose:
            print(f"⚠️ {images_dir} 에 이미지가 없어 기본값을 씁니다 {BG_HSV_DEFAULT}")
        return tuple(float(v) for v in BG_HSV_DEFAULT)  # type: ignore[return-value]

    rng = random.Random(SEED)
    if len(paths) > n:
        paths = rng.sample(paths, n)

    hs, ss, vs = [], [], []
    used = 0
    for p in paths:
        img = imread_unicode(p)
        if img is None:
            continue
        H, W = img.shape[:2]
        bg = np.ones((H, W), bool)
        if labels_dir is not None:
            lp = labels_dir / f"{p.stem}.txt"
            if lp.exists():
                for b in _read_yolo_label(lp, W, H):
                    x, y, w, h = b[:4]
                    # 박스 + 그림자 여유까지 넉넉히 제외
                    mx, my = 0.35 * w, 0.35 * h
                    x1 = int(max(0, x - mx)); y1 = int(max(0, y - my))
                    x2 = int(min(W, x + w + mx)); y2 = int(min(H, y + h + my))
                    bg[y1:y2, x1:x2] = False
        if int(bg.sum()) < 1000:
            continue
        hsv = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2HSV)
        hh = hsv[..., 0][bg].astype(np.float32)
        ang = hh * (np.pi / 90.0)
        hbar = math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean()))
        if hbar < 0:
            hbar += 2 * math.pi
        hs.append(hbar * 90.0 / np.pi)
        ss.append(float(np.median(hsv[..., 1][bg])))
        vs.append(float(np.median(hsv[..., 2][bg])))
        used += 1

    if not hs:
        if verbose:
            print(f"⚠️ 배경 픽셀을 못 찾아 기본값을 씁니다 {BG_HSV_DEFAULT}")
        return tuple(float(v) for v in BG_HSV_DEFAULT)  # type: ignore[return-value]

    out = (float(np.median(hs)), float(np.median(ss)), float(np.median(vs)))
    if verbose:
        print(f"★ train 배경 실측 {used}장 → HSV(H={out[0]:.1f}, S={out[1]:.1f}, "
              f"V={out[2]:.1f})  ※ 이 색으로 합성 배경을 고정합니다")
    return out


def resolve_bg_hsv() -> Tuple[float, float, float]:
    """합성 배경으로 쓸 고정 HSV 를 돌려줍니다 (BG_HSV → 없으면 기본값)."""
    v = BG_HSV if BG_HSV else BG_HSV_DEFAULT
    return (float(v[0]), float(v[1]), float(v[2]))


def make_bg_canvas(
    hw: Tuple[int, int],
    rng: random.Random,
    hsv: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    """★ 고정 색 배경 캔버스 — 합성 이미지(캔버스/Copy&Paste)의 **단일 출처**.

    장마다 색이 달라지지 않도록 지터는 BG_JITTER 만큼만 줍니다
    (기본 H±1, S±4, V±6 — 눈으로는 같은 색으로 보입니다).
    """
    H, W = int(hw[0]), int(hw[1])
    h0, s0, v0 = hsv if hsv else resolve_bg_hsv()
    jh, js, jv = (0.0, 0.0, 0.0) if BG_STRICT_UNIFORM else BG_JITTER
    hue = (h0 + rng.uniform(-jh, jh)) % 180.0
    sat = float(np.clip(s0 + rng.uniform(-js, js), 0, 255))
    val = float(np.clip(v0 + rng.uniform(-jv, jv), 0, 255))

    gy = np.linspace(rng.uniform(-BG_GRAD, BG_GRAD),
                     rng.uniform(-BG_GRAD, BG_GRAD), H)[:, None]
    gx = np.linspace(rng.uniform(-BG_GRAD, BG_GRAD),
                     rng.uniform(-BG_GRAD, BG_GRAD), W)[None, :]

    hsv_img = np.empty((H, W, 3), np.float32)
    hsv_img[..., 0] = hue
    hsv_img[..., 1] = sat
    hsv_img[..., 2] = np.clip(val + gy + gx
                              + np.random.normal(0, BG_NOISE, (H, W)), 0, 255)
    out = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    out += np.random.normal(0, BG_NOISE, (H, W, 3))
    return np.clip(out, 0, 255).astype(np.uint8)


def plan_uniform_slots(
    hw: Tuple[int, int],
    n: int,
    rng: random.Random,
    pad: Optional[int] = None,
    jitter: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """★ 알약 n 개를 화면에 **고르게** 놓을 중심 좌표 n 개를 만듭니다.

    캔버스를 rows x cols 격자로 나누되, 칸이 **정사각형에 가깝도록** rows/cols 를
    고릅니다 (976x1280 에 4개 → 2x2, 3개 → 2x2 중 세 칸, 2개 → 2x1).
    서로 다른 칸을 하나씩 차지하므로 뭉치거나 한쪽으로 쏠리지 않습니다.

    Returns:
        [(cx, cy), ...] — 캔버스 좌표계의 중심점 n 개 (칸 순서는 섞여 있습니다).
    """
    rows, cols, cw, ch, pad = _grid_dims(hw, n, pad)
    if jitter is None:
        jitter = UNIFORM_SLOT_JITTER
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    return [_cell_center(r, c, cw, ch, pad, rng, jitter) for r, c in cells[:n]]


def _grid_dims(hw: Tuple[int, int], n: int, pad: Optional[int] = None):
    """n 개를 담을 격자 (rows, cols, 칸너비, 칸높이, 여백).

    ★ n<=4 (우리 파이프라인의 실제 범위 — PILLS_RANGE/ANOTHER_PILLS_PER_CANVAS
      가 전부 2~4) 이면 **항상 2x2 사분면**을 씁니다. 참고 사진(BSP/DGTH/Noltec/
      TJA ER)이 정확히 이 구도라서, 4개면 네 칸 전부, 3개면 세 칸(한쪽 모서리는
      비움), 2개면 두 칸을 씁니다 — 절대 한쪽에 몰리지 않습니다.

    n>4 인 경우에만(기본 설정에서는 일어나지 않음) 동적으로 격자를 고르되,
    ★ **버려지는 칸(waste)이 없는 격자를 최우선**으로 고릅니다. 예전 버전은
      정사각형에 가까움만 보다가 4개인데 3x2(칸 6개 중 2개는 안 씀) 를 골라
      두 칸이 캔버스 절반에 몰리는 문제가 있었습니다.
    """
    H, W = int(hw[0]), int(hw[1])
    n = max(1, int(n))
    if pad is None:
        pad = int(UNIFORM_EDGE_PAD * min(H, W))
    inner_h, inner_w = max(1, H - 2 * pad), max(1, W - 2 * pad)

    if n <= 4:
        rows, cols = 2, 2
    else:
        best = None
        for rows in range(1, n + 1):
            cols = int(math.ceil(n / rows))
            waste = rows * cols - n                 # ★ 버려지는 칸 — 최우선으로 줄입니다
            chh, cww = inner_h / rows, inner_w / cols
            squareness = abs(math.log(max(cww, 1e-6) / max(chh, 1e-6)))
            score = waste * 1000.0 + squareness      # waste 가 있으면 무조건 후순위
            if best is None or score < best[0]:
                best = (score, rows, cols)
        _, rows, cols = best  # type: ignore[misc]

    return rows, cols, inner_w / cols, inner_h / rows, pad


def _cell_center(r: int, c: int, cw: float, ch: float, pad: int,
                 rng: random.Random, jitter: float) -> Tuple[float, float]:
    return (float(pad + (c + 0.5) * cw + rng.uniform(-jitter, jitter) * cw),
            float(pad + (r + 0.5) * ch + rng.uniform(-jitter, jitter) * ch))


def free_uniform_slots(
    hw: Tuple[int, int],
    boxes: Sequence[Sequence[float]],
    rng: random.Random,
    n: int = 1,
    pad: Optional[int] = None,
) -> List[Tuple[float, float]]:
    """★ 이미 알약이 있는 사진에서 **비어 있는 칸**의 중심을 n 개 돌려줍니다.

    기존 알약이 차지한 칸을 뺀 나머지 중, 기존 알약에서 **가장 멀리 떨어진**
    칸부터 고릅니다. 원본 사진의 배치 간격을 그대로 흉내 냅니다.

    Args:
        boxes: [(cx, cy, w, h, cls), ...] — Sample.boxes 형식(중심 좌표).
    """
    total = len(boxes) + max(1, int(n))
    rows, cols, cw, ch, pad = _grid_dims(hw, total, pad)
    centers = [(float(b[0]), float(b[1])) for b in boxes]

    taken = set()
    for bx, by in centers:
        c = int(np.clip((bx - pad) // cw, 0, cols - 1))
        r = int(np.clip((by - pad) // ch, 0, rows - 1))
        taken.add((r, c))

    free = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in taken]
    if not free:
        free = [(r, c) for r in range(rows) for c in range(cols)]

    def _far(rc):
        r, c = rc
        x, y = pad + (c + 0.5) * cw, pad + (r + 0.5) * ch
        if not centers:
            return rng.random()
        return min(math.hypot(x - bx, y - by) for bx, by in centers)

    free.sort(key=_far, reverse=True)
    keep = free[:max(1, int(n) * 2)]
    rng.shuffle(keep)
    return [_cell_center(r, c, cw, ch, pad, rng, UNIFORM_SLOT_JITTER)
            for r, c in keep[:n]]


def jitter_slot(slot: Tuple[float, float], cell: Tuple[float, float],
                rng: random.Random, spread: float = 0.5) -> Tuple[float, float]:
    """슬롯 주변을 조금씩 흔들어 재시도할 때 씁니다."""
    cw, ch = cell
    return (slot[0] + rng.uniform(-spread, spread) * cw,
            slot[1] + rng.uniform(-spread, spread) * ch)


def cast_shadow_on(canvas: np.ndarray, mask: np.ndarray,
                   y: int, x: int, rng: random.Random) -> None:
    """알약 마스크를 눕혀 그림자로 합성합니다 (원본 사진처럼 보이게)."""
    ph, pw = mask.shape[:2]
    pad = max(8, int(max(ph, pw) * 0.5))
    big = np.zeros((ph + 2 * pad, pw + 2 * pad), np.float32)
    big[pad:pad + ph, pad:pad + pw] = (mask > 0).astype(np.float32)
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
    canvas[y0:y1, x0:x1] = np.clip(roi * (1.0 - alpha * sub[..., None]),
                                   0, 255).astype(np.uint8)


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
        crops_dir: Optional[Union[PathLike, Sequence[PathLike]]] = None,  # ★ 크롭 폴더(들)
        check_dir: Optional[PathLike] = None,   # ★ 컷아웃 검수 저장 폴더(cutcheck)
    ):
        self.cache_dir = Path(cache_dir)
        self.max_per_class = int(max_per_class)
        self.margin = float(margin)
        self.check_dir: Optional[Path] = Path(check_dir) if check_dir else None
        # ★ 인자 > 전역(CROPPED_PILLS_DIR + team_work/cropped_output) 순으로 결정
        #   여러 폴더를 리스트로 넘기면 전부 재료로 씁니다 (AI Hub 추가분 합치기).
        self.crops_dirs: List[Path] = resolve_crop_dirs(crops_dir)
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
        if self.crops_dirs:
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
        """크롭 폴더(들)의 클래스별 하위 폴더에서 크롭 이미지를 읽어 옵니다.

        ★ 여러 뿌리 폴더를 동시에 씁니다 (`self.crops_dirs`).
          예) cropped_pills_review + team_work/cropped_output(AI Hub 추가분)
          같은 클래스 폴더가 여러 뿌리에 있으면 **번갈아 가며** 뽑아
          한쪽 출처가 max_per_class 를 독점하지 않게 합니다.
        """
        roots = [r for r in self.crops_dirs if r.is_dir()]
        if not roots:
            raise FileNotFoundError(
                f"크롭 폴더를 찾을 수 없습니다: {[str(r) for r in self.crops_dirs]}\n"
                f"→ pill_transforms.CROPPED_PILLS_DIR / TEAM_WORK_DIR 경로를 확인하거나\n"
                f"   pt.setup() 을 먼저 호출하세요."
            )
        if not names:
            raise ValueError(
                "크롭 폴더를 쓰려면 클래스명(names)이 필요합니다. "
                "build(records, names=names) 로 전달하세요."
            )

        lut = _build_name_lookup(names)

        # ---- 여러 뿌리의 하위 폴더를 클래스별로 모읍니다 ----
        by_cid: Dict[int, List[Path]] = defaultdict(list)
        n_sub = 0
        for root in roots:
            for d in sorted(p for p in root.iterdir() if p.is_dir()):
                n_sub += 1
                cid = _match_class_id(d.name, lut)
                if cid is None:
                    self.unmatched_folders.append(f"{root.name}/{d.name}")
                    self.stats["폴더 매칭 실패"] += 1
                    continue
                by_cid[int(cid)].append(d)

        if verbose:
            for root in roots:
                print(f"    크롭 폴더 {root}")
            print(f"    하위 폴더 {n_sub}개 → 매칭된 클래스 {len(by_cid)}종")

        subdirs = sorted(by_cid)          # 진행률 표시에 사용 (클래스 목록)
        for n_done, cid in enumerate(subdirs, 1):
            dirs = by_cid[cid]
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

            # ★ 여러 출처의 파일을 번갈아 섞습니다 (한 출처가 독점하지 않도록)
            per_dir = [
                sorted(p for p in d.iterdir()
                       if p.is_file() and p.suffix.lower() in CROPPED_PILLS_EXTS)
                for d in dirs
            ]
            files = [p for grp in zip_longest(*per_dir) for p in grp if p is not None]
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
                print(f"    크롭 라이브러리 {n_done}/{len(subdirs)} 클래스")

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
    # ★ 여러 크롭 뿌리 폴더를 동시에 검수합니다
    #   (cropped_pills_review + team_work/cropped_output 등)
    crop_roots = resolve_crop_dirs(crops_dir)
    crop_roots = [r for r in crop_roots if r.is_dir()]
    out_dir = Path(out_dir or CUTOUT_CHECK_DIR or "")
    if not crop_roots:
        raise FileNotFoundError(
            f"크롭 폴더를 찾을 수 없습니다: {crops_dir or CROPPED_PILLS_DIR}\n"
            f"→ CROPPED_PILLS_DIR 경로를 확인하거나 pt.setup() 을 먼저 호출하세요."
        )
    if str(out_dir) in ("", "."):
        raise ValueError("out_dir(CUTOUT_CHECK_DIR)를 지정하세요.")

    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lut = _build_name_lookup(names) if names else {}
    subdirs = []
    for _root in crop_roots:
        _subs = sorted([d for d in _root.iterdir() if d.is_dir()])
        subdirs += _subs if _subs else [_root]        # 하위 폴더가 없으면 평면 구조

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

    def blank_canvas(self, H, W, rng):
        """★ 인공 배경 — 기본값은 **train_images 와 같은 고정 색**입니다.

        BG_MODE
          "fixed"     ★ 권장. `BG_HSV`(없으면 `BG_HSV_DEFAULT`) 한 색으로 고정.
                      원본 사진과 배경이 같아져 배경이 클래스 단서가 되지 않습니다.
          "from_crop" 옛 동작 — 크롭마다 색조를 가져오고 밝기를 새로 뽑습니다.
                      (합성본이 원본보다 밝고 진해지는 원인이었습니다)
          "random"    완전 랜덤 단색.
        """
        if BG_MODE == "fixed":
            self.stats["고정 배경"] += 1
            return make_bg_canvas((H, W), rng)
        if BG_MODE == "random" or not (CP_BG_FROM_CROPS and self.bg_tones):
            return self.random_canvas(H, W, rng)

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
    def paste_one(self, canvas, occ, rng, tries: int = 40, exclude=None,
                  strict: bool = True, slot=None):
        """알약 1개를 붙입니다.

        exclude: 이미 이 이미지에 들어간 class_id 집합.
                 `UNIQUE_CLASS_PER_IMAGE=True` 면 여기 있는 클래스는 뽑지 않습니다
                 (= 한 장에 같은 종류의 알약이 두 번 나오지 않습니다).
        strict:  쓸 수 있는 클래스가 하나도 안 남았을 때
                 True  → 붙이지 않고 포기 (합성 이미지용)
                 False → 중복을 허용하고 진행 (원본 위에 덧붙일 때. 클래스 수가
                         적은 소규모 데이터에서 이미지가 통째로 버려지지 않게 합니다)
        """
        pool, weights = self.pool, self.weights
        if UNIQUE_CLASS_PER_IMAGE and exclude:
            keep = [(c, w) for c, w in zip(self.pool, self.weights) if c not in exclude]
            if not keep:
                self.stats["남은 클래스 없음"] += 1
                if strict:
                    return None
                self.stats["중복 허용 fallback"] += 1
            else:
                pool = [c for c, _ in keep]
                weights = [w for _, w in keep]
        cid = rng.choices(pool, weights=weights, k=1)[0]
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

        pad = int(UNIFORM_EDGE_PAD * min(H, W))
        pad = min(pad, max(0, (H - ph) // 2), max(0, (W - pw) // 2))

        def _ok(yy: int, xx: int) -> bool:
            return int(np.count_nonzero(occ[yy:yy + ph, xx:xx + pw][reg])) <= self.overlap * area

        def _put(yy: int, xx: int):
            self.cast_shadow(canvas, mask, yy, xx, rng)   # ★ 그림자 먼저
            self.blend(canvas, patch, mask, yy, xx)       # ★ 알약을 그 위에
            occ[yy:yy + ph, xx:xx + pw][reg] = 1
            return (xx + pw / 2.0, yy + ph / 2.0, float(pw), float(ph), int(cid))

        # ★ 슬롯(격자 칸)을 받았으면 그 중심에 놓습니다 → 화면에 고르게 퍼집니다
        if slot is not None and UNIFORM_LAYOUT:
            cell = (W / 2.0, H / 2.0)
            cx, cy = slot
            for _t in range(UNIFORM_SLOT_TRIES + 1):
                xx = int(np.clip(round(cx - pw / 2.0), pad, W - pw - pad))
                yy = int(np.clip(round(cy - ph / 2.0), pad, H - ph - pad))
                if _ok(yy, xx):
                    return _put(yy, xx)
                cx, cy = jitter_slot(slot, cell, rng, 0.18)

        for _ in range(tries):
            yy = rng.randint(pad, max(pad, H - ph - pad))
            xx = rng.randint(pad, max(pad, W - pw - pad))
            if _ok(yy, xx):
                return _put(yy, xx)
        self.stats["배치 실패"] += 1
        return None

    # ---------- 합성 이미지 1장 ----------
    def synth_one(self, H, W, rng) -> Optional[Sample]:
        canvas = self.blank_canvas(H, W, rng)
        occ = np.zeros((H, W), np.uint8)
        boxes = []
        seen: set = set()                       # ★ 이미 넣은 클래스
        lo, hi = int(self.pills_range[0]), int(self.pills_range[1])
        if UNIQUE_CLASS_PER_IMAGE:              # 클래스 수보다 많이 넣을 수 없습니다
            hi = max(1, min(hi, len(self.pool)))
            lo = max(1, min(lo, hi))
        n_want = rng.randint(lo, hi)
        # ★ 원본 사진처럼 고르게 배치할 중심 좌표를 미리 정합니다
        slots = plan_uniform_slots((H, W), n_want, rng) if UNIFORM_LAYOUT else [None] * n_want
        for sl in slots:
            b = self.paste_one(canvas, occ, rng, exclude=seen, slot=sl)
            if b:
                boxes.append(b)
                seen.add(int(b[4]))
        if not boxes:
            return None
        return Sample(canvas, boxes, {"kind": "cp_synth", "n_pills": len(boxes)})

    def paste_onto(self, base: Sample, rng) -> Optional[Sample]:
        """원본 train 이미지 위에 알약을 추가로 붙입니다."""
        s = base.clone()
        H, W = s.hw
        occ = np.zeros((H, W), np.uint8)
        for cx, cy, w, h, _ in s.boxes:            # 기존 알약 자리는 점유 처리
            x1, y1 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
            x2, y2 = int(min(W, cx + w / 2)), int(min(H, cy + h / 2))
            occ[y1:y2, x1:x2] = 1
        seen = {int(b[4]) for b in s.boxes}     # ★ 원본에 이미 있는 클래스는 제외
        added = 0
        n_add = rng.randint(*self.extra_range)
        # ★ 이미 알약이 있는 칸을 피해 "빈 칸"에 얹습니다
        free = free_uniform_slots((H, W), s.boxes, rng, n_add) if UNIFORM_LAYOUT else [None] * n_add
        for sl in free:
            b = self.paste_one(s.image, occ, rng, exclude=seen, strict=False, slot=sl)
            if b:
                s.boxes.append(b)
                seen.add(int(b[4]))
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
    cp_weighted: Optional[bool] = None,
    max_crops_per_class: Optional[int] = None,
    crops_dir: Optional[Union[PathLike, Sequence[PathLike]]] = None,  # ★ 크롭 폴더(들)
    cutout_check_dir: Optional[PathLike] = None,   # ★ 컷아웃 검수 저장(cutcheck)
    rebuild_crop_cache: bool = False,
    preprocess_val_test: bool = True,
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
    if cp_weighted is None:
        cp_weighted = DEFAULT_CP_WEIGHTED
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
        _crops_dir = resolve_crop_dirs(crops_dir)
        if verbose and _crops_dir:
            print("    ★ Copy&Paste 재료 소스:")
            for _r in _crops_dir:
                print(f"        {'✅' if _r.is_dir() else '❌'} {_r}")
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
                print(f"   → 크롭 폴더 경로도 확인하세요: {[str(r) for r in _crops_dir]}")
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

    # ---------- 4-B. ★ 원본(알약 3개) + train_another 알약 1개 = 알약 4개 ----------
    #  캔버스 합성과 달리 **진짜 사진**을 배경으로 쓰므로 도메인 갭이 가장 작습니다.
    if ANOTHER_ONTO_N > 0:
        _tgt = ANOTHER_ONTO_BASE_PILLS + ANOTHER_ONTO_ADD
        if verbose:
            print(f"[4-B/5] 원본 사진(알약 {ANOTHER_ONTO_BASE_PILLS}개)에 "
                  f"train_another 알약 {ANOTHER_ONTO_ADD}개를 얹어 "
                  f"알약 {_tgt}개 이미지 {ANOTHER_ONTO_N:,}장 생성")
        _pool = AnotherPillPool(
            ANOTHER_ROOT, names=names,
            max_per_class=ANOTHER_ONTO_MAX_PER_CLASS, verbose=verbose,
        ).build()
        for k, sm in enumerate(build_another_onto_train(
                train_recs, ANOTHER_ONTO_N, rng, _pool, verbose=verbose)):
            emit(sm, f"an_onto4_{k:06d}.png", "an4")
    elif verbose:
        print("[4-B/5] train_another 얹기 건너뜀 (ANOTHER_ONTO_N=0)")

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
        "crops_dir": [str(r) for r in resolve_crop_dirs(crops_dir)],
        "cutout_check_dir": str(cutout_check_dir if cutout_check_dir is not None
                                else CUTOUT_CHECK_DIR or ""),
        "cp_bg_from_crops": CP_BG_FROM_CROPS,
        "bg_mode": BG_MODE,
        "bg_hsv": list(resolve_bg_hsv()),
        "bg_jitter": list(BG_JITTER),
        "uniform_layout": UNIFORM_LAYOUT,
        "unique_class_per_image": UNIQUE_CLASS_PER_IMAGE,
        "another_onto": {
            "n": ANOTHER_ONTO_N,
            "base_pills": ANOTHER_ONTO_BASE_PILLS,
            "add_pills": ANOTHER_ONTO_ADD,
            "made": counts["an4"],
        },
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
    n_train = counts["orig"] + counts["aug"] + counts["cp"] + counts["an4"]
    target = len(train_recs) * geom_mult
    if verbose:
        print("\n■ 생성 결과")
        print(f"  train  원본 {counts['orig']:,} + 증강 {counts['aug']:,} "
              f"+ Copy&Paste {counts['cp']:,} + another얹기 {counts['an4']:,} "
              f"= {n_train:,}장")
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
        if ANOTHER_ONTO_N:
            print(f"  ★ another얹기 {counts['an4']:,} / 목표 {ANOTHER_ONTO_N:,}장 "
                  f"(알약 {ANOTHER_ONTO_BASE_PILLS + ANOTHER_ONTO_ADD}개) "
                  f"{'✅' if counts['an4'] == ANOTHER_ONTO_N else '⚠️ 재료 부족분 손실'}")
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
        if p.name.startswith("cp_"):
            kind = "cp"
        elif p.name.startswith("aug_"):
            kind = "aug"
        elif p.name.startswith("an_onto4_"):     # ★ 원본 사진 + train_another 알약
            kind = "an_onto4"
        elif p.name.startswith("an_"):           # train_another 캔버스 합성
            kind = "another"
        else:
            kind = "orig"
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
#  Part 8. ★ train_another 통합 — 검수패널 제거 · 크기그룹 EDA · 스케일 보존 병합
# ═══════════════════════════════════════════════════════════════════════════
#
#  폴더 구조 (클래스당 폴더 1개, 폴더 안은 알약 1개짜리 크롭 이미지)
#
#      train_another/
#      ├── 3832_뉴로메드정(옥시라세탐)/
#      │     ├── K-003351-003832-016232_0_2_0_2_90_000_200.png
#      │     └── ...
#      └── 3351_일양하이트린정2mg/
#
#  ★ 이 크롭들은 순수 사진이 아니라 **검수용 패널**입니다.
#      상단에 "3832 | 뉴로메드정(...)" 글자 배너, 그 아래 색 테두리가
#      픽셀로 박혀 있습니다. 그대로 학습에 넣으면 모델이 배너를 외웁니다.
#      → strip_review_panel() 이 자동으로 잘라냅니다.
#
#  ★ 진짜 문제는 해상도가 아니라 **객체 스케일**입니다.
#      train_images  : 976x1280 안에서 알약 bbox 면적비 약 5.6%
#      train_another : 396x396  안에서 알약 bbox 면적비 약 90%
#      두 쪽을 그냥 섞고 imgsz 로 letterbox 하면 같은 약이 180px 와 900px
#      두 가지 크기로 보여, 테스트에 없는 이중 스케일 분포를 학습합니다.
#
#      다행히 크롭은 원본에서 **리사이즈 없이 잘라낸 1:1 배율**입니다.
#      그래서 크롭을 확대·축소하지 않고 train_images 와 같은 크기의 캔버스에
#      그대로 얹으면 알약 픽셀 크기가 원본과 일치합니다  → mode="canvas"
#
#  세 가지 모드
#  ------------
#  "canvas" (★ 기본, 권장)
#      크롭을 리사이즈하지 않고 976x1280 캔버스에 배치. 배경은 크롭 자신의
#      배경 색조로 채웁니다. 알약 픽셀 크기 · 이미지 크기 · 종횡비가 전부
#      train_images 와 같아져 한 번의 학습으로 끝납니다.
#
#  "native"
#      크롭을 **원본 크기 그대로** 내보냅니다. 요청하신 "원본 이미지 크기 유지"
#      의 문자 그대로의 구현입니다. 크기별로 그룹을 나눠 저장하므로
#      그룹마다 따로 학습하거나 rect 배치를 쓸 수 있습니다.
#      ⚠️ 객체 스케일 불일치는 남습니다. 반드시 canvas 와 비교 실험하세요.
#
#  "cp_only"
#      YOLO 데이터로 만들지 않고 Copy&Paste 재료로만 씁니다.
#      (기존 CROPPED_PILLS_DIR 경로와 같은 방식)
#
# ---------------------------------------------------------------------------

# ---------- 경로 ----------
#  ★ None = import 시 자동 탐색 ({PILL_ROOT}/train_another)
ANOTHER_ROOT: Optional[str] = None

# ---------- 동작 ----------
ANOTHER_MODE = "canvas"          # "canvas" | "native" | "cp_only"
ANOTHER_STRIP_PANEL = True       # ★ 검수 배너/테두리 자동 제거
ANOTHER_BBOX_SOURCE = "cutout"   # "cutout" = 컷아웃으로 딱 맞는 박스 | "full" = 크롭 전체
ANOTHER_BBOX_MARGIN = 0.02       # bbox 여유 (크롭 짧은 변 대비 비율)
ANOTHER_MAX_PER_CLASS = 0        # 0 = 전부
ANOTHER_MIN_AREA_RATIO = 0.02    # 컷아웃 결과가 크롭의 이 비율 미만이면 실패 처리

# ---------- canvas 모드 ----------
ANOTHER_CANVAS_HW: Optional[Tuple[int, int]] = None  # None = train 원본 중앙값 자동
ANOTHER_PILLS_PER_CANVAS = (2, 4)   # 캔버스 1장에 얹을 크롭 수 (원본 분포와 동일)
# ★ 크롭이 리사이즈된 경우에만 켜세요. 크롭의 알약 짧은 변을 train_images 의
#   중앙값에 맞춰 배율을 보정합니다. 1:1 크롭이면 끄는 편이 정확합니다.
ANOTHER_RESCALE_TO_TRAIN = False
ANOTHER_RESCALE_CLAMP = (0.4, 2.5)  # 보정 배율 상·하한 (과보정 방지)
ANOTHER_CANVAS_OVERLAP = 0.0        # ★ 박스 겹침 허용 비율 (0 = 겹치지 않음)
ANOTHER_CANVAS_FEATHER = 2          # 경계 페더링(px)
ANOTHER_CANVAS_EDGE_PAD = 0.04      # 가장자리 여백 (캔버스 짧은 변 대비)
ANOTHER_CANVAS_SHADOW = True        # ★ 알약 밑에 그림자 (원본 사진처럼)
ANOTHER_PLACE_TRIES = 120           # 자리 찾기 재시도 횟수
ANOTHER_MIN_FIT_SCALE = 0.55        # 캔버스에 넣으려 줄일 수 있는 최소 배율

# ---------- native 모드 ----------
ANOTHER_GROUP_TOL = 16           # 크기 그룹 묶음 허용 오차(px). 0 이면 정확히 같은 크기끼리만

# ---------- 배너 탐지 파라미터 ----------
PANEL_BANNER_MAX_FRAC = 0.30     # 배너 높이 상한 (이미지 높이 대비)
PANEL_COLOR_TOL = 12             # 색 동일 판정 허용치
PANEL_BANNER_MIN_COV = 0.45      # 한 행에서 배너 배경색이 차지해야 할 최소 비율
PANEL_FRAME_MAX_PX = 14          # 테두리 두께 상한
PANEL_FRAME_MIN_COV = 0.90       # 테두리 선의 균일도

ANOTHER_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# ---------------------------------------------------------------------------
#  8-1. 검수 패널(배너 + 테두리) 자동 제거
# ---------------------------------------------------------------------------

def _modal_color(line: np.ndarray, tol: int = PANEL_COLOR_TOL) -> Tuple[np.ndarray, float]:
    """한 줄(행 또는 열)의 최빈 색과 그 색이 차지하는 비율을 돌려줍니다."""
    q = line.astype(np.int32) // max(1, tol)
    key = q[:, 0] * 1_000_000 + q[:, 1] * 1_000 + q[:, 2]
    vals, cnts = np.unique(key, return_counts=True)
    k = vals[int(np.argmax(cnts))]
    return line[key == k].mean(axis=0), float(cnts.max()) / len(key)


def detect_banner_height(img: np.ndarray) -> int:
    """상단 글자 배너의 높이(px). 배너가 없으면 0.

    배너는 '거의 한 가지 색 + 소수의 글자 픽셀' 이라는 성질로 찾습니다.
    사진 영역은 이 조건을 만족하지 않으므로 자연스럽게 멈춥니다.
    """
    if img is None or img.ndim != 3:
        return 0
    H, W = img.shape[:2]
    if H < 40 or W < 20:
        return 0

    c0, cov0 = _modal_color(img[0])
    if cov0 < PANEL_BANNER_MIN_COV:
        return 0

    limit = max(4, int(H * PANEL_BANNER_MAX_FRAC))
    h = 0
    for y in range(limit):
        c, cov = _modal_color(img[y])
        if cov < PANEL_BANNER_MIN_COV or float(np.abs(c - c0).max()) > 3 * PANEL_COLOR_TOL:
            break
        h = y + 1

    if h <= 2 or h >= limit:
        return 0

    # 배너 아래(사진 영역)가 배너와 같은 색이면 배너가 아니라 그냥 균일한 배경입니다.
    body = img[h:min(H, h + 20)]
    if body.size and float(np.abs(body.reshape(-1, 3).mean(axis=0) - c0).max()) < 3 * PANEL_COLOR_TOL:
        return 0
    return h


def detect_frame_px(img: np.ndarray) -> Dict[str, int]:
    """네 변의 색 테두리 두께(px)."""
    out = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    if img is None or img.ndim != 3:
        return out
    H, W = img.shape[:2]
    if H < 20 or W < 20:
        return out

    inner = img[H // 4:3 * H // 4, W // 4:3 * W // 4]
    if inner.size == 0:
        return out
    inner_med = np.median(inner.reshape(-1, 3), axis=0)

    cap = max(1, min(PANEL_FRAME_MAX_PX, min(H, W) // 4))
    for side in out:
        n = 0
        for i in range(cap):
            if side == "top":
                line = img[i]
            elif side == "bottom":
                line = img[H - 1 - i]
            elif side == "left":
                line = img[:, i]
            else:
                line = img[:, W - 1 - i]
            c, cov = _modal_color(line)
            # 균일한 선이면서 사진 내부 색과 뚜렷이 다를 때만 테두리로 봅니다.
            if cov < PANEL_FRAME_MIN_COV or float(np.abs(c - inner_med).max()) < 30:
                break
            n = i + 1
        out[side] = n
    return out


def strip_review_panel(img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """검수 패널에서 사진 영역만 잘라 냅니다.

    Returns:
        (crop, info)  info = {"banner": px, "frame": {...}, "stripped": bool}
    """
    info: Dict[str, Any] = {"banner": 0, "frame": {"top": 0, "bottom": 0, "left": 0, "right": 0},
                            "stripped": False}
    if img is None or img.ndim != 3:
        return img, info

    bh = detect_banner_height(img)
    body = img[bh:] if bh else img
    fr = detect_frame_px(body)
    y0, y1 = fr["top"], body.shape[0] - fr["bottom"]
    x0, x1 = fr["left"], body.shape[1] - fr["right"]
    if y1 - y0 < 16 or x1 - x0 < 16:      # 너무 많이 깎였으면 원본 유지
        return img, info

    info["banner"] = int(bh)
    info["frame"] = fr
    info["stripped"] = bool(bh or any(fr.values()))
    return body[y0:y1, x0:x1], info


# ---------------------------------------------------------------------------
#  8-2. 크롭 → bbox
# ---------------------------------------------------------------------------

def another_bbox(
    crop: np.ndarray,
    source: str = None,
    margin: float = None,
) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[np.ndarray], str]:
    """크롭 이미지에서 알약 bbox 를 구합니다.

    Returns:
        (bbox_xywh | None, mask | None, how)   how = "cutout" | "full" | "fail"
    """
    if source is None:
        source = ANOTHER_BBOX_SOURCE
    if margin is None:
        margin = ANOTHER_BBOX_MARGIN
    if crop is None or crop.ndim != 3:
        return None, None, "fail"

    h, w = crop.shape[:2]
    pad = margin * min(h, w)

    if source == "cutout":
        m = cutout_pill(crop)
        if m is not None:
            mb = m.astype(bool)
            ratio = float(mb.sum()) / float(h * w)
            if ratio >= ANOTHER_MIN_AREA_RATIO:
                ys, xs = np.nonzero(mb)
                x1 = max(0.0, float(xs.min()) - pad)
                y1 = max(0.0, float(ys.min()) - pad)
                x2 = min(float(w), float(xs.max()) + 1 + pad)
                y2 = min(float(h), float(ys.max()) + 1 + pad)
                return (x1, y1, x2 - x1, y2 - y1), m, "cutout"

    # fallback — 크롭 전체를 박스로 (크롭이 이미 알약에 딱 맞게 잘려 있다는 가정)
    return (0.0, 0.0, float(w), float(h)), None, ("full" if source == "full" else "fail")


# ---------------------------------------------------------------------------
#  8-3. EDA 스캔
# ---------------------------------------------------------------------------

def size_group_key(w: int, h: int, tol: int = None) -> str:
    """크기를 그룹 키로. tol 이 크면 비슷한 크기를 한 그룹으로 묶습니다."""
    if tol is None:
        tol = ANOTHER_GROUP_TOL
    if tol and tol > 1:
        w = int(round(w / tol) * tol)
        h = int(round(h / tol) * tol)
    return f"{int(w)}x{int(h)}"


def scan_another_root(
    root: PathLike = None,
    names: Optional[Sequence[str]] = None,
    max_per_class: Optional[int] = None,
    strip_panel: Optional[bool] = None,
    compute_bbox: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """train_another 폴더를 스캔해 EDA 레코드를 만듭니다. (이미지를 쓰지는 않습니다)

    Args:
        root:  train_another 폴더. None 이면 전역 ANOTHER_ROOT.
        names: data.yaml 의 클래스명 리스트. 주면 폴더명 → class_id 매칭을 합니다.
        compute_bbox: True 면 컷아웃까지 돌려 bbox·면적비를 계산합니다(느립니다).

    Returns:
        {"records": [...], "groups": {...}, "unmatched": [...], "stats": Counter}
    """
    if root is None:
        root = ANOTHER_ROOT
    if max_per_class is None:
        max_per_class = ANOTHER_MAX_PER_CLASS
    if strip_panel is None:
        strip_panel = ANOTHER_STRIP_PANEL

    if not root or not Path(root).is_dir():
        raise FileNotFoundError(
            f"train_another 폴더가 없습니다: {root}\n"
            "→ 노트북 설정 셀의 ANOTHER_ROOT 경로를 확인하세요."
        )
    root = Path(root)

    lut = _build_name_lookup(list(names)) if names else {}
    records: List[Dict[str, Any]] = []
    unmatched: List[str] = []
    stats: Counter = Counter()

    folders = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if verbose:
        print(f"train_another 스캔  폴더 {len(folders)}개  ← {root}")

    for fi, d in enumerate(folders, 1):
        cid = _match_class_id(d.name, lut) if lut else None
        if lut and cid is None:
            unmatched.append(d.name)
            stats["폴더 매칭 실패"] += 1

        files = [p for p in sorted(d.iterdir()) if p.suffix.lower() in ANOTHER_EXTS]
        if max_per_class:
            files = files[:max_per_class]

        for p in files:
            img = imread_unicode(p)
            if img is None:
                stats["읽기 실패"] += 1
                continue
            H0, W0 = img.shape[:2]

            if strip_panel:
                crop, pinfo = strip_review_panel(img)
            else:
                crop, pinfo = img, {"banner": 0, "frame": {"top": 0, "bottom": 0,
                                                           "left": 0, "right": 0},
                                    "stripped": False}
            h, w = crop.shape[:2]
            if pinfo["stripped"]:
                stats["패널 제거"] += 1

            rec: Dict[str, Any] = {
                "path": str(p),
                "file_name": p.name,
                "stem": p.stem,
                "folder": d.name,
                "class_id": cid,
                "raw_w": int(W0), "raw_h": int(H0),
                "banner_px": int(pinfo["banner"]),
                "frame_px": int(max(pinfo["frame"].values()) if pinfo["frame"] else 0),
                "crop_w": int(w), "crop_h": int(h),
                "size_group": size_group_key(w, h),
                "aspect": round(w / max(h, 1), 4),
            }

            if compute_bbox:
                bb, mask, how = another_bbox(crop)
                rec["bbox_how"] = how
                if bb is None:
                    stats["bbox 실패"] += 1
                    rec.update({"bx": "", "by": "", "bw": "", "bh": "",
                                "box_area_ratio": "", "mask_area_ratio": ""})
                else:
                    x, y, bw_, bh_ = bb
                    rec.update({
                        "bx": round(x, 1), "by": round(y, 1),
                        "bw": round(bw_, 1), "bh": round(bh_, 1),
                        "box_area_ratio": round(bw_ * bh_ / float(w * h), 4),
                        "mask_area_ratio": (round(float(mask.astype(bool).sum()) / float(w * h), 4)
                                            if mask is not None else ""),
                    })
                    stats[f"bbox {how}"] += 1

            records.append(rec)
            stats["이미지"] += 1

        if verbose and (fi % 50 == 0 or fi == len(folders)):
            print(f"    {fi}/{len(folders)} 폴더  누적 {stats['이미지']:,}장")

    groups = Counter(r["size_group"] for r in records)

    if verbose:
        print(f"\n■ 스캔 결과")
        print(f"  이미지        {stats['이미지']:,}장 / 폴더 {len(folders)}개")
        print(f"  검수패널 제거 {stats['패널 제거']:,}장")
        if lut:
            print(f"  클래스 매칭   {len(folders) - len(unmatched)}/{len(folders)} 폴더")
            if unmatched:
                print(f"  ⚠️ 매칭 실패 {len(unmatched)}개 예: {unmatched[:3]}")
        print(f"\n■ 크기 그룹 (허용오차 {ANOTHER_GROUP_TOL}px)")
        for g, n in groups.most_common(12):
            print(f"    {g:<14}{n:>7,}장")
        if len(groups) > 12:
            print(f"    ... 외 {len(groups) - 12}개 그룹")

    return {"records": records, "groups": dict(groups),
            "unmatched": unmatched, "stats": stats, "root": str(root)}


def export_another_eda(scan: Dict[str, Any], out_dir: PathLike, verbose: bool = True) -> Dict[str, str]:
    """스캔 결과를 CSV / JSON 으로 저장합니다. (01_eda 가 호출)"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = scan["records"]

    csv_path = out_dir / "another_meta.csv"
    cols = ["file_name", "folder", "class_id", "raw_w", "raw_h", "banner_px", "frame_px",
            "crop_w", "crop_h", "size_group", "aspect", "bbox_how",
            "bx", "by", "bw", "bh", "box_area_ratio", "mask_area_ratio", "path"]
    import csv as _csv
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)

    per_class = Counter(r["folder"] for r in recs)
    json_path = out_dir / "another_size_groups.json"
    payload = {
        "root": scan["root"],
        "n_images": len(recs),
        "n_folders": len(per_class),
        "groups": scan["groups"],
        "group_tol": ANOTHER_GROUP_TOL,
        "unmatched_folders": scan["unmatched"],
        "per_folder_counts": dict(per_class.most_common()),
        "stats": dict(scan["stats"]),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    if verbose:
        print(f"another_meta.csv          {len(recs):,}행  → {csv_path}")
        print(f"another_size_groups.json  그룹 {len(scan['groups'])}개 → {json_path}")
    return {"meta_csv": str(csv_path), "groups_json": str(json_path)}


def compare_scale_with_train(
    scan: Dict[str, Any],
    train_recs: Sequence[Dict[str, Any]],
    verbose: bool = True,
) -> Dict[str, float]:
    """★ train_images 와 train_another 의 **객체 스케일**을 비교합니다.

    해상도가 아니라 이 지표가 핵심입니다. 두 분포가 크게 다르면
    그냥 섞었을 때 모델이 테스트에 없는 이중 스케일을 학습합니다.

    Args:
        train_recs: `_collect_split(pill_raw, "train")` 의 결과.
    """
    tr_short, tr_area = [], []
    for r in train_recs:
        W, H = r["width"], r["height"]
        for x, y, w, h, _c in r["boxes"]:
            if w > 0 and h > 0:
                tr_short.append(min(w, h))
                tr_area.append(w * h / float(W * H))

    an_short, an_area, an_area_canvas = [], [], []
    for r in scan["records"]:
        if r.get("bw") in ("", None):
            continue
        w, h = float(r["bw"]), float(r["bh"])
        an_short.append(min(w, h))
        an_area.append(w * h / float(r["crop_w"] * r["crop_h"]))

    out: Dict[str, float] = {}
    if tr_short and an_short:
        out = {
            "train_short_median": float(np.median(tr_short)),
            "another_short_median": float(np.median(an_short)),
            "train_area_ratio_median": float(np.median(tr_area)),
            "another_area_ratio_median": float(np.median(an_area)),
        }
        out["short_side_ratio"] = out["another_short_median"] / max(out["train_short_median"], 1e-6)
        out["area_ratio_gap"] = out["another_area_ratio_median"] / max(out["train_area_ratio_median"], 1e-9)

        if verbose:
            print("\n■ ★ 객체 스케일 비교 — 해상도보다 이게 중요합니다")
            print(f"{'':<26}{'train_images':>14}{'train_another':>15}")
            print("-" * 56)
            print(f"{'박스 짧은 변 중앙값(px)':<26}{out['train_short_median']:>14.0f}"
                  f"{out['another_short_median']:>15.0f}")
            print(f"{'이미지 대비 면적비':<26}{out['train_area_ratio_median']:>13.2%}"
                  f"{out['another_area_ratio_median']:>15.2%}")
            print("-" * 56)
            print(f"  알약 실제 픽셀 크기 비율   {out['short_side_ratio']:.2f}배  "
                  f"(1.0 에 가까우면 같은 배율로 잘린 것 ✅)")
            print(f"  화면 점유 면적비 차이      {out['area_ratio_gap']:.1f}배  "
                  f"(그냥 섞으면 이만큼 스케일이 어긋납니다 ⚠️)")
            if 0.7 <= out["short_side_ratio"] <= 1.4:
                print("\n  → 크롭이 리사이즈 없이 1:1 로 잘렸습니다.")
                print("    ANOTHER_MODE='canvas' 로 캔버스에 그대로 얹으면")
                print("    알약 픽셀 크기가 train_images 와 정확히 같아집니다. ★ 권장")
            else:
                print("\n  ⚠️ 크롭이 리사이즈된 것 같습니다. canvas 모드에서")
                print("    ANOTHER_RESCALE_TO_TRAIN=True 로 배율을 맞추세요.")
    return out


# ---------------------------------------------------------------------------
#  8-4. 캔버스 합성 — 크롭을 리사이즈 없이 train 원본 크기에 얹기
# ---------------------------------------------------------------------------

def _canvas_from_tone(hw: Tuple[int, int], tone: Optional[Tuple[float, float]],
                      rng: random.Random) -> np.ndarray:
    """인공 배경을 만듭니다.

    ★ `BG_MODE == "fixed"`(기본) 이면 크롭 색조를 **쓰지 않고**
      `make_bg_canvas()` 의 고정 색(train_images 실측값)을 씁니다.
      크롭마다 배경색이 달라지던 문제를 여기서 끊습니다.
    """
    if BG_MODE == "fixed":
        return make_bg_canvas(hw, rng)
    H, W = hw
    if tone is None:
        tone = (110.0, 40.0)
    hue, sat = tone
    hue = float(np.clip(hue + rng.uniform(-CP_BG_HUE_JITTER, CP_BG_HUE_JITTER), 0, 179))
    sat = float(np.clip(sat * rng.uniform(*CP_BG_S_SCALE), 0, 255))
    val = float(rng.uniform(*CP_BG_V_RANGE))

    hsv = np.empty((H, W, 3), np.uint8)
    hsv[..., 0] = int(round(hue))
    hsv[..., 1] = int(round(sat))
    grad = np.linspace(-CP_BG_GRAD, CP_BG_GRAD, H, dtype=np.float32)[:, None]
    v = np.clip(val + grad + np.random.normal(0, CP_BG_NOISE, (H, W)), 0, 255)
    hsv[..., 2] = v.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _paste_soft(canvas: np.ndarray, patch: np.ndarray, mask: np.ndarray,
                x: int, y: int, feather: int) -> None:
    """패치를 알파 블렌딩으로 캔버스에 얹습니다 (제자리 수정)."""
    h, w = patch.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    px0, py0 = x0 - x, y0 - y
    sub_p = patch[py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)]
    sub_m = mask[py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)].astype(np.float32)
    if feather > 0:
        k = 2 * int(feather) + 1
        sub_m = cv2.GaussianBlur(sub_m, (k, k), 0)
    a = np.clip(sub_m, 0, 1)[..., None]
    roi = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = np.clip(roi * (1 - a) + sub_p.astype(np.float32) * a,
                                   0, 255).astype(np.uint8)


def _iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


# ---------------------------------------------------------------------------
#  8-5. YOLO 데이터로 내보내기
# ---------------------------------------------------------------------------

def build_another_yolo(
    another_root: PathLike = None,
    dst_root: PathLike = None,
    names: Optional[Sequence[str]] = None,
    *,
    mode: Optional[str] = None,
    canvas_hw: Optional[Tuple[int, int]] = None,
    scan: Optional[Dict[str, Any]] = None,
    n_canvas: Optional[int] = None,
    train_short_median: Optional[float] = None,   # ★ 배율 보정 기준 (compare_scale_with_train)
    seed: Optional[int] = None,
    overwrite: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """train_another 를 **YOLO 데이터셋**으로 만듭니다.

    Args:
        mode:      "canvas" (권장) | "native" | "cp_only"
        canvas_hw: canvas 모드에서 쓸 (H, W). None 이면 ANOTHER_CANVAS_HW,
                   그것도 None 이면 (1280, 976).
        n_canvas:  canvas 모드에서 만들 이미지 수. None 이면 크롭을 모두
                   한 번씩 쓰도록 자동 계산합니다.

    산출물::

        dst_root/
        ├── images/train/            ← canvas 모드: 976x1280 합성본
        │                               native 모드: 원본 크기 크롭
        ├── labels/train/
        ├── groups/{WxH}.txt         ← ★ 크기 그룹별 파일 목록 (native 모드)
        ├── another_info.json
        └── data.yaml
    """
    if another_root is None:
        another_root = ANOTHER_ROOT
    if mode is None:
        mode = ANOTHER_MODE
    if seed is None:
        seed = SEED
    if dst_root is None:
        dst_root = Path(another_root).parent / "pill_another"
    dst_root = Path(dst_root)

    if mode not in ("canvas", "native", "cp_only"):
        raise ValueError(f"mode 는 canvas|native|cp_only 중 하나여야 합니다: {mode}")

    rng = random.Random(seed)
    np.random.seed(seed)
    t0 = time.time()

    if verbose:
        print("═" * 66)
        print(f"  train_another → YOLO   mode = {mode}")
        print(f"  입력 {another_root}")
        print(f"  출력 {dst_root}")
        print("═" * 66)

    if scan is None:
        scan = scan_another_root(another_root, names=names, verbose=verbose)
    recs = [r for r in scan["records"] if r.get("class_id") is not None]
    dropped = len(scan["records"]) - len(recs)
    if verbose and dropped:
        print(f"\n⚠️ 클래스 매칭이 안 된 {dropped:,}장은 제외합니다 "
              f"(폴더명이 data.yaml 클래스명과 달라서입니다)")
    if not recs:
        raise RuntimeError(
            "쓸 수 있는 이미지가 없습니다.\n"
            "→ 폴더명이 '{category_id}_{약이름}' 형식인지, names 리스트를 넘겼는지 확인하세요."
        )

    if mode == "cp_only":
        if verbose:
            print("\ncp_only 모드 — YOLO 데이터를 만들지 않습니다.")
            print(f"→ 02 에서 CROPPED_PILLS_DIR 에 {another_root} 를 추가하세요.")
        return {"mode": mode, "n_images": 0, "records": recs, "dst_root": str(dst_root)}

    if overwrite and dst_root.exists():
        shutil.rmtree(dst_root)
    (dst_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dst_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dst_root / "groups").mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    group_files: Dict[str, List[str]] = defaultdict(list)

    # ══════════════════ native 모드 ══════════════════
    if mode == "native":
        if verbose:
            print(f"\n[native] 크롭을 **원본 크기 그대로** 내보냅니다 ({len(recs):,}장)")
        for r in recs:
            img = imread_unicode(r["path"])
            if img is None:
                counts["읽기 실패"] += 1
                continue
            crop, _ = strip_review_panel(img) if ANOTHER_STRIP_PANEL else (img, None)
            h, w = crop.shape[:2]
            if r.get("bw") in ("", None):
                counts["bbox 실패"] += 1
                continue
            box = (float(r["bx"]), float(r["by"]), float(r["bw"]), float(r["bh"]),
                   int(r["class_id"]))
            fn = f"an_{r['folder'][:24]}_{r['stem']}.png"
            fn = re.sub(r"[^\w가-힣.\-]", "_", fn)
            imwrite_unicode(dst_root / "images" / "train" / fn, crop)
            _write_yolo_label(dst_root / "labels" / "train" / f"{Path(fn).stem}.txt",
                              [box], w, h)
            counts["native"] += 1
            group_files[size_group_key(w, h)].append(fn)

    # ══════════════════ canvas 모드 ══════════════════
    else:
        if canvas_hw is None:
            canvas_hw = ANOTHER_CANVAS_HW or (1280, 976)
        CH, CW = int(canvas_hw[0]), int(canvas_hw[1])

        # 캔버스에 안 들어가는 큰 크롭은 미리 걸러 냅니다.
        usable = []
        for r in recs:
            if r.get("bw") in ("", None):
                continue
            if float(r["bw"]) < CW * 0.9 and float(r["bh"]) < CH * 0.9:
                usable.append(r)
            else:
                counts["캔버스보다 큼"] += 1
        if not usable:
            raise RuntimeError(f"크롭이 전부 캔버스({CW}x{CH})보다 큽니다. canvas_hw 를 키우세요.")

        per = ANOTHER_PILLS_PER_CANVAS
        pad = int(ANOTHER_CANVAS_EDGE_PAD * min(CH, CW))

        # ── ★ 클래스별 풀 — 한 장에 같은 종류가 두 번 들어가지 않게 합니다 ──
        by_cls: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for r in usable:
            by_cls[int(r["class_id"])].append(r)
        for v in by_cls.values():
            rng.shuffle(v)
        cursor = {c: 0 for c in by_cls}     # 클래스별 다음에 꺼낼 위치
        used = {c: 0 for c in by_cls}       # 클래스별 사용 횟수
        n_cls = len(by_cls)

        lo = max(1, min(int(per[0]), n_cls if UNIQUE_CLASS_PER_IMAGE else int(per[0])))
        hi = max(lo, min(int(per[1]), n_cls if UNIQUE_CLASS_PER_IMAGE else int(per[1])))

        if n_canvas is None:
            n_canvas = max(1, int(math.ceil(len(usable) / ((lo + hi) / 2.0))))

        bg_hsv = resolve_bg_hsv()

        if verbose:
            print(f"\n[canvas] 크롭을 **리사이즈 없이** {CW}x{CH} 캔버스에 배치")
            print(f"    사용 가능 크롭 {len(usable):,}개 / 클래스 {n_cls}종 "
                  f"→ 합성 이미지 {n_canvas:,}장 (1장당 {lo}~{hi}개)")
            print(f"    배경  = 고정 HSV(H={bg_hsv[0]:.0f}, S={bg_hsv[1]:.0f}, "
                  f"V={bg_hsv[2]:.0f})   [BG_MODE={BG_MODE}]")
            print(f"    규칙  = 같은 클래스 중복 {'금지' if UNIQUE_CLASS_PER_IMAGE else '허용'}"
                  f" / 박스 겹침 허용 {ANOTHER_CANVAS_OVERLAP}"
                  f" / 가장자리 여백 {pad}px (알약이 잘리지 않습니다)")
            if UNIQUE_CLASS_PER_IMAGE and hi < int(per[1]):
                print(f"    ⚠️ 클래스가 {n_cls}종뿐이라 1장당 최대 {hi}개로 낮췄습니다")
            if counts["캔버스보다 큼"]:
                print(f"    ⚠️ 캔버스보다 큰 크롭 {counts['캔버스보다 큼']:,}개 제외")

        def _take(cid: int) -> Dict[str, Any]:
            """클래스 cid 의 크롭을 순환하며 하나 꺼냅니다 (고르게 소진)."""
            lst = by_cls[cid]
            if cursor[cid] >= len(lst):
                rng.shuffle(lst)
                cursor[cid] = 0
            r = lst[cursor[cid]]
            cursor[cid] += 1
            used[cid] += 1
            return r

        for k in range(n_canvas):
            want = rng.randint(lo, hi)

            # ★ 아직 덜 쓴 클래스부터 고릅니다 → 크롭이 고르게 쓰이고 편중이 줄어듭니다
            order = sorted(by_cls, key=lambda c: (used[c] / len(by_cls[c]), rng.random()))
            if UNIQUE_CLASS_PER_IMAGE:
                chosen = order[:want]                     # ← 서로 다른 클래스만
            else:
                chosen = [rng.choice(order) for _ in range(want)]
            picks = [_take(c) for c in chosen]

            canvas = make_bg_canvas((CH, CW), rng, bg_hsv)   # ★ 배경부터 고정색으로
            occ = np.zeros((CH, CW), np.uint8)               # 픽셀 점유 맵
            placed: List[Tuple[float, float, float, float]] = []
            boxes: List[Tuple[float, float, float, float, int]] = []

            # ★ 알약을 화면에 고르게 흩뿌릴 중심 좌표를 미리 정해 둡니다
            slots = (plan_uniform_slots((CH, CW), len(picks), rng, pad=pad)
                     if UNIFORM_LAYOUT else [])

            for r in picks:
                img = imread_unicode(r["path"])
                if img is None:
                    counts["읽기 실패"] += 1
                    continue
                crop, _ = strip_review_panel(img) if ANOTHER_STRIP_PANEL else (img, None)
                bx, by = float(r["bx"]), float(r["by"])
                bw_, bh_ = float(r["bw"]), float(r["bh"])

                # ★ 배율 보정 — 크롭이 리사이즈된 데이터일 때만.
                #   1:1 크롭이면 ANOTHER_RESCALE_TO_TRAIN=False 로 두는 편이 정확합니다.
                if ANOTHER_RESCALE_TO_TRAIN and train_short_median:
                    sc = float(train_short_median) / max(min(bw_, bh_), 1e-6)
                    sc = float(np.clip(sc, *ANOTHER_RESCALE_CLAMP))
                    if abs(sc - 1.0) > 0.02:
                        interp = cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC
                        crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=interp)
                        bx, by, bw_, bh_ = bx * sc, by * sc, bw_ * sc, bh_ * sc
                        counts["배율 보정"] += 1

                ch_, cw_ = crop.shape[:2]

                # ★ 크롭 **전체**가 여백 안에 들어가야 알약이 잘리지 않습니다
                fit_w, fit_h = CW - 2 * pad, CH - 2 * pad
                if cw_ > fit_w or ch_ > fit_h:
                    s2 = min(fit_w / float(cw_), fit_h / float(ch_))
                    if s2 < ANOTHER_MIN_FIT_SCALE:
                        counts["캔버스보다 큼"] += 1
                        continue
                    crop = cv2.resize(crop, None, fx=s2, fy=s2,
                                      interpolation=cv2.INTER_AREA)
                    bx, by, bw_, bh_ = bx * s2, by * s2, bw_ * s2, bh_ * s2
                    ch_, cw_ = crop.shape[:2]
                    counts["여백맞춤 축소"] += 1

                m = cutout_pill(crop)
                if m is None:
                    m = np.ones((ch_, cw_), np.uint8)
                m = (np.asarray(m) > 0).astype(np.uint8)
                reg = m > 0
                if int(reg.sum()) <= 0:
                    counts["컷아웃 실패"] += 1
                    continue

                # ── ★ 배치: 배정받은 슬롯(격자 칸) 중심에 알약 박스를 맞춥니다 ──
                slot = slots.pop() if slots else None

                def _try_at(cx: float, cy: float):
                    """알약 bbox 중심이 (cx, cy) 에 오도록 붙일 좌상단을 계산·검사."""
                    ox_ = int(round(cx - (bx + bw_ / 2.0)))
                    oy_ = int(round(cy - (by + bh_ / 2.0)))
                    ox_ = int(np.clip(ox_, pad, CW - cw_ - pad))
                    oy_ = int(np.clip(oy_, pad, CH - ch_ - pad))
                    cd = (ox_ + bx, oy_ + by, bw_, bh_)
                    if any(_iou_xywh(cd, q) > ANOTHER_CANVAS_OVERLAP for q in placed):
                        return None
                    if int(np.count_nonzero(occ[oy_:oy_ + ch_, ox_:ox_ + cw_][reg])) > 0:
                        return None
                    return ox_, oy_, cd

                ok, ox, oy, cand = False, pad, pad, None
                got = None
                if slot is not None:
                    cell = (CW / 2.0, CH / 2.0)
                    got = _try_at(*slot)
                    for _try in range(UNIFORM_SLOT_TRIES):   # 칸 안에서 조금씩 흔들며 재시도
                        if got is not None:
                            break
                        got = _try_at(*jitter_slot(slot, cell, rng, 0.18))
                if got is None:                              # 슬롯이 막히면 무작위 탐색
                    for _try in range(ANOTHER_PLACE_TRIES):
                        got = _try_at(rng.uniform(pad, CW - pad), rng.uniform(pad, CH - pad))
                        if got is not None:
                            break
                if got is None:
                    counts["배치 실패"] += 1
                    continue
                ox, oy, cand = got
                ok = True

                if ANOTHER_CANVAS_SHADOW:
                    cast_shadow_on(canvas, m, oy, ox, rng)      # ★ 그림자 먼저
                _paste_soft(canvas, crop, m.astype(np.float32),
                            ox, oy, ANOTHER_CANVAS_FEATHER)     # ★ 알약을 그 위에
                occ[oy:oy + ch_, ox:ox + cw_][reg] = 1
                placed.append(cand)
                boxes.append((cand[0], cand[1], cand[2], cand[3], int(r["class_id"])))

            if not boxes:
                counts["빈 캔버스"] += 1
                continue

            counts[f"알약 {len(boxes)}개"] += 1
            fn = f"an_cv_{k:06d}.png"
            imwrite_unicode(dst_root / "images" / "train" / fn, canvas)
            _write_yolo_label(dst_root / "labels" / "train" / f"{Path(fn).stem}.txt",
                              boxes, CW, CH)
            counts["canvas"] += 1
            counts["박스"] += len(boxes)
            group_files[size_group_key(CW, CH)].append(fn)

            if verbose and ((k + 1) % 200 == 0 or k + 1 == n_canvas):
                print(f"    {k + 1:,}/{n_canvas:,} 합성")

        if verbose:
            dist = ", ".join(f"{i}개={counts[f'알약 {i}개']:,}장"
                             for i in range(1, hi + 1) if counts[f"알약 {i}개"])
            print(f"    1장당 알약 분포  {dist}")
            print(f"    크롭 사용 편차   최소 {min(used.values())}회 / "
                  f"최대 {max(used.values())}회 (클래스 기준)")

    # ---------- 그룹 목록 + data.yaml ----------
    for g, fs in group_files.items():
        (dst_root / "groups" / f"{g}.txt").write_text("\n".join(sorted(fs)), encoding="utf-8")

    names_list = list(names) if names else []
    yaml_path = dst_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {dst_root.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/train\n")
        f.write(f"nc: {len(names_list)}\nnames:\n")
        for i, n in enumerate(names_list):
            f.write(f"  {i}: {n}\n")

    info = {
        "mode": mode,
        "another_root": str(another_root),
        "counts": dict(counts),
        "size_groups": {g: len(fs) for g, fs in group_files.items()},
        "canvas_hw": list(canvas_hw) if mode == "canvas" else None,
        "rescale_to_train": bool(ANOTHER_RESCALE_TO_TRAIN and train_short_median),
        "train_short_median": float(train_short_median) if train_short_median else None,
        "strip_panel": ANOTHER_STRIP_PANEL,
        "bbox_source": ANOTHER_BBOX_SOURCE,
        "bg_mode": BG_MODE,
        "bg_hsv": list(resolve_bg_hsv()),
        "unique_class_per_image": bool(UNIQUE_CLASS_PER_IMAGE),
        "pills_per_canvas": list(ANOTHER_PILLS_PER_CANVAS),
        "canvas_overlap": ANOTHER_CANVAS_OVERLAP,
        "seed": seed,
    }
    (dst_root / "another_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    if verbose:
        n_out = counts["canvas"] + counts["native"]
        print(f"\n■ 생성 결과  {n_out:,}장")
        for g, fs in sorted(group_files.items(), key=lambda kv: -len(kv[1])):
            print(f"    크기그룹 {g:<14}{len(fs):>7,}장")
        if counts["배율 보정"]:
            print(f"    배율 보정 {counts['배율 보정']:,}개 "
                  f"(기준 짧은 변 {train_short_median:.0f}px)")
        for k in ("배치 실패", "빈 캔버스", "bbox 실패", "읽기 실패", "캔버스보다 큼"):
            if counts[k]:
                print(f"    ⚠️ {k}: {counts[k]:,}")
        print(f"  소요 {time.time() - t0:.1f}초")
        print(f"★ data.yaml = {yaml_path}")

    info["records"] = recs
    info["dst_root"] = str(dst_root)
    info["yaml"] = str(yaml_path)
    return info


# ---------------------------------------------------------------------------
#  8-6. ★ train_images(알약 3개) + train_another 알약 1개 → 알약 4개 합성
# ---------------------------------------------------------------------------

def _tight_patch(bgr: np.ndarray, mask: np.ndarray):
    """마스크의 최소 외접 사각형으로 잘라 냅니다."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return bgr[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def _rotate_patch(bgr: np.ndarray, mask: np.ndarray, deg: float):
    """알약 패치를 회전하고 다시 딱 맞게 잘라 냅니다."""
    h, w = mask.shape[:2]
    diag = int(np.ceil(np.hypot(h, w)))
    ci = np.zeros((diag, diag, 3), np.uint8)
    cm = np.zeros((diag, diag), np.uint8)
    oy, ox = (diag - h) // 2, (diag - w) // 2
    ci[oy:oy + h, ox:ox + w] = bgr
    cm[oy:oy + h, ox:ox + w] = mask
    M = cv2.getRotationMatrix2D((diag / 2, diag / 2), float(deg), 1.0)
    ci = cv2.warpAffine(ci, M, (diag, diag), flags=cv2.INTER_CUBIC)
    cm = cv2.warpAffine(cm, M, (diag, diag), flags=cv2.INTER_NEAREST)
    return _tight_patch(ci, cm)


class AnotherPillPool:
    """★ `train_another` 크롭에서 **알약만 오려 낸 패치**를 꺼내 쓰는 재료 창고.

    검수 패널(배너·테두리)을 떼고 컷아웃한 뒤 딱 맞게 잘라 캐시에 담습니다.
    필요할 때 하나씩 읽으므로 메모리를 크게 쓰지 않습니다.
    """

    def __init__(
        self,
        another_root: Optional[PathLike] = None,
        names: Optional[Sequence[str]] = None,
        max_per_class: int = 0,
        verbose: bool = True,
    ):
        self.root = Path(another_root or ANOTHER_ROOT or "")
        self.names = list(names) if names else []
        self.max_per_class = int(max_per_class)
        self.verbose = bool(verbose)
        self.by_cls: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self.cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.stats: Counter = Counter()

    # ---------- 구축 ----------
    def build(self, scan: Optional[Dict[str, Any]] = None) -> "AnotherPillPool":
        if not self.root or not self.root.is_dir():
            if self.verbose:
                print(f"⚠️ train_another 폴더가 없어 건너뜁니다: {self.root}")
            return self
        if scan is None:
            scan = scan_another_root(self.root, names=self.names, verbose=False)
        for r in scan["records"]:
            if r.get("class_id") is None or r.get("bw") in ("", None):
                continue
            self.by_cls[int(r["class_id"])].append(r)
        if self.max_per_class > 0:
            for c, lst in self.by_cls.items():
                self.by_cls[c] = lst[: self.max_per_class]
        if self.verbose:
            n = sum(len(v) for v in self.by_cls.values())
            print(f"    train_another 재료 {n:,}개 / 클래스 {len(self.by_cls)}종")
        return self

    @property
    def classes(self) -> List[int]:
        return sorted(self.by_cls)

    # ---------- 조회 ----------
    def _load(self, rec: Dict[str, Any]):
        key = str(rec["path"])
        if key in self.cache:
            return self.cache[key]
        img = imread_unicode(rec["path"])
        if img is None:
            self.stats["읽기 실패"] += 1
            return None
        crop, _ = strip_review_panel(img) if ANOTHER_STRIP_PANEL else (img, None)
        m = cutout_pill(crop)
        if m is None:
            self.stats["컷아웃 실패"] += 1
            return None
        got = _tight_patch(crop, (np.asarray(m) > 0).astype(np.uint8))
        if got is None:
            self.stats["컷아웃 실패"] += 1
            return None
        if len(self.cache) < 400:          # 메모리 상한
            self.cache[key] = got
        return got

    def patch(self, rng: random.Random, exclude: Optional[set] = None,
              rotate: bool = True):
        """(bgr, mask, class_id) 를 하나 돌려줍니다. 실패하면 None."""
        cands = [c for c in self.classes if not (exclude and c in exclude)]
        if not cands:
            self.stats["남은 클래스 없음"] += 1
            return None
        rng.shuffle(cands)
        for cid in cands[:8]:
            lst = self.by_cls[cid]
            for rec in rng.sample(lst, k=min(4, len(lst))):
                got = self._load(rec)
                if got is None:
                    continue
                bgr, mask = got
                if rotate and ANOTHER_ONTO_ROTATE:
                    rot = _rotate_patch(bgr, mask, rng.uniform(0, 360))
                    if rot is None:
                        continue
                    bgr, mask = rot
                return bgr, mask, int(cid)
        self.stats["패치 생성 실패"] += 1
        return None


def build_another_onto_train(
    train_recs: Sequence[Dict[str, Any]],
    n: int,
    rng: random.Random,
    pool: "AnotherPillPool",
    base_pills: Optional[int] = None,
    add_pills: Optional[int] = None,
    verbose: bool = True,
):
    """★ **알약 3개짜리 원본 사진** 위에 train_another 알약을 1개 얹어 4개로 만듭니다.

    배경·조명·그림자가 실제 사진 그대로라 캔버스 합성보다 도메인 갭이 작습니다.
    새로 얹는 알약은
      · 원본에 이미 있는 클래스는 피하고 (한 장에 같은 종류 두 번 금지)
      · 기존 알약이 없는 **빈 칸**에 놓아 배치가 고르게 유지되고
      · 원본 알약 크기의 `ANOTHER_ONTO_SIZE_CLAMP` 배 범위로 맞춥니다.

    Yields:
        Sample — 알약 (base_pills + add_pills) 개짜리 합성 이미지.
    """
    if base_pills is None:
        base_pills = ANOTHER_ONTO_BASE_PILLS
    if add_pills is None:
        add_pills = ANOTHER_ONTO_ADD

    bases = [r for r in train_recs
             if len(r["boxes"]) == int(base_pills)
             and not Path(r["file_name"]).stem.startswith(("an_", "cp_", "aug_"))]
    if verbose:
        print(f"    재료 원본(알약 {base_pills}개) {len(bases):,}장 / "
              f"train_another 클래스 {len(pool.classes)}종")
    if not bases or not pool.classes:
        if verbose:
            print("    ⚠️ 재료가 없어 건너뜁니다 "
                  "(알약 3개짜리 원본 또는 train_another 크롭 부족)")
        return

    order: List[Dict[str, Any]] = []
    made = 0
    tried = 0
    lo, hi = ANOTHER_ONTO_SIZE_CLAMP
    while made < int(n) and tried < int(n) * 6:
        tried += 1
        if not order:
            order = list(bases)
            rng.shuffle(order)
        r = order.pop()

        img = imread_unicode(r["src"])
        if img is None:
            continue
        H, W = img.shape[:2]
        base_boxes = [tuple(b) for b in r["boxes"] if b[2] > 0 and b[3] > 0]
        if len(base_boxes) != int(base_pills):
            continue
        sample = Sample.from_xywh(img, base_boxes)     # boxes = 중심 좌표 형식

        # 기존 알약 자리를 점유 처리
        occ = np.zeros((H, W), np.uint8)
        for cx, cy, w, h, _c in sample.boxes:
            x1, y1 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
            x2, y2 = int(min(W, cx + w / 2)), int(min(H, cy + h / 2))
            occ[y1:y2, x1:x2] = 1

        med_short = float(np.median([min(b[2], b[3]) for b in sample.boxes]))
        seen = {int(b[4]) for b in sample.boxes}
        added = 0

        for slot in free_uniform_slots((H, W), sample.boxes, rng, int(add_pills)):
            got = pool.patch(rng, exclude=seen if UNIQUE_CLASS_PER_IMAGE else None)
            if got is None:
                break
            bgr, mask, cid = got

            # ---- 크기 보정: 원본 알약 중앙값 대비 lo~hi 배 안으로 ----
            ph, pw = mask.shape[:2]
            ratio = min(pw, ph) / max(med_short, 1e-6)
            if ratio < lo or ratio > hi:
                sc = (lo if ratio < lo else hi) / max(ratio, 1e-6)
                nw, nh = max(8, int(round(pw * sc))), max(8, int(round(ph * sc)))
                interp = cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC
                bgr = cv2.resize(bgr, (nw, nh), interpolation=interp)
                mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
                ph, pw = nh, nw
            if ph >= H or pw >= W:
                continue

            reg = mask > 0
            area = int(reg.sum())
            if area <= 0:
                continue

            pad = min(int(UNIFORM_EDGE_PAD * min(H, W)),
                      max(0, (H - ph) // 2), max(0, (W - pw) // 2))
            cx, cy = slot
            placed_ok = False
            for _t in range(UNIFORM_SLOT_TRIES + 1):
                xx = int(np.clip(round(cx - pw / 2.0), pad, W - pw - pad))
                yy = int(np.clip(round(cy - ph / 2.0), pad, H - ph - pad))
                if int(np.count_nonzero(occ[yy:yy + ph, xx:xx + pw][reg])) == 0:
                    placed_ok = True
                    break
                cx, cy = jitter_slot(slot, (W / 2.0, H / 2.0), rng, 0.18)
            if not placed_ok:
                continue

            if ANOTHER_ONTO_SHADOW:
                cast_shadow_on(sample.image, mask, yy, xx, rng)
            _paste_soft(sample.image, bgr, mask.astype(np.float32), xx, yy, CP_FEATHER)
            occ[yy:yy + ph, xx:xx + pw][reg] = 1
            sample.boxes.append((xx + pw / 2.0, yy + ph / 2.0,
                                 float(pw), float(ph), int(cid)))
            seen.add(int(cid))
            added += 1

        if added == 0:
            continue
        sample.meta = {"kind": "another_onto", "n_pills": len(sample.boxes),
                       "base": r["file_name"]}
        made += 1
        yield sample

    if verbose:
        print(f"    생성 {made:,}/{int(n):,}장  {dict(pool.stats) if pool.stats else ''}")


def merge_yolo_into(
    src_root: PathLike,
    dst_root: PathLike,
    src_split: str = "train",
    dst_split: str = "train",
    prefix: str = "",
    verbose: bool = True,
) -> int:
    """`src_root/images/{src_split}` 를 `dst_root/images/{dst_split}` 에 합칩니다.

    02 에서 pill_another 를 pill_raw 의 train 에 얹을 때 씁니다.
    이렇게 하면 이어지는 `build_augmented_yolo_dataset()` 이
    train_another 분량까지 **똑같이 기하 증강**해 줍니다.
    """
    src_root, dst_root = Path(src_root), Path(dst_root)
    si = src_root / "images" / src_split
    sl = src_root / "labels" / src_split
    di = dst_root / "images" / dst_split
    dl = dst_root / "labels" / dst_split
    di.mkdir(parents=True, exist_ok=True)
    dl.mkdir(parents=True, exist_ok=True)

    n = 0
    if not si.is_dir():
        if verbose:
            print(f"⚠️ {si} 가 없습니다 — 병합을 건너뜁니다.")
        return 0
    for p in sorted(si.iterdir()):
        if p.suffix.lower() not in ANOTHER_EXTS:
            continue
        fn = f"{prefix}{p.name}"
        shutil.copy2(p, di / fn)
        lp = sl / f"{p.stem}.txt"
        if lp.exists():
            shutil.copy2(lp, dl / f"{Path(fn).stem}.txt")
        n += 1
    if verbose:
        print(f"병합 {n:,}장  {si}  →  {di}")
    return n


def write_group_yamls(
    dst_root: PathLike,
    names: Sequence[str],
    verbose: bool = True,
) -> List[str]:
    """★ 크기 그룹마다 별도의 data.yaml 을 만듭니다 (native 모드에서 그룹별 학습용).

    Ultralytics 는 폴더 또는 txt 목록 파일을 train 경로로 받으므로
    `groups/{WxH}.txt` 를 절대경로 목록으로 바꿔 쓰게 합니다.
    """
    dst_root = Path(dst_root)
    gdir = dst_root / "groups"
    if not gdir.is_dir():
        return []

    out = []
    for gp in sorted(gdir.glob("*.txt")):
        g = gp.stem
        lines = [str((dst_root / "images" / "train" / fn).resolve())
                 for fn in gp.read_text(encoding="utf-8").splitlines() if fn.strip()]
        if not lines:
            continue
        list_path = dst_root / "groups" / f"{g}_abs.txt"
        list_path.write_text("\n".join(lines), encoding="utf-8")

        y = dst_root / f"data_{g}.yaml"
        with open(y, "w", encoding="utf-8") as f:
            f.write(f"path: {dst_root.resolve()}\n")
            f.write(f"train: groups/{g}_abs.txt\n")
            f.write(f"val: groups/{g}_abs.txt\n")
            f.write(f"nc: {len(names)}\nnames:\n")
            for i, n in enumerate(names):
                f.write(f"  {i}: {n}\n")
        out.append(str(y))
        if verbose:
            print(f"  그룹 {g:<14}{len(lines):>7,}장  → {y.name}")
    return out


def preview_another(
    dst_root: PathLike,
    out_dir: Optional[PathLike] = None,
    n: int = 4,
    names: Optional[Sequence[str]] = None,
) -> List[str]:
    """생성된 train_another 데이터에 박스를 그려 저장합니다 (눈으로 검수)."""
    dst_root = Path(dst_root)
    out_dir = Path(out_dir) if out_dir else dst_root / "_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    files = [p for p in sorted((dst_root / "images" / "train").iterdir())
             if p.suffix.lower() in ANOTHER_EXTS][:n]
    for p in files:
        img = imread_unicode(p)
        if img is None:
            continue
        H, W = img.shape[:2]
        vis = img.copy()
        for x, y, w, h, c in _read_yolo_label(dst_root / "labels" / "train" / f"{p.stem}.txt", W, H):
            col = ((37 * int(c)) % 255, (91 * int(c)) % 255, (151 * int(c)) % 255)
            cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), col, 3)
            # cv2.putText 는 한글을 못 그리므로 숫자 코드만 표시합니다.
            # 한글 라벨이 필요하면 노트북의 draw_detections(PIL 기반)를 쓰세요.
            lab = str(int(c))
            if names and int(c) < len(names):
                m_ = re.match(r"\d+", str(names[int(c)]))
                lab = f"{int(c)}:{m_.group(0)}" if m_ else str(int(c))
            cv2.putText(vis, lab, (int(x), max(14, int(y) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
        op = out_dir / f"{p.stem}_check.png"
        imwrite_unicode(op, vis)
        saved.append(str(op))
    print(f"검수 이미지 {len(saved)}장 → {out_dir}")
    return saved


# ═══════════════════════════════════════════════════════════════════════════
#  셀프 테스트 — 주피터에서도 `pt.selftest()` 로 바로 돌릴 수 있습니다.
# ═══════════════════════════════════════════════════════════════════════════

def check_paths(verbose: bool = True) -> Dict[str, bool]:
    """★ 노트북 첫 셀에서 경로가 실제로 존재하는지 확인합니다.

        import pill_transforms as pt
        pt.check_paths()
    """
    targets = {
        "PROJECT_ROOT": PROJECT_ROOT,          # 저장소 루트 (config.yaml) — 없어도 됨
        "PILL_ROOT": PILL_ROOT,                # ★ 원본 데이터 루트
        "  train_images": str(Path(PILL_ROOT) / "train_images") if PILL_ROOT else None,
        "  train_annotations": str(Path(PILL_ROOT) / "train_annotations") if PILL_ROOT else None,
        "  test_images": str(Path(PILL_ROOT) / "test_images") if PILL_ROOT else None,
        "ANOTHER_ROOT": ANOTHER_ROOT,          # ★ Part 8 — 추가 학습 데이터
        "CROPPED_PILLS_DIR": CROPPED_PILLS_DIR,
        "TEAM_WORK_DIR": TEAM_WORK_DIR,
        "CUTOUT_CHECK_DIR": CUTOUT_CHECK_DIR,
        "WORK_ROOT": WORK_ROOT,                # ★ 산출물 루트
    }
    out = {}
    for k, v in targets.items():
        exists = bool(v) and Path(v).is_dir()
        out[k] = exists
        if verbose:
            optional = k in ("CUTOUT_CHECK_DIR", "WORK_ROOT", "TEAM_WORK_DIR",
                             "PROJECT_ROOT")
            mark = "✅" if exists else ("—" if optional else "❌ 없음")
            print(f"  {k:<20} {v}  {mark}")
    if verbose and not out.get("ANOTHER_ROOT"):
        print("  ※ train_another 를 안 쓰면 무시해도 됩니다 (02 의 USE_ANOTHER=False).")
    if verbose:
        _cd = [r for r in resolve_crop_dirs() if r.is_dir()]
        if _cd:
            print(f"  ※ Copy&Paste 재료 폴더 {len(_cd)}개 사용:")
            for r in _cd:
                n_sub = len([d for d in r.iterdir() if d.is_dir()])
                print(f"       {r}  (클래스 폴더 {n_sub}개)")
        else:
            print("  ⚠️ Copy&Paste 재료 폴더가 없습니다. "
                  "train 라벨 박스에서 직접 컷아웃합니다 (품질이 조금 떨어집니다).")
    if verbose and not out["CUTOUT_CHECK_DIR"]:
        print("  ※ 컷아웃 검수 폴더는 실행할 때 자동으로 만들어집니다.")
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

    if HAS_ALBUMENTATIONS:
        tf = get_train_transform(image_size=320, to_tensor=False)
        r = tf(image=np.random.randint(0, 255, (1280, 976, 3), np.uint8),
               bboxes=[[100, 150, 300, 400]], labels=[1])
        print(f"\nAlbumentations 파이프라인 OK — image {r['image'].shape}, "
              f"bboxes {len(r['bboxes'])}")
    else:
        print("\n(albumentations 미설치 — 온라인 transform 테스트는 건너뜀)")

    n_cut = len(list(Path(CUTOUT_CHECK_DIR).glob("*/cut/*.png")))
    print(f"\n컷아웃 검수 산출물 {n_cut}장 (임시 폴더)")

    shutil.rmtree(tmp, ignore_errors=True)
    CROPPED_PILLS_DIR, CUTOUT_CHECK_DIR = _keep
    print("\n✅ 셀프 테스트 통과 — 이 환경에서 pill_transforms 가 정상 동작합니다.")
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  Part 9-B. ★★★ 실행 환경 · 경로 자동 설정 (Colab / 로컬 공용) ★★★
# ═══════════════════════════════════════════════════════════════════════════
#
#  팀원마다 OS 도 경로도 다릅니다. 그래서 노트북에 경로를 하드코딩하는 대신
#  이 파일이 **환경을 감지해서 경로를 채웁니다.**
#
#      import pill_transforms as pt
#      pt.setup()          # ← 이 한 줄이면 끝
#
#  기대하는 데이터 구조 (zip 을 풀면 이 모양이어야 합니다)
#
#      pilldata/
#      ├── train_images/            원본 사진 (976x1280)
#      ├── train_annotations/       COCO json (폴더=조합키)
#      ├── test_images/             캐글 제출용
#      ├── train_another/           ★ 추가 데이터 — {category_id}_{약이름}/*.png
#      │     └── 3832_뉴로메드정(옥시라세탐)/
#      │           └── K-003351-003832-016232_0_2_0_2_90_000_200.png
#      ├── cropped_pills_review/    Copy&Paste 재료 — {category_id}_{약이름}/*.png
#      └── team_work/               ★ AI Hub 추가분 (팀 가이드 PDF 산출물)
#            └── cropped_output/
#                  ├── {category_id}_{category_name}/*.png
#                  ├── crop_metadata.csv
#                  └── class_summary.csv
#
#  ★★★ 두 가지 폴더 배치를 모두 지원합니다 ★★★
#
#  [A] "project" 배치 — 팀 git 저장소 구조 (config.yaml 이 있는 쪽)
#
#      pill-object-detection/            ← PROJECT_ROOT
#      ├── config.yaml
#      ├── src/
#      │     ├── PillDetectionDataset.py
#      │     └── pill_transforms.py      ← 이 파일
#      ├── notebooks/
#      ├── data/
#      │   ├── dataset/cleaning_data/sprint_ai_project1_data_260809_baseline_dataset/
#      │   │       ├── train_images/  train_annotations/  test_images/
#      │   ├── processed/              YOLO 데이터셋 (pill_detection_dataset.ipynb 산출)
#      │   └── processed_aug/          ★ 증강 결과 (이 모듈이 만듭니다)
#      └── outputs/
#            ├── checkpoints/ predictions/ submissions/     (config.yaml 규약)
#            ├── yolo/                                       (Ultralytics runs)
#            ├── experiments/ figures/ cutcheck/             (이 모듈이 만듭니다)
#
#  [B] "pilldata" 배치 — zip 한 덩어리로 공유하는 구조
#
#      pilldata/                         ← PILL_ROOT
#      ├── train_images/ train_annotations/ test_images/
#      ├── train_another/                {category_id}_{약이름}/*.png
#      ├── cropped_pills_review/         {category_id}_{약이름}/*.png
#      └── team_work/cropped_output/     ★ AI Hub 추가분 (팀 가이드 PDF 산출물)
#
#      이때 산출물은 WORK_ROOT 아래로 분리합니다 (원본을 건드리지 않기 위함).
#          WORK_ROOT/{data/pill_raw, data/pill_aug, outputs, runs, experiments, cutcheck}
#
#  어느 쪽이든 `pt.setup()` 한 줄이면 자동으로 감지해서 경로를 채웁니다.
# ---------------------------------------------------------------------------

PROJECT_ROOT: Optional[str] = None     # [A] 저장소 루트 (config.yaml 이 있는 폴더)
PILL_ROOT: Optional[str] = None        # 원본 데이터 루트 (train_images 의 상위)
WORK_ROOT: Optional[str] = None        # 산출물 루트
LAYOUT: str = "unknown"                # "project" | "pilldata" | "unknown"
CONFIG: Dict[str, Any] = {}            # config.yaml 을 읽었으면 그 내용

#  ★ 저장소(PROJECT_ROOT)를 찾을 후보 — config.yaml + src 또는 data 가 있으면 채택
PROJECT_ROOT_CANDIDATES: Tuple[str, ...] = (
    ".", "..", "../..",
    "/content/pill-object-detection",
    "/content/drive/MyDrive/pill-object-detection",
    "/content/drive/MyDrive/코드잇 AI 13기/AI 13기 프로젝트/pill-object-detection",
    "/content/drive/MyDrive/pill_project/pill-object-detection",
)

#  ★ 원본 데이터(train_images 의 상위)를 찾을 후보 — 위에서부터 확인
PILL_ROOT_CANDIDATES: Tuple[str, ...] = (
    # [A] project 배치 — config.yaml 의 dataset_root
    "./data/dataset/cleaning_data/sprint_ai_project1_data_260809_baseline_dataset",
    "../data/dataset/cleaning_data/sprint_ai_project1_data_260809_baseline_dataset",
    # [B] pilldata 배치
    "/content/pilldata",                              # Colab: zip 을 로컬 디스크에 푼 경우 (★ 가장 빠름)
    "/content/PillData/pilldata",
    "/content/drive/MyDrive/pilldata",                # Colab: Drive 에 그대로 둔 경우 (느림)
    "/content/drive/MyDrive/pill_project/pilldata",
    "/content/drive/MyDrive/PillData/pilldata",
    "D:/PillData/pilldata",                           # Windows 팀원
    "./pilldata", "../pilldata", "../../pilldata",
    "./data/pilldata", "./data/dataset", "../data/dataset",
)

#  ★ Copy&Paste 재료로 쓸 폴더 이름들 (PILL_ROOT 기준 상대경로)
CROP_SOURCE_SUBDIRS: Tuple[str, ...] = (
    "cropped_pills_review",          # 기존 팀 크롭
    "team_work/cropped_output",      # ★ AI Hub 추가분 (팀 공유 가이드 PDF 산출물)
)


def in_colab() -> bool:
    """지금 Google Colab 에서 돌고 있는지."""
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def mount_drive(mount_point: str = "/content/drive", verbose: bool = True) -> Optional[str]:
    """Colab 에서 구글 드라이브를 마운트합니다 (이미 돼 있으면 그대로 둡니다)."""
    if not in_colab():
        if verbose:
            print("※ Colab 이 아니므로 드라이브 마운트를 건너뜁니다.")
        return None
    if Path(mount_point, "MyDrive").is_dir():
        if verbose:
            print(f"✅ 드라이브가 이미 마운트돼 있습니다: {mount_point}")
        return mount_point
    from google.colab import drive  # type: ignore
    drive.mount(mount_point)
    return mount_point


def find_pilldata_zip(search_dirs: Optional[Sequence[PathLike]] = None,
                      verbose: bool = True) -> Optional[str]:
    """드라이브에서 pilldata zip 을 찾아 줍니다 (이름에 'pilldata' 포함, .zip)."""
    if search_dirs is None:
        search_dirs = [
            "/content/drive/MyDrive",
            "/content/drive/MyDrive/pill_project",
            "/content/drive/MyDrive/PillData",
            ".",
        ]
    for d in search_dirs:
        p = Path(d)
        if not p.is_dir():
            continue
        for z in sorted(p.glob("*.zip")):
            if "pilldata" in z.name.lower() or "pill_data" in z.name.lower():
                if verbose:
                    print(f"zip 발견: {z}  ({z.stat().st_size / 1e9:.2f} GB)")
                return str(z)
    return None


def unzip_pilldata(zip_path: Optional[PathLike] = None,
                   dest: PathLike = "/content",
                   force: bool = False,
                   verbose: bool = True) -> Optional[str]:
    """★ 드라이브의 pilldata.zip 을 **로컬 디스크로** 풉니다.

    Colab 에서 드라이브 경로를 직접 읽으면 이미지 수만 장을 한 장씩 네트워크로
    받아오게 되어 학습·증강이 몇 배 느려집니다. zip 한 개만 복사해 로컬에서
    푸는 편이 훨씬 빠릅니다 (보통 몇 분).

    Returns:
        압축을 푼 뒤 찾은 pilldata 루트 경로. 실패하면 None.
    """
    import zipfile

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if zip_path is None:
        zip_path = find_pilldata_zip(verbose=verbose)
    if zip_path is None:
        if verbose:
            print("⚠️ pilldata zip 을 찾지 못했습니다. zip_path 를 직접 지정하세요.")
        return None

    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"zip 이 없습니다: {zip_path}")

    # 이미 풀려 있으면 건너뜁니다
    already = _first_existing(
        [dest / "pilldata", dest / zip_path.stem, dest / zip_path.stem / "pilldata"]
    )
    if already and not force:
        if verbose:
            print(f"✅ 이미 풀려 있습니다 → {already}  (다시 풀려면 force=True)")
        return str(already)

    t0 = time.time()
    if verbose:
        print(f"압축 해제 중… {zip_path}  →  {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    if verbose:
        print(f"완료 ({time.time() - t0:.0f}초)")

    root = _first_existing([dest / "pilldata", dest / zip_path.stem,
                            dest / zip_path.stem / "pilldata"])
    if root is None:
        # zip 안 폴더 이름이 다를 수 있으므로 train_images 를 기준으로 탐색
        for p in sorted(dest.rglob("train_images")):
            if p.is_dir():
                root = p.parent
                break
    if verbose:
        print(f"pilldata 루트: {root}")
    return str(root) if root else None


def _first_existing(paths: Sequence[PathLike]) -> Optional[Path]:
    for p in paths:
        p = Path(p)
        if p.is_dir():
            return p
    return None


def _looks_like_pilldata(p: PathLike) -> bool:
    """train_images 와 train_annotations 가 있으면 데이터 루트로 봅니다."""
    p = Path(p)
    return (p / "train_images").is_dir() and (p / "train_annotations").is_dir()


def detect_pill_root(extra: Optional[Sequence[PathLike]] = None) -> Optional[str]:
    """후보 경로를 훑어 원본 데이터 루트(train_images 의 상위)를 찾습니다."""
    cands: List[PathLike] = list(extra or [])
    # 저장소 루트를 이미 알고 있으면 그 아래를 먼저 봅니다
    for base in (PROJECT_ROOT, os.getcwd()):
        if base:
            cands += [
                Path(base) / "data" / "dataset" / "cleaning_data"
                / "sprint_ai_project1_data_260809_baseline_dataset",
                Path(base) / "data" / "pilldata",
                Path(base) / "pilldata",
            ]
    cands += list(PILL_ROOT_CANDIDATES)
    for c in cands:
        try:
            if _looks_like_pilldata(c):
                return str(Path(c).resolve())
        except Exception:
            continue
    # 마지막 수단 — 얕은 탐색 (cleaning_data 처럼 깊은 구조도 잡히게)
    bases = [".", "/content", "/content/drive/MyDrive"]
    if PROJECT_ROOT:
        bases.insert(0, str(Path(PROJECT_ROOT) / "data"))
    for base in bases:
        b = Path(base)
        if not b.is_dir():
            continue
        try:
            hits = []
            for pat in ("*/train_images", "*/*/train_images", "*/*/*/train_images"):
                hits += list(b.glob(pat))[:40]
            for p in hits:
                if _looks_like_pilldata(p.parent):
                    return str(p.parent.resolve())
        except Exception:
            continue
    return None


def _looks_like_project(p: PathLike) -> bool:
    """config.yaml 또는 src/ + data/ 가 있으면 저장소 루트로 봅니다."""
    p = Path(p)
    if not p.is_dir():
        return False
    if (p / "config.yaml").is_file():
        return True
    return (p / "src").is_dir() and (p / "data").is_dir()


def detect_project_root(extra: Optional[Sequence[PathLike]] = None) -> Optional[str]:
    """config.yaml 이 있는 저장소 루트를 찾습니다 (없으면 None).

    이 파일이 `src/pill_transforms.py` 로 놓여 있으면 그 부모를 먼저 봅니다.
    """
    cands: List[PathLike] = list(extra or [])
    here = Path(__file__).resolve().parent
    cands += [here, here.parent, here.parent.parent]      # src/ 에 있는 경우 대비
    cands += [Path(os.getcwd()), Path(os.getcwd()).parent]
    cands += list(PROJECT_ROOT_CANDIDATES)
    for c in cands:
        try:
            if _looks_like_project(c):
                return str(Path(c).resolve())
        except Exception:
            continue
    # Colab 드라이브에서 이름으로 찾기
    for pat in ("/content/drive/MyDrive/**/pill-object-detection",
                "/content/drive/MyDrive/**/config.yaml"):
        try:
            import glob as _glob
            for hit in _glob.glob(pat, recursive=True)[:20]:
                cand = Path(hit) if Path(hit).is_dir() else Path(hit).parent
                if _looks_like_project(cand):
                    return str(cand.resolve())
        except Exception:
            continue
    return None


def load_config(path: Optional[PathLike] = None, verbose: bool = True) -> Dict[str, Any]:
    """`config.yaml` 을 읽어 dict 로 돌려줍니다.

    `${paths.project_root}` 같은 OmegaConf 보간을 직접 풀어 주므로
    omegaconf 가 없어도 동작합니다. Faster R-CNN 노트북이 쓰는 config 와
    같은 파일을 읽으므로 **경로 규약이 한 곳에서 관리됩니다.**

        cfg = pt.load_config()
        cfg["paths"]["dataset_root"]
        cfg["train"]["epochs"]
    """
    if path is None:
        base = PROJECT_ROOT or os.getcwd()
        path = Path(base) / "config.yaml"
    path = Path(path)
    if not path.is_file():
        if verbose:
            print(f"※ config.yaml 이 없습니다 ({path}) — 기본 경로 규약을 씁니다.")
        return {}
    try:
        import yaml as _yaml
    except ImportError:
        if verbose:
            print("※ pyyaml 이 없어 config.yaml 을 건너뜁니다.")
        return {}

    cfg = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # ---- ${a.b.c} 보간 풀기 (최대 5회 반복) ----
    def _get(dotted: str):
        cur: Any = cfg
        for k in dotted.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    def _resolve(v, depth=0):
        if isinstance(v, str) and "${" in v and depth < 5:
            out = re.sub(r"\$\{([^}]+)\}",
                         lambda m: str(_get(m.group(1)) if _get(m.group(1)) is not None
                                       else m.group(0)), v)
            return _resolve(out, depth + 1) if out != v else out
        if isinstance(v, dict):
            return {k: _resolve(x, depth) for k, x in v.items()}
        if isinstance(v, list):
            return [_resolve(x, depth) for x in v]
        return v

    cfg = _resolve(cfg)
    if verbose:
        print(f"config.yaml 로드: {path}")
    return cfg


def resolve_crop_dirs(
    crops_dir: Optional[Union[PathLike, Sequence[PathLike]]] = None,
) -> List[Path]:
    """Copy&Paste 재료 폴더 목록을 확정합니다.

    우선순위
      1) 인자로 직접 넘긴 경로(하나 또는 여러 개)
      2) 전역 CROPPED_PILLS_DIR + TEAM_WORK_DIR/cropped_output
      3) PILL_ROOT 아래의 CROP_SOURCE_SUBDIRS

    존재하지 않는 경로도 그대로 돌려줍니다 (호출부에서 표시·경고용).
    """
    if crops_dir is not None:
        if isinstance(crops_dir, (str, Path)):
            return [Path(crops_dir)]
        return [Path(c) for c in crops_dir if c]

    out: List[Path] = []
    if CROPPED_PILLS_DIR:
        out.append(Path(CROPPED_PILLS_DIR))
    if TEAM_WORK_DIR:
        tw = Path(TEAM_WORK_DIR) / "cropped_output"
        if tw.is_dir():
            out.append(tw)
        elif Path(TEAM_WORK_DIR).is_dir():
            out.append(Path(TEAM_WORK_DIR))
    if not out and PILL_ROOT:
        for sub in CROP_SOURCE_SUBDIRS:
            out.append(Path(PILL_ROOT) / sub)

    # 중복 제거 (순서 유지)
    seen, uniq = set(), []
    for p in out:
        k = str(p)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def install_colab_deps(ultralytics: bool = True,
                       albumentations: bool = False,
                       korean_font: bool = True,
                       verbose: bool = True) -> None:
    """Colab 세션에 필요한 패키지·폰트를 깝니다 (이미 있으면 건너뜀).

    Colab 은 세션이 끊기면 설치가 사라지므로 노트북 첫 셀에서 호출하세요.
    """
    import subprocess
    import importlib

    def _has(mod: str) -> bool:
        try:
            importlib.import_module(mod)
            return True
        except Exception:
            return False

    todo = []
    if ultralytics and not _has("ultralytics"):
        todo.append("ultralytics")
    if albumentations and not _has("albumentations"):
        todo.append("albumentations")
    if todo:
        if verbose:
            print(f"설치 중: {' '.join(todo)}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *todo], check=False)
    elif verbose:
        print("필요한 파이썬 패키지가 이미 설치돼 있습니다.")

    if korean_font and in_colab():
        if not find_korean_font():
            if verbose:
                print("한글 폰트 설치 중 (fonts-nanum)…")
            subprocess.run(["apt-get", "-qq", "install", "-y", "fonts-nanum"], check=False)
            subprocess.run(["fc-cache", "-f"], check=False)
        if verbose:
            print(f"한글 폰트: {find_korean_font() or '⚠️ 미발견'}")


def find_korean_font() -> Optional[str]:
    """한글 폰트 파일 경로를 찾습니다 (matplotlib·PIL 라벨용).

    OpenCV 는 한글을 못 그리므로 노트북은 PIL + 이 폰트를 씁니다.
    """
    import glob as _glob

    cands = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",      # Colab (apt 설치 후)
        "C:/Windows/Fonts/malgun.ttf",                          # Windows
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",           # macOS
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    ]
    for pat in ("/usr/share/fonts/**/*Nanum*.ttf",
                "/usr/share/fonts/**/*CJK*.ttc",
                "/usr/share/fonts/**/*Gothic*.ttf"):
        cands += sorted(_glob.glob(pat, recursive=True))
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def add_src_to_path(*dirs: PathLike, verbose: bool = True) -> Optional[str]:
    """`PillDetectionDataset.py` 가 있는 폴더를 sys.path 에 넣습니다.

    팀마다 파일 이름이 `PillDetectionDataset.py` 이기도 하고 `pill_dataset.py`
    이기도 해서, 둘 중 무엇이든 `from pill_dataset import PillDetectionDataset`
    로 import 할 수 있게 별칭도 등록합니다.
    """
    cands: List[PathLike] = [d for d in dirs if d]
    cands += [os.getcwd(), Path(os.getcwd()) / "src", Path(os.getcwd()).parent / "src",
              Path(__file__).parent]
    if PILL_ROOT:
        cands += [PILL_ROOT, Path(PILL_ROOT).parent, Path(PILL_ROOT).parent / "src"]
    cands += ["/content", "/content/src", "/content/drive/MyDrive/pill_project/src"]

    found = None
    for d in cands:
        try:
            d = Path(d)
        except Exception:
            continue
        for fname in ("PillDetectionDataset.py", "pill_dataset.py"):
            f = d / fname
            if f.exists():
                if str(d) not in sys.path:
                    sys.path.insert(0, str(d))
                found = str(f)
                break
        if found:
            break

    if found is None:
        if verbose:
            print("⚠️ PillDetectionDataset.py 를 찾지 못했습니다. "
                  "파일 위치를 add_src_to_path('...') 로 알려 주세요.")
        return None

    # ★ 두 이름 모두로 import 되게 별칭 등록
    import importlib
    mod = None
    for name in ("PillDetectionDataset", "pill_dataset"):
        try:
            mod = importlib.import_module(name)
            break
        except Exception:
            continue
    if mod is not None:
        sys.modules.setdefault("pill_dataset", mod)
        sys.modules.setdefault("PillDetectionDataset", mod)
    if verbose:
        print(f"Dataset 모듈: {found}")
    return found


def paths() -> Dict[str, Optional[str]]:
    """지금 설정된 경로를 dict 로 돌려줍니다 (노트북에서 그대로 쓰세요).

        P = pt.setup()
        RAW_DIR, AUG_DIR = P["RAW_DIR"], P["AUG_DIR"]

    ★ 배치에 따라 산출물 위치가 달라집니다.
      · "project" 배치 → PROJECT_ROOT/data/processed(_aug), PROJECT_ROOT/outputs/...
        (팀 저장소·config.yaml 규약과 같은 자리에 떨어집니다)
      · "pilldata" 배치 → WORK_ROOT/data/pill_raw(_aug), WORK_ROOT/outputs/...
    """
    w = Path(WORK_ROOT) if WORK_ROOT else None
    proj = LAYOUT == "project"

    if w is None:
        data_dir = out_dir = run_dir = exp_dir = None
        raw = anod = aug = None
    else:
        data_dir = w / "data"
        out_dir = w / "outputs"
        if proj:
            raw = data_dir / "processed"              # 팀 노트북이 이미 쓰는 이름
            aug = data_dir / "processed_aug"
            anod = data_dir / "processed_another"
            run_dir = out_dir / "yolo"                # yolo11s_baseline_cp.ipynb 규약
            exp_dir = out_dir / "experiments"
        else:
            raw = data_dir / "pill_raw"
            aug = data_dir / "pill_aug"
            anod = data_dir / "pill_another"
            run_dir = w / "runs"
            exp_dir = w / "experiments"

    def _s(x):
        return str(x) if x is not None else None

    return {
        # ---- 입력 ----
        "PROJECT_ROOT": PROJECT_ROOT,
        "PILL_ROOT": PILL_ROOT,                       # = config.yaml 의 dataset_root
        "DATASET_ROOT": PILL_ROOT,                    # 별칭 (config.yaml 용어)
        "SRC_DIR": _s(Path(PROJECT_ROOT) / "src") if PROJECT_ROOT else None,
        "IMAGE_DIR": _s(Path(PILL_ROOT) / "train_images") if PILL_ROOT else None,
        "ANN_DIR": _s(Path(PILL_ROOT) / "train_annotations") if PILL_ROOT else None,
        "TEST_DIR": _s(Path(PILL_ROOT) / "test_images") if PILL_ROOT else None,
        "ANOTHER_ROOT": ANOTHER_ROOT,
        "CROPPED_PILLS_DIR": CROPPED_PILLS_DIR,
        "TEAM_WORK_DIR": TEAM_WORK_DIR,
        # ---- 산출물 ----
        "WORK_ROOT": WORK_ROOT,
        "LAYOUT": LAYOUT,
        "RAW_DIR": _s(raw),                           # 증강 전 YOLO 데이터셋
        "PROCESSED_DIR": _s(raw),                     # 별칭 (팀 노트북 용어)
        "ANOTHER_DIR": _s(anod),
        "AUG_DIR": _s(aug),                           # ★ 학습에 쓰는 증강 데이터셋
        "OUT_ROOT": _s(out_dir),
        "FIG_DIR": _s(out_dir / "figures") if out_dir else None,
        "PRED_DIR": _s(out_dir / "predictions") if out_dir else None,
        "SUB_DIR": _s(out_dir / "submissions") if out_dir else None,
        "CKPT_DIR": _s(out_dir / "checkpoints") if out_dir else None,
        "RUN_DIR": _s(run_dir),
        "EXP_DIR": _s(exp_dir),
        "RESULT_CSV": _s(exp_dir / "result.csv") if exp_dir else None,
        "CUTOUT_CHECK_DIR": CUTOUT_CHECK_DIR,
    }


def setup(
    pill_root: Optional[PathLike] = None,
    work_root: Optional[PathLike] = None,
    *,
    project_root: Optional[PathLike] = None,
    config: Optional[PathLike] = None,
    zip_path: Optional[PathLike] = None,
    mount: bool = True,
    unzip: bool = True,
    install: bool = True,
    make_dirs: bool = True,
    verbose: bool = True,
) -> Dict[str, Optional[str]]:
    """★ 노트북 첫 셀에서 부르는 단 하나의 준비 함수.

    하는 일
      1. (Colab) 구글 드라이브 마운트
      2. 저장소 루트(config.yaml) 탐색 → 있으면 그 규약을 따릅니다
      3. 원본 데이터 루트(train_images 의 상위) 탐색
         · 못 찾고 Colab 이면 드라이브의 zip 을 /content 로 풀어 봅니다
      4. 산출물 폴더 생성
      5. (Colab) ultralytics·한글 폰트 설치
      6. PillDetectionDataset.py 를 sys.path 에 등록

    Args:
        pill_root: 원본 데이터 루트를 직접 지정 (train_images 의 상위 폴더).
            config.yaml 의 `paths.dataset_root` 와 같은 값입니다.
        work_root: 산출물 루트를 직접 지정.
            · None + project 배치 → PROJECT_ROOT (저장소 안에 그대로 떨어짐)
            · None + Colab       → /content/PillWork (로컬 디스크, ★ 빠름)
              ⚠️ 세션이 끊기면 사라집니다 → pt.save_outputs_to_drive()
            · None + 로컬        → pill_root 의 부모 폴더
        project_root: 저장소 루트를 직접 지정 (None = 자동 탐색)
        config: config.yaml 경로 (None = PROJECT_ROOT/config.yaml 자동)
        zip_path: pilldata zip 경로 (None = 드라이브에서 자동 검색)

    Returns:
        paths() 와 같은 dict.

            P = pt.setup()
            RAW_DIR, AUG_DIR = P["RAW_DIR"], P["AUG_DIR"]
    """
    g = globals()
    colab = in_colab()
    if verbose:
        print("═" * 62)
        print(f"  실행 환경   {'Google Colab' if colab else '로컬'}")

    # ---------- 1. 드라이브 ----------
    if colab and mount:
        try:
            mount_drive(verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"⚠️ 드라이브 마운트 실패({e}) — 계속 진행합니다.")

    # ---------- 2. 저장소 루트 + config.yaml ----------
    proj = str(Path(project_root).resolve()) if project_root else detect_project_root()
    g["PROJECT_ROOT"] = proj
    cfg = load_config(config, verbose=verbose) if (proj or config) else {}
    g["CONFIG"] = cfg

    # ---------- 3. 데이터 루트 ----------
    root = None
    if pill_root is not None:
        root = str(Path(pill_root).resolve())
        if not _looks_like_pilldata(root):
            raise FileNotFoundError(
                f"{root} 안에 train_images / train_annotations 가 없습니다.\n"
                "→ 원본 데이터 폴더(train_images 의 상위)를 지정하세요."
            )
    else:
        # config.yaml 의 dataset_root 를 최우선으로
        cfg_root = (cfg.get("paths") or {}).get("dataset_root")
        if cfg_root and _looks_like_pilldata(cfg_root):
            root = str(Path(cfg_root).resolve())
        else:
            root = detect_pill_root()
        if root is None and colab and unzip:
            if verbose:
                print("  데이터를 찾지 못했습니다 — zip 을 찾아 풀어 봅니다.")
            unz = unzip_pilldata(zip_path, dest="/content", verbose=verbose)
            if unz and _looks_like_pilldata(unz):
                root = str(Path(unz).resolve())
            else:
                root = detect_pill_root()

    if root is None:
        raise FileNotFoundError(
            "원본 데이터 폴더(train_images / train_annotations)를 찾지 못했습니다.\n"
            "→ pt.setup(pill_root='...') 로 직접 지정하거나,\n"
            "  pt.unzip_pilldata('/content/drive/MyDrive/pilldata.zip') 을 먼저 실행하세요."
        )
    g["PILL_ROOT"] = root

    # ---------- 4. 배치 판정 ----------
    #   저장소 루트가 있고 데이터가 그 안에 있으면 "project" 배치입니다.
    layout = "pilldata"
    if proj:
        try:
            Path(root).relative_to(Path(proj))
            layout = "project"
        except ValueError:
            layout = "project" if (Path(proj) / "config.yaml").is_file() else "pilldata"
    g["LAYOUT"] = layout

    # ---------- 5. 산출물 루트 ----------
    if work_root is None:
        if layout == "project" and proj:
            work_root = proj
        elif colab:
            work_root = "/content/PillWork"
        else:
            work_root = str(Path(root).parent)
    g["WORK_ROOT"] = str(Path(work_root).resolve())

    # ---------- 6. 데이터 하위 경로 ----------
    #   train_another / cropped_pills_review / team_work 는 배치마다 위치가 달라
    #   데이터 루트와 저장소 data/ 폴더를 모두 뒤집니다.
    search_bases = [Path(root)]
    if proj:
        search_bases += [Path(proj) / "data", Path(proj)]
    search_bases += [Path(root).parent]

    def _find(*names: str) -> Optional[str]:
        for base in search_bases:
            for n in names:
                p = base / n
                if p.is_dir():
                    return str(p)
        return None

    g["ANOTHER_ROOT"] = _find("train_another")
    g["CROPPED_PILLS_DIR"] = _find("cropped_pills_review", "cropped_pills")
    g["TEAM_WORK_DIR"] = _find("team_work")
    g["CUTOUT_CHECK_DIR"] = str(
        Path(g["WORK_ROOT"]) / ("outputs/cutcheck" if layout == "project" else "cutcheck")
    )

    P = paths()

    # ---------- 7. 폴더 생성 ----------
    if make_dirs:
        for k in ("RAW_DIR", "ANOTHER_DIR", "AUG_DIR", "OUT_ROOT", "FIG_DIR",
                  "PRED_DIR", "SUB_DIR", "CKPT_DIR", "RUN_DIR", "EXP_DIR",
                  "CUTOUT_CHECK_DIR"):
            if P.get(k):
                Path(P[k]).mkdir(parents=True, exist_ok=True)

    # ---------- 8. 패키지 ----------
    if install and colab:
        install_colab_deps(verbose=verbose)

    # ---------- 9. Dataset 모듈 ----------
    add_src_to_path(verbose=verbose)

    if verbose:
        print(f"  폴더 배치   {layout}"
              + ("  (config.yaml 규약을 따릅니다)" if layout == "project" else ""))
        if proj:
            print(f"  저장소      {proj}")
        print(f"  데이터      {PILL_ROOT}")
        print(f"  산출물      {WORK_ROOT}")
        if colab and str(WORK_ROOT).startswith("/content/") \
                and not str(WORK_ROOT).startswith("/content/drive"):
            print("     ⚠️ 로컬 디스크입니다 — 세션이 끊기면 사라집니다.")
            print("        결과 보관: pt.save_outputs_to_drive()")
        print("─" * 62)
        check_paths()
        print("═" * 62)
    return P


def save_outputs_to_drive(
    dst: PathLike = "/content/drive/MyDrive/pill_project/outputs",
    include: Sequence[str] = ("experiments", "outputs", "runs"),
    weights_only: bool = True,
    verbose: bool = True,
) -> Optional[str]:
    """★ Colab 세션이 끝나기 전에 결과만 드라이브로 복사합니다.

    `weights_only=True` 면 runs 폴더에서 `*.pt` 와 png/csv 만 가져옵니다
    (학습 중간 이미지까지 옮기면 드라이브가 금방 찹니다).
    """
    if not WORK_ROOT:
        print("⚠️ 먼저 pt.setup() 을 호출하세요.")
        return None
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for sub in include:
        src = Path(WORK_ROOT) / sub
        if not src.is_dir():
            continue
        for p in src.rglob("*"):
            if not p.is_file():
                continue
            if weights_only and sub == "runs" \
                    and p.suffix.lower() not in (".pt", ".png", ".csv", ".yaml", ".jpg"):
                continue
            rel = p.relative_to(WORK_ROOT)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(p, out)
                n += 1
            except Exception:
                pass
    if verbose:
        print(f"{n:,}개 파일을 복사했습니다 → {dst}")
    return str(dst)


def _autodetect_paths_on_import(verbose: bool = False) -> None:
    """import 시점에 가벼운 경로 탐색을 합니다 (setup() 을 안 불러도 동작하도록).

    실패해도 조용히 넘어갑니다 — 그 경우 pt.setup() 이 다시 시도합니다.
    """
    g = globals()
    try:
        if g.get("PROJECT_ROOT") is None:
            g["PROJECT_ROOT"] = detect_project_root()
        if g.get("PILL_ROOT") is None:
            root = detect_pill_root()
            if root:
                g["PILL_ROOT"] = root
        root = g.get("PILL_ROOT")
        if not root:
            return
        _bases = [Path(root)]
        if g.get("PROJECT_ROOT"):
            _bases += [Path(g["PROJECT_ROOT"]) / "data", Path(g["PROJECT_ROOT"])]
        _bases += [Path(root).parent]

        def _f(*names):
            for b in _bases:
                for n in names:
                    q = b / n
                    if q.is_dir():
                        return str(q)
            return None

        if g.get("ANOTHER_ROOT") is None:
            g["ANOTHER_ROOT"] = _f("train_another")
        if g.get("CROPPED_PILLS_DIR") is None:
            g["CROPPED_PILLS_DIR"] = _f("cropped_pills_review", "cropped_pills")
        if g.get("TEAM_WORK_DIR") is None:
            g["TEAM_WORK_DIR"] = _f("team_work")
        proj = g.get("PROJECT_ROOT")
        if g.get("LAYOUT") in (None, "unknown"):
            inside = False
            if proj:
                try:
                    Path(root).relative_to(Path(proj)); inside = True
                except ValueError:
                    inside = False
            g["LAYOUT"] = "project" if inside else "pilldata"
        if g.get("WORK_ROOT") is None:
            if g["LAYOUT"] == "project" and proj:
                g["WORK_ROOT"] = proj
            else:
                g["WORK_ROOT"] = "/content/PillWork" if in_colab() else str(Path(root).parent)
        if g.get("CUTOUT_CHECK_DIR") is None:
            g["CUTOUT_CHECK_DIR"] = str(Path(g["WORK_ROOT"]) / (
                "outputs/cutcheck" if g["LAYOUT"] == "project" else "cutcheck"))
    except Exception as e:  # pragma: no cover
        if verbose:
            print(f"경로 자동 탐색 실패({e}) — pt.setup() 을 호출하세요.")


_autodetect_paths_on_import()


# ═══════════════════════════════════════════════════════════════════════════
#  Part 10. ★★★ 설정 잠금 — 노트북이 이 파일의 값을 덮어쓰지 못하게 막습니다
# ═══════════════════════════════════════════════════════════════════════════
#
#  왜 필요한가
#    02 가 `pt.CLAHE_CLIP = 6.0` 처럼 자기 값을 밀어 넣으면, 이 파일에 적힌
#    5.0 은 아무 의미가 없어집니다. 팀원마다 노트북 설정이 달라지면
#    "같은 pill_transforms.py 를 쓰는데 결과가 다른" 상황이 생깁니다.
#
#  어떻게 막는가
#    모듈 객체의 __setattr__ 을 바꿔서, 아래 LOCKED_PARAMS 에 든 이름을
#    **다른 값으로** 대입하면 AttributeError 를 냅니다.
#    (같은 값 대입은 통과시킵니다 — 03·04·05 가 augment_info.json 으로
#     설정을 '복원'하는 코드가 그대로 돌아가야 하기 때문입니다.
#     값이 다르면 그 자리에서 에러가 나므로 불일치를 즉시 알 수 있습니다.)
#
#  값을 바꾸려면
#    이 파일을 직접 고치고 커널을 재시작하세요. 그게 유일한 방법입니다.
#
#  잠금을 잠깐 풀어야 한다면 (권장하지 않습니다)
#    이 파일의 ALLOW_RUNTIME_OVERRIDE 를 True 로 고치세요.
# ---------------------------------------------------------------------------

ALLOW_RUNTIME_OVERRIDE = False      # ★ True 로 바꾸면 잠금이 풀립니다 (비권장)

LOCKED_PARAMS = frozenset({
    # 학습 예산 · 해상도
    "DEFAULT_MODEL", "DEFAULT_EPOCHS", "DEFAULT_PATIENCE",
    "DEFAULT_SMOKE_EPOCHS", "DEFAULT_SMOKE_PATIENCE",
    "DEFAULT_IMGSZ", "DEFAULT_BATCH", "DEFAULT_WORKERS",
    "DEFAULT_AMP", "DEFAULT_DETERMINISTIC", "SEED",
    # 추론 · 평가
    "DEFAULT_CONF", "DEFAULT_IOU_NMS", "DEFAULT_MAX_DET", "DEFAULT_EVAL_CONF",
    # 온라인 증강
    "ONLINE_AUG_PRESETS", "DEFAULT_ONLINE_AUG",
    # 오프라인 증강량
    "DEFAULT_GEOM_MULT", "DEFAULT_N_SYNTH", "DEFAULT_CP_WEIGHTED",
    "ANOTHER_ONTO_N",
    # 전처리
    "USE_WHITE_BALANCE", "USE_CLAHE", "CLAHE_CLIP", "CLAHE_GRID",
    # 기하 증강
    "ROT_LIMIT", "ROT_PROB", "SCALE_RANGE", "MIN_VISIBILITY", "USE_FLIP",
    "P_BLUR", "P_NOISE", "P_TONE",
    "HSV_H_LIMIT", "HSV_S_LIMIT", "HSV_V_LIMIT",
    # Copy & Paste
    "PILLS_RANGE", "CP_EXTRA_RANGE", "CP_MODE", "CP_SYNTH_RATIO",
    "CP_OVERLAP", "CP_FEATHER", "MAX_CROPS_PER_CLASS",
    "UNIQUE_CLASS_PER_IMAGE",
    # 합성 배경 (BG_HSV 는 실측값이라 잠그지 않습니다)
    "BG_MODE", "BG_JITTER", "BG_STRICT_UNIFORM", "BG_GRAD", "BG_NOISE",
    # 컷아웃
    "CUT_CHROMA_K", "CUT_MIN_CHROMA", "CUT_L_K", "CUT_SHRINK",
})


def _same_value(a: Any, b: Any) -> bool:
    """같은 값 대입인지 판정 (리스트/튜플, float 오차 허용)."""
    if a is b:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same_value(a[k], b[k]) for k in a)
    try:
        return bool(a == b)
    except Exception:
        return False


def assert_config_matches(info: Optional[Dict[str, Any]], verbose: bool = True) -> bool:
    """augment_info.json / category_map.json 의 설정이 이 파일과 같은지 확인.

    03·04·05 가 `pt.CLAHE_CLIP = ...` 로 값을 되돌려 넣는 대신 이 함수를 부르면
    됩니다. 데이터를 만든 시점의 설정과 지금 파일의 설정이 다르면 경고합니다
    (= 02 를 다시 돌려야 한다는 뜻입니다).
    """
    pp = ((info or {}).get("preprocess") or {})
    checks = [
        ("white_balance", pp.get("white_balance"), USE_WHITE_BALANCE),
        ("clahe", pp.get("clahe"), USE_CLAHE),
        ("clahe_clip", pp.get("clahe_clip"), CLAHE_CLIP),
        ("clahe_grid", pp.get("clahe_grid"), CLAHE_GRID),
        ("geom_mult", (info or {}).get("geom_mult"), DEFAULT_GEOM_MULT),
        ("n_synth", (info or {}).get("n_synth"), DEFAULT_N_SYNTH),
    ]
    bad = [(k, was, now) for k, was, now in checks
           if was is not None and not _same_value(was, now)]
    if verbose:
        if bad:
            print("⚠️ 데이터를 만든 설정과 지금 pill_transforms.py 가 다릅니다:")
            for k, was, now in bad:
                print(f"     {k:<14} 데이터 {was}  ≠  현재 {now}")
            print("   → 02_baseline 을 다시 실행해 데이터를 새로 만드세요.")
        else:
            print("✅ 데이터 생성 설정 = 현재 pill_transforms.py 설정 (일치)")
    return not bad


def _install_config_lock() -> None:
    """모듈 전역 대입을 가로채 잠긴 이름을 보호합니다."""
    import sys as _sys
    import types as _types
    from types import MappingProxyType as _MPT

    mod = _sys.modules[__name__]

    # 프리셋 dict 는 읽기 전용 뷰로 바꿔 내용 변경까지 막습니다
    try:
        presets = mod.__dict__["ONLINE_AUG_PRESETS"]
        if not isinstance(presets, _MPT):
            mod.__dict__["ONLINE_AUG_PRESETS"] = _MPT(
                {k: _MPT(dict(v)) for k, v in presets.items()}
            )
    except Exception:
        pass

    class _LockedModule(_types.ModuleType):
        def __setattr__(self, name, value):
            if (name in LOCKED_PARAMS
                    and not self.__dict__.get("ALLOW_RUNTIME_OVERRIDE", False)
                    and name in self.__dict__
                    and not _same_value(self.__dict__[name], value)):
                raise AttributeError(
                    f"\n"
                    f"══════════════════════════════════════════════════════════\n"
                    f" pill_transforms.{name} 은 잠겨 있습니다.\n"
                    f"   현재 값 : {self.__dict__[name]!r}\n"
                    f"   시도한 값: {value!r}\n"
                    f"\n"
                    f" 학습 하이퍼파라미터·증강·전처리 값은 pill_transforms.py 가\n"
                    f" 단일 출처입니다. 노트북에서 덮어쓰면 팀원마다 다른 설정으로\n"
                    f" 학습하게 되어 실험 비교가 무의미해집니다.\n"
                    f"\n"
                    f" → 바꾸려면 pill_transforms.py 의 {name} 을 직접 고치고\n"
                    f"   커널을 재시작하세요.\n"
                    f"══════════════════════════════════════════════════════════"
                )
            super().__setattr__(name, value)

    mod.__class__ = _LockedModule


_install_config_lock()


if __name__ == "__main__":
    selftest()
    print("\n■ 모든 테스트 통과")