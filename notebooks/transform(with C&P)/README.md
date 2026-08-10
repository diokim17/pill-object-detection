# pill_transforms.py 사용 가이드

경구약제(알약) Object Detection 프로젝트용 전처리 / 증강 통합 모듈입니다.
**Google Colab, Anaconda(Jupyter Notebook/Lab) 어디서든 동일하게** 사용할 수 있습니다.

---

## 1. 설치 & 불러오기

### 1-1. Google Colab

```python
# ① 구글 드라이브 마운트 (crops_dir 등 드라이브 경로를 쓸 경우)
from google.colab import drive
drive.mount('/content/drive')

# ② pill_transforms.py 를 노트북과 같은 위치(또는 드라이브)에 두고 경로 추가
import sys
sys.path.append('/content/drive/MyDrive/pill_project')   # pill_transforms.py 가 있는 폴더

import pill_transforms as pt
```

> Colab 셀 맨 위에서 파일을 직접 업로드해도 됩니다:
> ```python
> from google.colab import files
> files.upload()  # pill_transforms.py 선택
> ```

### 1-2. Anaconda (Jupyter Notebook/Lab)

```bash
# 가상환경 활성화 후 필요한 패키지 설치
conda activate <내 환경>
pip install opencv-python numpy albumentations pyyaml
```

```python
# 노트북 코드 셀
import sys, os
sys.path.append(os.path.abspath("."))   # pill_transforms.py 가 있는 폴더

import pill_transforms as pt
```

> `pill_transforms.py` 는 `baseline.ipynb` 와 **같은 폴더**에 두는 것이 가장 간단합니다.
> 이 경우 `sys.path.append` 없이 바로 `import pill_transforms as pt` 만 해도 됩니다.

### 1-3. 필요 패키지

| 패키지 | 필수 여부 | 용도 |
|---|---|---|
| `opencv-python` | 필수 | 이미지 I/O, 색공간 변환 |
| `numpy` | 필수 | 배열 연산 |
| `albumentations` | 선택 (A 갈래) | Faster R-CNN 온라인 transform |
| `pyyaml` | 선택 | data.yaml 읽기 (Copy&Paste 클래스명 매칭 시) |

`pip install opencv-python numpy albumentations pyyaml` 한 줄이면 전부 해결됩니다.

---

## 2. 두 갈래 사용법

| 노트북 | 이 모듈에서 쓰는 것 |
|---|---|
| `pill_detection_dataset.ipynb` | `get_train_transforms()` |
| `base_model_faster-rcnn_train.ipynb` | `get_train_transform()`, `get_eval_transform()` |
| `yolo11s_baseline.ipynb` / `baseline.ipynb` | `build_augmented_yolo_dataset()` |

```python
# [A] Faster R-CNN 계열 — 매 배치 온라인 증강
from pill_transforms import get_train_transform, get_eval_transform
train_transforms = get_train_transform(image_size=640)
eval_transforms  = get_eval_transform(image_size=640)

# [B] YOLO 계열 — 디스크에 증강 데이터셋을 미리 생성
from pill_transforms import build_augmented_yolo_dataset
aug_yaml = build_augmented_yolo_dataset(
    src_root="../data/processed",
    dst_root="../data/processed_aug",
    geom_mult=3,
    n_synth=600,
)
```

---

## 3. `cropped_pills_review` 크롭 이미지로 Copy&Paste 재료 쓰기

이미 잘라 둔 알약 이미지 폴더(`cropped_pills_review/클래스별 하위 폴더/*.png`)를
Copy&Paste 합성 재료로 그대로 사용할 수 있습니다.

```python
import pill_transforms as pt

# 방법 1) 전역으로 한 번만 지정
pt.CROPPED_PILLS_DIR = "/content/drive/MyDrive/.../cropped_pills_review"

aug_yaml = pt.build_augmented_yolo_dataset(
    src_root="../data/processed",
    geom_mult=3, n_synth=600,
)

# 방법 2) 함수 호출 때마다 지정
aug_yaml = pt.build_augmented_yolo_dataset(
    src_root="../data/processed",
    geom_mult=3, n_synth=600,
    crops_dir="/content/drive/MyDrive/.../cropped_pills_review",
)
```

- 폴더명(`3351_일양하이트린정 2mg`)은 `data.yaml`의 클래스명과 자동으로 매칭됩니다
  (공백·기호 무시, 코드번호 단독 매칭도 지원).
- 매칭 실패한 폴더는 실행 로그에 `⚠️ 클래스명 매칭 실패 폴더 N개: ...` 로 표시됩니다.
- `crops_dir` 을 지정하지 않으면(=`None`) 기존 동작(train 라벨 박스에서 직접 컷아웃)으로 자동 복귀합니다.

---

## 4. 설정값(CONFIG) 전부 `baseline.ipynb`에서 바꾸기

`pill_transforms.py` 상단 **Part 0. 기본 설정** 블록에 있는 모든 값은
파일을 직접 수정하지 않고 **노트북에서 `import` 한 뒤 덮어쓰는 것만으로 적용**됩니다.

```python
import pill_transforms as pt

pt.SEED = 42

# ── 오프라인 증강 매수 ──
pt.DEFAULT_GEOM_MULT = 3      # train 1장 → 최종 3장
pt.DEFAULT_N_SYNTH   = 600    # Copy&Paste 합성 이미지 수

# ── 전처리 ──
pt.USE_WHITE_BALANCE = True
pt.USE_CLAHE   = True
pt.CLAHE_CLIP  = 3.0
pt.CLAHE_GRID  = 8

# ── 기하 증강 ──
pt.ROT_LIMIT       = 180
pt.ROT_PROB        = 0.8
pt.SCALE_RANGE     = (0.95, 1.05)
pt.MIN_VISIBILITY  = 0.2
pt.USE_FLIP        = False

pt.P_BLUR  = 0.20
pt.P_NOISE = 0.30
pt.P_TONE  = 0.30
pt.HSV_H_LIMIT = 8
pt.HSV_S_LIMIT = 25
pt.HSV_V_LIMIT = 30

# ── Copy & Paste ──
pt.PILLS_RANGE       = (2, 4)
pt.CP_EXTRA_RANGE    = (1, 2)
pt.CP_MODE           = "mix"       # "synth" | "onto_train" | "mix"
pt.CP_SYNTH_RATIO    = 0.5
pt.CP_OVERLAP        = 0.10
pt.CP_SCALE_JIT      = (0.95, 1.05)
pt.CP_FEATHER        = 2
pt.SHADOW_ALPHA      = (0.20, 0.55)
pt.SHADOW_BLUR       = (3, 12)
pt.MAX_CROPS_PER_CLASS = 40

# ↓ 값을 다 바꾼 뒤 build_augmented_yolo_dataset() 를 호출하면
#   위에서 바꾼 값들이 그대로 반영됩니다.
aug_yaml = pt.build_augmented_yolo_dataset(
    src_root="../data/processed",
    dst_root="../data/processed_aug",
)
```

### 4-1. 딱 이번 실행만 바꾸고 싶다면 → 함수 인자로

전역을 건드리지 않고 **이번 호출 한 번만** 바꾸고 싶은 값들은
`build_augmented_yolo_dataset()` 의 키워드 인자로 바로 넘길 수 있습니다
(내부적으로 위 전역값들의 기본값을 그대로 씁니다).

```python
aug_yaml = pt.build_augmented_yolo_dataset(
    src_root="../data/processed",
    geom_mult=5,              # ← pt.DEFAULT_GEOM_MULT 대신 이번만 5
    n_synth=1000,              # ← pt.DEFAULT_N_SYNTH 대신 이번만 1000
    use_flip=True,             # ← pt.USE_FLIP 대신 이번만 True
    cp_mode="onto_train",      # ← pt.CP_MODE 대신 이번만
    max_crops_per_class=60,    # ← pt.MAX_CROPS_PER_CLASS 대신 이번만
    seed=0,                    # ← pt.SEED 대신 이번만
)
```

`ROT_LIMIT`, `SCALE_RANGE`, `HSV_*`, `CP_SCALE_JIT` 등 나머지 세부 값들은
함수 인자로는 노출돼 있지 않으므로, **바꾸고 싶으면 `pt.변수명 = 값` 으로
전역을 먼저 바꾼 뒤 함수를 호출**하세요 (위 4번 예시 방식).

### 4-2. Faster R-CNN(`get_train_transform` 등) 경로에서 전처리 값을 바꿀 때

`build_augmented_yolo_dataset()` 은 호출될 때마다 현재 전역값으로
전처리를 새로 구성하므로 바로 반영됩니다. 반면 `get_train_transform()` /
`get_eval_transform()` 이 쓰는 **추론용 전역 싱글턴 `PREPROCESS`** 는
`USE_WHITE_BALANCE` / `USE_CLAHE` / `CLAHE_CLIP` / `CLAHE_GRID` 를 바꾼 뒤
아래처럼 한 번 재생성해줘야 합니다.

```python
pt.CLAHE_CLIP = 3.0
pt.rebuild_preprocess()   # ★ 이 줄을 꼭 호출
```

### 4-3. 값 하나하나가 뭘 뜻하는지

| 그룹 | 변수 | 의미 |
|---|---|---|
| 시드 | `SEED` | 재현성용 랜덤 시드 |
| 증강 매수 | `DEFAULT_GEOM_MULT` | train 1장당 최종 장수 (3=원본1+증강2) |
| | `DEFAULT_N_SYNTH` | Copy&Paste 합성 이미지 총 장수 |
| 전처리 | `USE_WHITE_BALANCE` | Shades-of-Gray 화이트밸런스 on/off |
| | `USE_CLAHE` | Lab L채널 CLAHE on/off (항상 켜기 권장) |
| | `CLAHE_CLIP` | CLAHE 대비 제한 강도 (클수록 대비 강해짐) |
| | `CLAHE_GRID` | CLAHE 타일 격자 크기 |
| 기하 증강 | `ROT_LIMIT` | 회전 각도 한계(도). 알약은 방향이 무의미해 크게 잡음 |
| | `ROT_PROB` | 회전+스케일 변환이 적용될 확률 |
| | `SCALE_RANGE` | 확대/축소 비율 범위 `(최소, 최대)` |
| | `MIN_VISIBILITY` | 잘린 뒤 남는 면적 비율이 이보다 작으면 박스 삭제 |
| | `USE_FLIP` | 좌우/상하 반전 on/off (각인 뒤집힘 때문에 기본 off) |
| 색상/노이즈 | `P_BLUR` | 모션 블러 적용 확률 |
| | `P_NOISE` | ISO 노이즈 적용 확률 |
| | `P_TONE` | 톤커브(대비 곡선) 적용 확률 |
| | `HSV_H_LIMIT` | 색상(Hue) 흔들림 한계 — 색이 클래스 정보라 작게 |
| | `HSV_S_LIMIT` | 채도(Saturation) 흔들림 한계 |
| | `HSV_V_LIMIT` | 명도(Value) 흔들림 한계 |
| Copy&Paste | `PILLS_RANGE` | 합성 이미지 1장에 붙일 알약 개수 범위 |
| | `CP_EXTRA_RANGE` | `onto_train` 모드에서 추가로 붙일 알약 개수 범위 |
| | `CP_MODE` | `"synth"`(빈 배경) \| `"onto_train"`(원본 위에 추가) \| `"mix"`(섞음) |
| | `CP_SYNTH_RATIO` | `mix` 모드일 때 인공 배경 비율 |
| | `CP_OVERLAP` | 알약끼리 허용하는 겹침 비율 |
| | `CP_SCALE_JIT` | 붙여넣을 때 크기를 얼마나 흔들지 `(최소배율, 최대배율)` |
| | `CP_FEATHER` | 붙여넣기 경계 페더링(부드럽게) 픽셀 수 |
| | `SHADOW_ALPHA` | 합성 그림자 진하기 범위 |
| | `SHADOW_BLUR` | 합성 그림자 블러 커널 크기 범위 |
| | `MAX_CROPS_PER_CLASS` | 크롭 라이브러리에 클래스당 보관할 최대 장수 |

---

## 5. `CP_SCALE_JIT` (크기 흔들기)를 완전히 끄고 싶을 때

`CP_SCALE_JIT` 은 Copy&Paste 로 알약을 붙일 때 크기를 무작위로 살짝
키우거나 줄이는 범위 `(최소배율, 최대배율)` 입니다. 예: `(0.9, 1.10)` →
원래 크기의 90%~110% 사이에서 무작위 결정.

**완전히 끄려면 최소·최대를 같은 값(1.0)으로 맞추면 됩니다.**
`random.uniform(1.0, 1.0)` 은 항상 `1.0` 을 반환하므로, 크기 변화 없이
원본 GT 박스 통계 기준 크기 그대로만 붙습니다.

```python
import pill_transforms as pt
pt.CP_SCALE_JIT = (1.0, 1.0)   # ★ 크기 흔들기 완전 비활성화

aug_yaml = pt.build_augmented_yolo_dataset(
    src_root="../data/processed",
    geom_mult=3, n_synth=600,
)
```

> 참고: `0` 이나 빈 튜플을 넣지 마세요. `(0.0, 0.0)` 을 넣으면 크기가 0이 되어
> 이미지가 사라집니다. 반드시 `(1.0, 1.0)` 으로 "고정"하는 방식입니다.

---

## 6. 자주 겪는 문제

| 증상 | 원인 / 해결 |
|---|---|
| `⚠️ 컷아웃에 전부 실패해 Copy&Paste 를 건너뜁니다` | `CUT_MIN_CHROMA`를 5.0, `CUT_CHROMA_K`를 1.6으로 낮춰보세요. `crops_dir`을 쓰는 경우 경로/폴더명 매칭을 확인하세요. |
| `⚠️ 클래스명 매칭 실패 폴더 N개` | `cropped_pills_review`의 폴더명이 `data.yaml`의 클래스명과 다릅니다. 폴더명 앞 코드번호(예: `3351_`)나 뒤 이름 중 하나라도 `data.yaml` 클래스명에 포함되면 자동 매칭됩니다. |
| 값을 바꿨는데 반영이 안 됨 | `pt.변수명 = 값` 을 **함수 호출 전에** 실행했는지 확인하세요. Faster R-CNN 전처리(`PREPROCESS`)를 바꾼 경우 `pt.rebuild_preprocess()` 호출이 필요합니다. |
| 한글 경로에서 이미지가 안 읽힘 | 이 모듈은 `imread_unicode` / `imwrite_unicode` 를 내부적으로 사용해 한글 경로를 지원합니다. 직접 `cv2.imread`를 쓰지 말고 `pt.imread_unicode()`를 사용하세요. |
| Colab에서 파일이 안 보임 | 드라이브 마운트(`drive.mount`) 여부와 `sys.path.append` 경로를 확인하세요. |

---

## 7. 결과 확인

```python
# 생성된 증강 데이터셋을 눈으로 검수
from pill_transforms import preview_augmented
preview_augmented(dst_root="../data/processed_aug", per_kind=3)
```

- `images/train`에는 `orig_*`(원본), `aug_*`(기하증강), `cp_*`(Copy&Paste) 3종류가 섞여 저장됩니다.
- `augment_info.json`에 이번 실행에 사용된 설정값이 전부 기록됩니다.
