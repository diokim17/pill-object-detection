# pill_transforms.py 팀 사용 가이드

경구약제(알약) Object Detection 프로젝트의 **전처리 · 증강 공용 모듈**입니다.
노트북 4개가 이 파일 하나만 import 하면 되고, Colab / 로컬 / Windows 어디서든
같은 결과가 나오도록 만들었습니다.

---

## 0. 공유 파일 역할

| 파일 | 역할 |
|---|---|
| `src/pill_transforms.py` | 전처리 · 증강 · Copy&Paste · 사전검증의 **유일한 기준** |
| `README.md` | 이 문서. 노트북별 실행 방법 |
| `data/processed/` | `pill_detection_dataset.ipynb` 가 만든 원본 YOLO 데이터셋 (입력) |
| `data/processed_aug/` | 이 모듈이 만드는 증강 데이터셋 (출력) |
| `data/cropped_pills_review/` | Copy&Paste 재료 (클래스별 crop 폴더, 없어도 동작) |

### 핵심 원칙

- **`pill_transforms.py` 가 source of truth 입니다.** 노트북 안에서 전처리 코드를 다시 정의하지 않습니다.
- **먼저 `preflight`, 성공한 뒤 실제 실행합니다.** (AI Hub 팀 공유 가이드와 동일한 2단계)
- **전처리는 train / val / test / 추론에 전부 동일하게 겁니다.** 학습에만 걸면 분포가 어긋납니다.
- **배경색은 고정 상수입니다.** 무작위로 만들지 않습니다 (아래 3장).
- 원본 `data/processed`, crop 원본, 체크포인트는 이 모듈이 **수정하거나 삭제하지 않습니다.**
- 경로를 하드코딩하지 않습니다. `pt.configure()` 가 실행 환경을 감지합니다.

---

## 1. 시작 전 준비

### 파일 배치

```
pill-object-detection/
├── src/
│   ├── pill_transforms.py          ← 이 파일을 여기에 둡니다
│   └── PillDetectionDataset.py
├── notebooks/
│   ├── pill_detection_dataset.ipynb
│   ├── yolo11s_baseline_cp.ipynb
│   ├── base_model_faster-rcnn_train.ipynb
│   └── base_model_faster-rcnn_predict.ipynb
├── data/
│   ├── dataset/cleaning_data/...   원본 이미지 + annotation
│   ├── processed/                  YOLO 데이터셋 (images/labels/data.yaml)
│   ├── processed_aug/              ← 자동 생성
│   └── cropped_pills_review/       Copy&Paste 재료 (선택)
├── outputs/
│   ├── yolo/ · checkpoints/ · submissions/
│   └── cutcheck/                   ← 자동 생성 (컷아웃 검수)
└── config.yaml
```

### 패키지

```bash
python3 -m pip install opencv-python numpy albumentations pyyaml tqdm
```

Colab 노트북 첫 셀에서는 아래 한 줄로도 됩니다.

```python
pt.ensure_deps()      # 부족한 것만 설치합니다
```

> `albumentations` 는 Faster R-CNN 쪽(온라인 transform)에만 필요합니다.
> YOLO 오프라인 증강은 `cv2` + `numpy` 만으로 동작합니다.

---

## 2. 공통 3단계 — 모든 노트북이 동일합니다

```python
# ── 1단계. 모듈 로드 + 경로 설정 ────────────────────────────────
from google.colab import drive; drive.mount("/content/drive")     # Colab만

import sys, importlib
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/코드잇 AI 13기/AI 13기 프로젝트/pill-object-detection")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

sys.modules.pop("pill_transforms", None)      # 수정본을 다시 읽기 위해
importlib.invalidate_caches()
import pill_transforms as pt

pt.configure(project_root=PROJECT_ROOT)       # 인자 없이 pt.configure() 도 가능(자동 감지)
```

`configure()` 출력 예시

```
  DATASET_DIR        .../data/processed                ✅
  AUG_DATASET_DIR    .../data/processed_aug            — 증강 실행 시 만들어집니다
  CROPPED_PILLS_DIR  .../data/cropped_pills_review     ✅
  CUTOUT_CHECK_DIR   .../outputs/cutcheck              — 실행할 때 자동으로 만들어집니다
  배경색(고정)        #696E80  RGB(105, 110, 128) / BGR(128, 110, 105)  mode=fixed
```

```python
# ── 2단계. 사전검증 (파일을 하나도 만들지 않습니다) ──────────────
rep = pt.preflight()
rep["ok"]     # True 여야 다음 단계로 갑니다
```

**반드시 0이어야 하는 항목**

| 항목 | 의미 |
|---|---|
| 이미지 읽기 오류 | 깨진 PNG. 0이 아니면 원본을 다시 받으세요 |
| 라벨 없는 이미지 | `labels/train` 에 같은 이름의 `.txt` 가 없음 |
| bbox 오류 | 좌표가 0~1 정규화값이 아니거나 폭/높이가 0 이하 |
| 클래스 ID 오류 | `data.yaml` 의 `nc` 범위를 벗어난 클래스 번호 |

**0이 아니어도 되는 항목**: val/test 장수, 크롭 클래스 미매칭(재료만 줄어듭니다), 기존 출력 이미지.

```python
# ── 3단계. 실제 실행 ─────────────────────────────────────────
aug_yaml = pt.build_augmented_yolo_dataset(geom_mult=3, n_synth=600)
```

`build_augmented_yolo_dataset()` 는 내부에서 `preflight()` 를 한 번 더 돌리고,
실패하면 **아무것도 만들지 않고 중단**합니다. (검증을 건너뛰려면 `force=True`)

---

## 3. ★ 고정 배경색

Copy&Paste 합성 배경과 letterbox 패딩 여백은 **무작위가 아니라 실측 고정값**입니다.
팀 crop 산출물 `K-003351-003832-016232_0_2_0_2_90_000_200.png` 에서
상단 라벨바·파란 테두리를 제외하고 알약 바깥 픽셀만 모아 측정했습니다.

| 항목 | 값 |
|---|---|
| RGB | **(105, 110, 128)** |
| BGR (cv2) | (128, 110, 105) |
| HEX | **#696E80** |
| HSV (OpenCV, H 0~179) | H 113 · S 44 · V 128 |
| 채널 노이즈 std | 5.0 |
| 밝기 기울기 진폭 | ±6 (실측 V p5~p95 = 119~134) |

모듈 상단 `Part 0-2` 에 `PILL_BG_RGB` / `PILL_BG_BGR` / `PILL_BG_HEX` /
`PILL_BG_NOISE_STD` / `PILL_BG_GRAD` 로 박혀 있습니다. 바꾸려면 그 숫자만 고치면 됩니다.

내 crop 이미지와 값이 맞는지 확인:

```bash
python src/pill_transforms.py --bg-check "data/cropped_pills_review/3351_일양하이트린정 2mg/K-0033....png"
```

```
  측정 배경 RGB (105, 111, 128)
  고정 배경 RGB (105, 110, 128)  #696E80
  최대 채널 차이 1.0  ✅ 일치
```

관련 설정

| 상수 | 기본값 | 설명 |
|---|---|---|
| `CP_BG_MODE` | `"fixed"` | `"crops"` = 예전처럼 crop 배경 색조 표본 사용, `"random"` = 무작위 |
| `USE_BG_PAD` | `True` | letterbox 여백을 배경색으로 채움. `False` 면 검정 0 패딩 |
| `PILL_BG_V_JITTER` | `0.0` | 장마다 전체 밝기를 흔들고 싶을 때만 > 0 |

> 참고: 고정 배경색은 **화이트밸런스·CLAHE 전** 단계에 적용됩니다. 실제 촬영 이미지가
> 파이프라인에 들어오는 지점과 같기 때문에, 합성 이미지와 원본 이미지가 같은 전처리를 거칩니다.
> CLAHE 는 패딩 **앞**에서 적용되므로 패딩 영역이 히스토그램을 오염시키지 않습니다.

---

## 4. 노트북별 사용법

### 4-1. `pill_detection_dataset.ipynb` — 데이터셋 생성

이 노트북은 **원본 `data/processed` 를 만드는 단계**라 증강을 쓰지 않습니다.
Dataset 동작 확인용으로 transform 을 붙여 볼 때만 사용합니다.

```python
import pill_transforms as pt
pt.configure()

dataset = PillDetectionDataset(
    root=dataset_root,
    transforms=pt.get_train_transforms(image_size=640, to_tensor=False),
    label_offset=1, strict=False,
)
```

YOLO 폴더 내보내기 셀을 끝낸 뒤 **반드시** 다음을 실행해 다음 단계로 넘길 수 있는 상태인지 확인하세요.

```python
pt.preflight()      # 여기서 ok=True 가 나와야 YOLO / R-CNN 노트북이 돕니다
```

---

### 4-2. `yolo11s_baseline_cp.ipynb` — YOLO11s 학습

> **주의:** 기존 노트북은 `import pill_transforms_cp as pt` 로 되어 있습니다.
> 파일명이 `pill_transforms.py` 이므로 아래 둘 중 하나를 하세요.
> 1. 셀 2-3 의 import 를 `import pill_transforms as pt` 로 고친다 **(권장)**
> 2. 또는 `src/pill_transforms_cp.py` 에 `from pill_transforms import *` 한 줄짜리 파일을 둔다

**셀 2-3 (모듈 로드)** — 교체

```python
import sys, importlib
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
for m in ["pill_transforms", "pill_transforms_cp"]:
    sys.modules.pop(m, None)
importlib.invalidate_caches()

import pill_transforms as pt
pt.configure(project_root=PROJECT_ROOT, dataset_dir=DATASET_DIR)
print("로드된 파일:", pt.__file__)
```

**셀 3-2 (증강 데이터셋 생성)** — 교체

```python
# 1단계: 사전검증
rep = pt.preflight(geom_mult=3, n_synth=600)
assert rep["ok"], "사전검증 실패 — 위 출력의 ❌ 항목을 먼저 고치세요"

# 2단계: 실제 생성
aug_yaml = pt.build_augmented_yolo_dataset(
    geom_mult=3,        # train 1장 → 3장 (원본 1 + 증강 2)
    n_synth=600,        # Copy&Paste 합성 장수 (200 → 400 → 600 순으로 실험)
    crops_dir=None,     # None 이면 configure() 가 찾은 폴더를 사용
)
print("증강 data.yaml:", aug_yaml)

pt.preview_augmented(Path(aug_yaml).parent, per_kind=3)   # ★ 학습 전 눈으로 확인
```

학습 셀(4-2)은 그대로 두면 됩니다. `data=aug_yaml` 이고 Ultralytics 기본 증강은 전부 0으로
꺼져 있어야 합니다 — 증강은 이 모듈이 오프라인에서 이미 적용했기 때문입니다.

**검수 포인트** (`_preview` 폴더 이미지를 열어서)

| 확인 | 어긋났을 때 |
|---|---|
| 박스가 알약에 딱 맞는가 | `RotateScale` / `MIN_VISIBILITY` |
| 그림자까지 박스에 들어가지 않았는가 | `MIN_VISIBILITY` 상향 |
| 각인 글자가 읽히는가 | `P_BLUR = 0.0` |
| 붙인 알약 경계에 후광이 없는가 | `CP_FEATHER` 조절 |
| 합성 배경이 원본 배경과 같은 색인가 | `--bg-check` 로 재측정 |

---

### 4-3. `base_model_faster-rcnn_train.ipynb` — Faster R-CNN 학습

기존 셀 15에서 노트북 안에 직접 정의하던 `FasterRCNNTransform` 클래스는 **삭제**하고
모듈에서 가져옵니다. (predict 노트북과 전처리를 100% 동일하게 맞추기 위해서입니다)

**셀 6 (모듈 import)** — 교체

```python
SRC_DIR = Path("./src").resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from PillDetectionDataset import PillDetectionDataset, detection_collate_fn
import pill_transforms as pt

pt.configure(verbose=True)
```

**셀 15 (Transform 설정)** — 통째로 교체

```python
IMAGE_SIZE = 640

# baseline: 증강 없이 letterbox + CLAHE 만 (augment=True 로 켤 수 있습니다)
train_transforms = pt.get_frcnn_train_transform(image_size=IMAGE_SIZE, augment=False)
eval_transforms  = pt.get_frcnn_eval_transform(image_size=IMAGE_SIZE)

print("Train transform:", train_transforms)
print("Eval  transform:", eval_transforms)
```

셀 16 이후(Dataset · DataLoader · 학습 루프)는 수정할 필요가 없습니다.
출력은 이전과 동일하게 `torch.Tensor [C,H,W] float32 0~1` 입니다.

> 증강을 켜서 성능을 비교하려면 `augment=True` 만 바꾸면 됩니다.
> 이때 회전·크롭으로 박스가 전부 사라지는 샘플이 걱정되면
> `pt.SafeAlbumentationsTransform` 로 감싸세요.

---

### 4-4. `base_model_faster-rcnn_predict.ipynb` — 추론

추론에서 **가장 흔한 실수는 학습과 다른 전처리를 쓰거나, letterbox 좌표를 원본 좌표로
되돌리지 않는 것**입니다. 되돌리지 않으면 제출 mAP 가 0에 가깝게 나옵니다.
모듈이 두 가지를 모두 처리합니다.

```python
# ── 셀 2. 모듈 로드 ─────────────────────────────────────────
import sys, torch
from pathlib import Path

sys.path.insert(0, str(PROJECT_ROOT / "src"))
import pill_transforms as pt
pt.configure(project_root=PROJECT_ROOT)

# ── 셀 3. 체크포인트 로드 ────────────────────────────────────
ckpt = torch.load(PROJECT_ROOT / "outputs/checkpoints/faster_rcnn_best.pth",
                  map_location=device)
model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
model.to(device).eval()

# ── 셀 4. 이미지 1장 추론 ────────────────────────────────────
img_t, meta = pt.prepare_image_for_inference(img_path, image_size=640)

with torch.no_grad():
    pred = model([img_t.to(device)])[0]

# ★ 640 letterbox 좌표 → 원본 976x1280 좌표로 복원
boxes  = pt.undo_letterbox_boxes(pred["boxes"].cpu().numpy(), meta)
scores = pred["scores"].cpu().numpy()
labels = pred["labels"].cpu().numpy()          # label_offset=1 이므로 -1 하면 원래 클래스

keep = scores >= 0.05
boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
```

라벨이 있는 평가셋으로 mAP 를 계산할 때는 Dataset 에 학습과 같은 transform 을 붙입니다.

```python
eval_dataset = PillDetectionDataset(
    root=dataset_root,
    transforms=pt.get_frcnn_eval_transform(image_size=640),
    label_offset=1, strict=False,
)
```

제출 CSV 좌표는 **반드시 `undo_letterbox_boxes()` 를 거친 원본 좌표**여야 합니다.

---

## 5. 터미널 실행 (노트북 없이)

가이드와 동일하게 **검증 → 실행** 순서입니다.

```bash
# 1) 사전검증 — 파일을 만들지 않습니다 (기본 동작)
python src/pill_transforms.py --preflight

# 2) 실제 생성
python src/pill_transforms.py --execute --geom-mult 3 --n-synth 600 --preview

# 3) 경로를 직접 지정할 때
python src/pill_transforms.py --execute \
    --project-root "/content/drive/MyDrive/.../pill-object-detection" \
    --src-root  "/path/to/data/processed" \
    --dst-root  "/path/to/data/processed_aug" \
    --crops-dir "/path/to/team_work/cropped_output"

# 4) 고정 배경색 검증
python src/pill_transforms.py --bg-check "/path/to/crop.png"

# 5) 환경 점검 (더미 데이터로 전체 파이프라인 1회전, 실제 데이터는 건드리지 않음)
python src/pill_transforms.py --selftest
```

경로에 공백이나 한글이 있으면 반드시 큰따옴표로 감싸세요.

---

## 6. 자주 바꾸는 설정값

노트북에서 `pt.<이름> = 값` 으로 덮어쓴 뒤 함수를 호출하면 반영됩니다.

| 이름 | 기본값 | 설명 |
|---|---|---|
| `DEFAULT_GEOM_MULT` | 3 | train 1장 → 최종 몇 장 (1 = 증강 없음) |
| `DEFAULT_N_SYNTH` | 600 | Copy&Paste 합성 장수 (0 = 끔) |
| `CLAHE_CLIP` | 5.0 | 대비 강도. 노이즈가 뜨면 4.0 |
| `USE_FLIP` | False | 각인이 뒤집히므로 기본 off |
| `CP_MODE` | `"mix"` | `"synth"` · `"onto_train"` · `"mix"` |
| `PILLS_RANGE` | (3, 4) | 합성 1장에 붙일 알약 수 |
| `MAX_CROPS_PER_CLASS` | 40 | 크롭 라이브러리 클래스당 최대 장수 |
| `CP_BG_MODE` | `"fixed"` | 합성 배경 방식 |

```python
pt.CLAHE_CLIP = 4.0
pt.rebuild_preprocess()     # ★ 전처리 전역값을 바꿨으면 이 호출이 필요합니다
```

`build_augmented_yolo_dataset()` 는 호출 시점의 전역값을 직접 읽으므로
`rebuild_preprocess()` 없이도 반영됩니다. 필요한 것은 Faster R-CNN 경로뿐입니다.

---

## 7. 문제가 생겼을 때

| 증상 | 확인할 것 |
|---|---|
| `ModuleNotFoundError: pill_transforms` | `sys.path` 에 `src` 를 넣었는지, 파일이 `src/` 에 있는지 |
| 노트북에서 수정이 반영되지 않음 | `sys.modules.pop("pill_transforms", None)` 후 다시 import (또는 런타임 재시작) |
| `images/train 이 없습니다` | `pill_detection_dataset.ipynb` 의 YOLO 내보내기 셀을 먼저 실행 |
| `bbox 오류 > 0` | 라벨이 YOLO 정규화 형식(`cls cx cy w h`, 0~1)인지 확인 |
| `클래스 ID 오류 > 0` | `data.yaml` 의 `nc` 와 라벨의 최대 클래스 번호 불일치 |
| `크롭 클래스 매칭 0` | crop 폴더명이 `{category_id}_{category_name}` 형식인지 확인 |
| `크롭 라이브러리가 비어 있습니다` | `CUT_MIN_CHROMA` / `CUT_CHROMA_K` 를 낮춰 보거나 `crops_dir=None` 으로 실행 |
| albumentations 설치 오류 | YOLO 증강만 쓸 거면 설치 없이도 동작합니다 |
| 합성 배경이 원본과 달라 보임 | `--bg-check` 로 재측정 → `PILL_BG_RGB` 숫자 갱신 |
| 추론 mAP 가 0에 가까움 | `undo_letterbox_boxes()` 를 빠뜨렸는지 확인 |

---

## 8. 최종 체크리스트

- [ ] `pt.configure()` 출력에서 `DATASET_DIR ✅`
- [ ] `pt.preflight()` → **이미지 읽기 오류 · 라벨 없는 이미지 · bbox 오류 · 클래스 ID 오류 = 0**
- [ ] `pt.build_augmented_yolo_dataset()` → 기하 증강 검산 ✅ / Copy&Paste 검산 ✅
- [ ] `pt.preview_augmented()` 이미지 육안 확인 (박스 · 각인 · 합성 경계 · 배경색)
- [ ] YOLO 학습 시 Ultralytics 기본 증강이 전부 0인지
- [ ] Faster R-CNN train / predict 가 **같은** `get_frcnn_*_transform` 을 쓰는지
- [ ] 제출 좌표가 `undo_letterbox_boxes()` 를 거친 원본 좌표인지

원본 데이터(`data/dataset`, `data/processed`, crop 원본)는 팀 취합과 전체 검증이
끝나기 전까지 삭제하지 마세요. 이 모듈에는 원본을 지우는 기능이 없습니다.
