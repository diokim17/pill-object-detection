# pill_transforms.py 사용 설명서

경구약제(알약) Object Detection 프로젝트의 **전처리 · 증강 · 학습 설정 단일 출처** 모듈입니다.
YOLO 노트북, Faster R-CNN 노트북, EDA 노트북이 전부 이 파일 하나만 import 합니다.

> **처음이시면 [1. Colab 3분 시작](#1-colab-3분-시작)만 따라 하세요.**
> 나머지는 필요할 때 찾아보는 참고 문서입니다.

---

## 목차

1. [Colab 3분 시작](#1-colab-3분-시작)
2. [import 하는 법 (외부 ipynb에서)](#2-import-하는-법-외부-ipynb에서)
3. [폴더 배치 — 두 가지를 모두 지원합니다](#3-폴더-배치--두-가지를-모두-지원합니다)
4. [Colab 실전 가이드](#4-colab-실전-가이드)
5. [함수·클래스 사용법 (API)](#5-함수클래스-사용법-api)
6. [노트북별 붙이는 법](#6-노트북별-붙이는-법)
7. [하이퍼파라미터 전체 설명](#7-하이퍼파라미터-전체-설명)
8. [증강이 어떻게 되어 있나](#8-증강이-어떻게-되어-있나)
9. [왜 Copy&Paste 증강이 총 800장인가](#9-왜-copypaste-증강이-총-800장인가)
10. [설정 잠금 — 값을 바꾸려면](#10-설정-잠금--값을-바꾸려면)
11. [자주 나는 오류](#11-자주-나는-오류)

---

## 1. Colab 3분 시작

새 노트북 첫 셀에 이것만 붙여넣고 실행하세요.

```python
# ── 1) 드라이브 마운트 ──
from google.colab import drive
drive.mount('/content/drive')

# ── 2) pill_transforms.py 가 있는 "폴더"를 경로에 추가 ──
import sys
sys.path.insert(0, '/content/drive/MyDrive/pill-object-detection/src')   # ← 본인 경로

# ── 3) import + 환경 준비 ──
import pill_transforms as pt
P = pt.setup()
```

출력이 이렇게 나오면 성공입니다.

```
══════════════════════════════════════════════════════════════
  실행 환경   Google Colab
  폴더 배치   project  (config.yaml 규약을 따릅니다)
  저장소      /content/drive/MyDrive/pill-object-detection
  데이터      /content/.../data/dataset/cleaning_data/sprint_..._dataset
  산출물      /content/drive/MyDrive/pill-object-detection
──────────────────────────────────────────────────────────────
  PROJECT_ROOT   ...  ✅
  PILL_ROOT      ...  ✅
    train_images ...  ✅
  ...
══════════════════════════════════════════════════════════════
```

`setup()` 이 자동으로 하는 일:

| 단계 | 내용 |
|---|---|
| 1 | 드라이브 마운트 (안 돼 있으면) |
| 2 | `config.yaml` 이 있는 저장소 루트 탐색 → 있으면 그 경로 규약을 따름 |
| 3 | `train_images` / `train_annotations` 가 있는 데이터 루트 탐색 |
| 4 | 못 찾고 Colab 이면 드라이브의 `pilldata*.zip` 을 `/content` 에 압축 해제 |
| 5 | 산출물 폴더 생성 (`data/processed_aug`, `outputs/...`) |
| 6 | `ultralytics` + 한글 폰트(`fonts-nanum`) 설치 |
| 7 | `PillDetectionDataset.py` 를 `sys.path` 에 등록 |

자동으로 못 찾으면 직접 알려 주세요.

```python
P = pt.setup(
    project_root="/content/drive/MyDrive/pill-object-detection",
    pill_root="/content/pilldata",                       # train_images 의 상위 폴더
    work_root="/content/PillWork",                       # 산출물 위치
    zip_path="/content/drive/MyDrive/pilldata.zip",      # zip 을 쓸 때
)
```

**잘 됐는지 확인:**

```python
pt.check_paths()                  # 경로가 전부 ✅ 인지
print(pt.describe_hparams())      # 현재 하이퍼파라미터 표
pt.selftest()                     # 가짜 데이터로 전 과정 1회 실행 (30초쯤)
```

---

## 2. import 하는 법 (외부 ipynb에서)

`pill_transforms.py` 는 **평범한 파이썬 모듈**입니다. 그 파일이 있는 **폴더**를
`sys.path` 에 넣기만 하면 어디서든 import 됩니다.

### 방법 A — 경로를 직접 추가 (가장 확실, 권장)

```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/pill-object-detection/src')
import pill_transforms as pt
```

### 방법 B — 저장소 루트에서 실행하는 경우

```python
%cd /content/drive/MyDrive/pill-object-detection
import sys; sys.path.insert(0, 'src')
import pill_transforms as pt
```

### 방법 C — 파일을 세션에 복사 (드라이브가 느릴 때)

```python
!cp "/content/drive/MyDrive/pill-object-detection/src/pill_transforms.py" /content/
!cp "/content/drive/MyDrive/pill-object-detection/src/PillDetectionDataset.py" /content/
import sys; sys.path.insert(0, '/content')
import pill_transforms as pt
```

### 방법 D — 파일을 못 찾겠을 때 (자동 탐색 스니펫)

```python
import sys, glob, os

hits = glob.glob('/content/**/pill_transforms.py', recursive=True) + \
       glob.glob('/content/drive/MyDrive/**/pill_transforms.py', recursive=True)
assert hits, 'pill_transforms.py 를 못 찾았습니다. 드라이브에 올렸는지 확인하세요.'
print('찾음:', hits[0])
sys.path.insert(0, os.path.dirname(hits[0]))

import pill_transforms as pt
```

### ⚠️ 파일을 수정한 뒤에는 반드시 다시 읽어야 합니다

Colab 은 한 번 import 한 모듈을 **캐시에 물고 있습니다.** 드라이브에서 파일을 덮어써도
같은 세션에서는 옛 버전이 계속 돕니다. 그래서 이렇게 강제로 다시 읽으세요.

```python
import sys, importlib
sys.modules.pop('pill_transforms', None)      # ★ 캐시 제거
importlib.invalidate_caches()
import pill_transforms as pt
importlib.reload(pt)
print('로드된 파일:', pt.__file__)             # ★ 경로를 꼭 확인하세요
```

`AttributeError: module 'pill_transforms' has no attribute 'setup'` 이 나면
**십중팔구 이 캐시 문제**입니다. 위 스니펫을 실행하거나 **런타임 → 세션 다시 시작**하세요.

### 사본이 여러 개면 사고가 납니다

드라이브에 `pill_transforms.py` 가 여러 곳에 있으면 어느 것이 로드됐는지 헷갈립니다.
항상 `print(pt.__file__)` 로 확인하고, 옛 사본은 지우거나 이름을 바꾸세요.

---

## 3. 폴더 배치 — 두 가지를 모두 지원합니다

`setup()` 이 어느 쪽인지 자동 판정해서 (`P["LAYOUT"]`) 경로를 다르게 채웁니다.

### [A] `project` 배치 — 팀 git 저장소 (config.yaml 이 있는 쪽)

```
pill-object-detection/                ← PROJECT_ROOT
├── config.yaml
├── src/
│     ├── PillDetectionDataset.py
│     └── pill_transforms.py          ← 이 파일
├── notebooks/
├── data/
│   ├── dataset/cleaning_data/sprint_ai_project1_data_260809_baseline_dataset/
│   │        ├── train_images/  train_annotations/  test_images/     ← PILL_ROOT
│   ├── processed/                    YOLO 데이터셋 (pill_detection_dataset.ipynb 산출)
│   ├── processed_aug/                ★ 증강 결과 (이 모듈이 생성)
│   ├── train_another/                ★ 추가 데이터
│   ├── cropped_pills_review/         Copy&Paste 재료
│   └── team_work/cropped_output/     ★ AI Hub 추가분
└── outputs/
      ├── checkpoints/ predictions/ submissions/    (config.yaml 규약)
      ├── yolo/                                      (Ultralytics runs)
      └── experiments/ figures/ cutcheck/            (이 모듈이 생성)
```

이 배치에서는 **`config.yaml` 을 직접 읽어** `paths.dataset_root` 를 그대로 씁니다.
`${paths.project_root}` 같은 보간도 풀어 주므로 omegaconf 가 없어도 됩니다.

```python
cfg = pt.CONFIG                      # setup() 이 읽어 둔 config.yaml
cfg['paths']['dataset_root']         # ./data/dataset/cleaning_data/...
cfg['train']['epochs']               # 20
```

### [B] `pilldata` 배치 — zip 한 덩어리로 공유

```
pilldata/                             ← PILL_ROOT
├── train_images/ train_annotations/ test_images/
├── train_another/                    {category_id}_{약이름}/*.png
├── cropped_pills_review/             {category_id}_{약이름}/*.png
└── team_work/cropped_output/         ★ AI Hub 추가분

WORK_ROOT/                            ← 산출물은 분리 (원본을 안 건드리려고)
├── data/pill_raw, data/pill_aug
└── outputs/, runs/, experiments/, cutcheck/
```

### 배치별 경로 대응표

| `paths()` 키 | project 배치 | pilldata 배치 |
|---|---|---|
| `RAW_DIR` (= `PROCESSED_DIR`) | `data/processed` | `data/pill_raw` |
| `AUG_DIR` | `data/processed_aug` | `data/pill_aug` |
| `ANOTHER_DIR` | `data/processed_another` | `data/pill_another` |
| `RUN_DIR` | `outputs/yolo` | `runs` |
| `EXP_DIR` | `outputs/experiments` | `experiments` |
| `CKPT_DIR` | `outputs/checkpoints` | `outputs/checkpoints` |

**어느 배치든 노트북 코드는 똑같습니다.** `P["AUG_DIR"]` 처럼 키로만 쓰면 됩니다.

### 추가 데이터 폴더 이름 규칙

`train_another` 와 크롭 폴더는 **`{category_id}_{약이름}`** 형식이어야 클래스가 매칭됩니다.

```
train_another/3832_뉴로메드정(옥시라세탐)/K-003351-003832-016232_0_2_0_2_90_000_200.png
cropped_pills_review/1900_보령부스파정 5mg/*.png
team_work/cropped_output/16548_일양하이트린정 2mg/*.png
```

매칭 실패 시 실행 중에 알려 줍니다: `⚠️ 클래스명 매칭 실패 폴더 3개: ...`

**Copy&Paste 재료는 `cropped_pills_review` 와 `team_work/cropped_output` 을 동시에 씁니다.**
같은 클래스가 양쪽에 있으면 번갈아 뽑아 한쪽이 `MAX_CROPS_PER_CLASS` 를 독점하지 않게 합니다.

```python
pt.resolve_crop_dirs()   # 실제로 쓰이는 재료 폴더 목록
```

---

## 4. Colab 실전 가이드

### ① 드라이브를 직접 읽지 마세요 — 학습이 몇 배 느려집니다

이미지 수만 장을 드라이브에서 한 장씩 읽으면 네트워크 지연이 누적됩니다.
**zip 한 개만 복사해 로컬 디스크(`/content`)에서 푸는 게 훨씬 빠릅니다.**

```python
root = pt.unzip_pilldata('/content/drive/MyDrive/pilldata.zip', dest='/content')
P = pt.setup(pill_root=root)
```

`setup()` 은 데이터를 못 찾으면 이걸 자동으로 시도합니다.

### ② `/content` 는 세션이 끊기면 사라집니다

학습이 끝나면 결과를 드라이브로 옮기세요.

```python
pt.save_outputs_to_drive('/content/drive/MyDrive/pill-object-detection/outputs')
# 가중치(.pt)·csv·png 만 골라서 복사합니다 (중간 이미지까지 옮기면 드라이브가 금방 참)
```

산출물을 처음부터 드라이브에 쓰고 싶으면:

```python
P = pt.setup(work_root='/content/drive/MyDrive/pill-object-detection')
```
> 다만 증강 이미지 수천 장을 드라이브에 쓰면 **증강 단계가 매우 느려집니다.**
> 증강은 `/content` 에서 하고 결과만 옮기는 쪽을 권합니다.

### ③ 패키지·폰트

```python
pt.install_colab_deps()      # ultralytics + fonts-nanum (setup() 이 자동 호출)
pt.find_korean_font()        # 폰트 경로 (라벨이 □□□ 로 보이면 확인)
```

### ④ GPU 확인

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```
T4(16GB)에서 `imgsz=1024, batch=16` 이 도는 상한입니다.
**OOM 이 나면 `IMGSZ` 가 아니라 `DEFAULT_BATCH` 를 먼저 줄이세요** (해상도를 내리면 각인이 안 읽힙니다).

### ⑤ 팀 공유 체크리스트

- [ ] 드라이브에 `src/pill_transforms.py`, `src/PillDetectionDataset.py`, `config.yaml` 올리기
- [ ] 데이터는 zip 으로 (풀면 `train_images/` 가 나오게)
- [ ] 팀원은 노트북 첫 셀에 [1절](#1-colab-3분-시작) 스니펫 붙여넣기
- [ ] `pt.selftest()` 로 환경 확인
- [ ] 학습 후 `pt.save_outputs_to_drive()`

---

## 5. 함수·클래스 사용법 (API)

### 5-1. 환경 · 경로

```python
P = pt.setup(pill_root=None, work_root=None, *, project_root=None, config=None,
             zip_path=None, mount=True, unzip=True, install=True, verbose=True)
```
★ 전체 준비. 반환값은 아래 `paths()` 와 같은 dict.

```python
pt.paths()                # 현재 경로 dict (setup 이후 언제든)
pt.check_paths()          # 경로 존재 여부를 ✅/❌ 로 출력
pt.in_colab()             # Colab 인지 True/False
pt.mount_drive()          # 드라이브 마운트
pt.detect_project_root()  # config.yaml 이 있는 저장소 루트 탐색
pt.detect_pill_root()     # train_images 가 있는 데이터 루트 탐색
pt.load_config('config.yaml')       # ${} 보간까지 풀어서 dict 반환
pt.find_pilldata_zip()              # 드라이브에서 zip 찾기
pt.unzip_pilldata(zip_path, dest)   # 로컬 디스크에 압축 해제
pt.install_colab_deps()             # ultralytics + 한글 폰트
pt.add_src_to_path()                # PillDetectionDataset.py 등록
pt.resolve_crop_dirs()              # Copy&Paste 재료 폴더 목록
pt.save_outputs_to_drive()          # 결과만 드라이브로 복사
```

`paths()` 가 돌려주는 키:

```python
P["PROJECT_ROOT"] P["PILL_ROOT"] P["DATASET_ROOT"] P["SRC_DIR"] P["LAYOUT"]
P["IMAGE_DIR"] P["ANN_DIR"] P["TEST_DIR"]
P["ANOTHER_ROOT"] P["CROPPED_PILLS_DIR"] P["TEAM_WORK_DIR"]
P["RAW_DIR"] P["PROCESSED_DIR"] P["ANOTHER_DIR"] P["AUG_DIR"]
P["OUT_ROOT"] P["FIG_DIR"] P["PRED_DIR"] P["SUB_DIR"] P["CKPT_DIR"]
P["RUN_DIR"] P["EXP_DIR"] P["RESULT_CSV"] P["CUTOUT_CHECK_DIR"]
```

### 5-2. 전처리 — **학습·평가·추론에 반드시 똑같이**

가장 흔한 실패가 **학습에만 전처리를 걸고 추론에 안 거는 것**입니다.

```python
img = pt.imread_unicode(path)        # 한글 경로 안전한 cv2.imread (BGR)
out = pt.preprocess(img)             # ★ 화이트밸런스 + CLAHE (BGR in / BGR out)
pt.imwrite_unicode(path, out)        # 한글 경로 안전한 저장

pt.Preprocess.shades_of_gray(img)    # 화이트밸런스만
pt.rebuild_preprocess()              # 설정을 바꾼 뒤 싱글턴 갱신
```

추론에서는 이렇게 씁니다 (크기를 안 바꾸므로 좌표 보정 불필요).

```python
img = pt.imread_unicode(test_path)
r = model.predict(pt.preprocess(img), imgsz=pt.DEFAULT_IMGSZ, conf=0.25)[0]
```

### 5-3. 증강 데이터셋 생성

```python
aug_yaml = pt.build_augmented_yolo_dataset(
    src_root=P["RAW_DIR"],          # YOLO 구조(images/labels/data.yaml) 입력
    dst_root=P["AUG_DIR"],
    geom_mult=None,                 # None = pill_transforms 의 값 (권장)
    n_synth=None,
    crops_dir=None,                 # None = resolve_crop_dirs() 전체
    cutout_check_dir=P["CUTOUT_CHECK_DIR"],
    preprocess_val_test=True,
    overwrite=True, verbose=True,
)
# → dst_root/data.yaml 경로를 반환. 그대로 model.train(data=aug_yaml) 에 넣으면 됩니다.
```

만들어지는 것: `images/{train,...}`, `labels/{...}`, `data.yaml`, **`augment_info.json`**
(어떤 설정으로 만들었는지 기록 — 평가·추론 노트북이 이걸 읽어 설정을 대조합니다).

### 5-4. `train_another` 통합

```python
info = pt.build_another_yolo(
    another_root=P["ANOTHER_ROOT"], dst_root=P["ANOTHER_DIR"],
    names=NAMES_LIST,               # data.yaml 의 클래스명 리스트 (매칭에 필요)
    mode="canvas",                  # "canvas"(권장) | "native" | "cp_only"
    canvas_hw=(1280, 976), n_canvas=None, seed=42, overwrite=True,
)
n = pt.merge_yolo_into(P["ANOTHER_DIR"], P["RAW_DIR"],
                       src_split="train", dst_split="train")
```

**모드 설명**

| 모드 | 하는 일 | 언제 |
|---|---|---|
| `canvas` ★ | 크롭을 **리사이즈 없이** 976×1280 캔버스에 배치 → 알약 픽셀 크기가 원본과 일치 | 기본 |
| `native` | 크롭을 원본 크기로 내보내고 크기 그룹별 `data_{W}x{H}.yaml` 생성 | 그룹별 학습 실험 |
| `cp_only` | YOLO 데이터로 안 만들고 Copy&Paste 재료로만 사용 | 가장 보수적 |

> **왜 그냥 섞으면 안 되나**
> `train_images` 는 976×1280 안에서 알약 면적비 **5.6%**, `train_another` 는 396×396 안에서 **90%** 입니다.
> 그냥 섞으면 같은 약이 180px / 900px 두 크기로 학습됩니다. 또 크롭 상단에 **클래스 이름 배너가 글자로 박혀 있어**
> 모델이 알약이 아니라 글자를 읽습니다 → `strip_review_panel()` 이 자동 제거합니다.

### 5-5. 컷아웃 검수 (Copy&Paste 에 실제로 붙는 그림 확인)

```python
res = pt.export_cutout_check(crops_dir=None, out_dir=P["CUTOUT_CHECK_DIR"],
                             names=NAMES_LIST, max_per_class=0, overwrite=True)
```
```
cutcheck/{category_id}_{클래스명}/cut/    ← 배경 투명(RGBA) — 실제로 붙는 그림
                                 /panel/  ← 원본 | 경계 | 체커보드 (눈으로 검수)
        /_failed/, cutout_report.csv, cutout_summary.csv
```

**학습 전에 panel 몇 장은 꼭 눈으로 보세요.** 그림자를 물고 있으면 `CUT_MIN_CHROMA` 를 낮춥니다.

### 5-6. 검수 시각화

```python
pt.preview_augmented(P["AUG_DIR"], out_dir=fig, per_kind=3, names=NAMES_LIST)
pt.preview_another(P["ANOTHER_DIR"], out_dir=fig, n=3, names=NAMES_LIST)
pt.measure_train_bg_hsv(f'{RAW}/images/train', f'{RAW}/labels/train', n=60)
```

### 5-7. 학습 인자 (YOLO)

```python
TRAIN_KW = pt.get_train_kwargs(smoke=False, rect=False, resume=False)
model.train(data=aug_yaml, project=P["RUN_DIR"], name="exp01", **TRAIN_KW)
```
epochs·patience·imgsz·batch·workers·증강 인자를 **전부** 채워 줍니다.
`smoke=True` 면 스모크 예산(10 에포크)을 씁니다.

> ⚠️ **인자를 빠뜨리면 Ultralytics 기본값이 조용히 켜집니다**
> (`translate=0.1`, `erasing=0.4`, `mosaic=1.0`). 그래서 이 dict 하나로 전부 넘깁니다.

```python
pt.get_online_aug("off")           # 증강 인자만
pt.describe_online_aug("off")      # 실제 값 표로 출력
pt.describe_hparams()              # 전체 하이퍼파라미터 표
pt.assert_config_matches(AUG_INFO) # 데이터 생성 설정 ↔ 현재 설정 대조
```

### 5-8. Albumentations transform (Faster R-CNN 용)

```python
from pill_transforms import get_valid_transforms

base = get_valid_transforms(image_size=640, to_tensor=False)
r = base(image=img_rgb, bboxes=[[x1, y1, x2, y2]], labels=[1])
r["image"], r["bboxes"], r["labels"]
```

| 함수 | 용도 |
|---|---|
| `get_train_transforms(image_size, to_tensor=True)` | 학습용 (증강 포함) |
| `get_valid_transforms(image_size, to_tensor=True)` | 검증용 (리사이즈+패딩만) ← Faster R-CNN baseline 이 사용 |
| `get_test_transforms(image_size)` | 추론용 |
| `get_train_transform` / `get_eval_transform` | 단수형 별칭 |
| `denormalize(tensor)` | 정규화 되돌리기 (시각화용) |
| `SafeAlbumentationsTransform` | 박스가 전멸하면 원본을 돌려주는 래퍼 |

bbox 포맷은 **`pascal_voc` = `[x1, y1, x2, y2]` 절대 픽셀**, 라벨 필드는 `labels` 입니다
(`PillDetectionDataset` 이 내보내는 형식과 같습니다).

### 5-9. 주요 클래스

| 클래스 | 역할 |
|---|---|
| `Preprocess` | 화이트밸런스 + CLAHE. 전역 싱글턴 `PREPROCESS` 를 `preprocess()` 가 씁니다 |
| `GeometricAugmentor` | 기하 증강 (회전·스케일·HSV·톤커브·노이즈·블러). 회전 박스를 **내접 타원**으로 계산 |
| `PillCropLibrary` | 크롭 재료 창고. 여러 폴더에서 읽어 클래스별로 보관, RGBA 캐시 |
| `CopyPasteAugmentor` | 컷아웃한 알약을 새 배경/원본 위에 배치 + 그림자 재합성 |
| `AnotherPillPool` | `train_another` 크롭에서 알약 패치를 꺼내 쓰는 재료 풀 |
| `Sample` / `Transform` / `Compose` | 내부 증강 파이프라인 기본 단위 |

보통은 직접 만들 필요 없이 `build_augmented_yolo_dataset()` 이 알아서 씁니다.

---

## 6. 노트북별 붙이는 법

### `pill_detection_dataset.ipynb` (YOLO 데이터셋 생성)

경로만 `pt` 에서 받아 오면 됩니다.

```python
import sys; sys.path.insert(0, 'src')
import pill_transforms as pt
P = pt.setup()

yolo_root = Path(P["RAW_DIR"])        # project 배치면 data/processed
dataset_root = P["PILL_ROOT"]
```

### `yolo11s_baseline_cp.ipynb` (YOLO 학습)

```python
import pill_transforms as pt
P = pt.setup()

aug_yaml = pt.build_augmented_yolo_dataset(
    src_root=P["RAW_DIR"], dst_root=P["AUG_DIR"],
)

from ultralytics import YOLO
model = YOLO(pt.DEFAULT_MODEL)
model.train(data=aug_yaml, project=P["RUN_DIR"], name="yolo11s_aug_cp",
            exist_ok=True, **pt.get_train_kwargs())
```

제출 단계에서 **전처리를 반드시 동일하게**:

```python
img = pt.imread_unicode(str(f))
img = pt.preprocess(img)                      # ★ 학습과 동일
r = model.predict(img, imgsz=pt.DEFAULT_IMGSZ, conf=0.25, max_det=4)[0]
```

### `base_model_faster-rcnn_train.ipynb`

기존 코드를 그대로 두고 경로만 `pt` 로 바꾸면 됩니다.

```python
import sys; sys.path.insert(0, str(Path('./src').resolve()))
import pill_transforms as pt
P = pt.setup()
cfg = pt.CONFIG                                # config.yaml 을 이미 읽어 뒀습니다

from PillDetectionDataset import PillDetectionDataset, detection_collate_fn
from pill_transforms import get_valid_transforms

dataset = PillDetectionDataset(
    root=P["PILL_ROOT"],
    image_dir_name=cfg["dataset"]["image_dir_name"],
    annotation_dir_name=cfg["dataset"]["annotation_dir_name"],
    label_offset=cfg["dataset"]["label_offset"],
    strict=cfg["dataset"]["strict"],
)
base_transform = get_valid_transforms(image_size=640, to_tensor=False)
checkpoint_dir = Path(P["CKPT_DIR"])           # outputs/checkpoints
```

### `base_model_faster-rcnn_predict.ipynb`

```python
import pill_transforms as pt
P = pt.setup()
ckpt = Path(P["CKPT_DIR"]) / "faster_rcnn_best.pth"
test_dir = P["TEST_DIR"]

img = pt.preprocess(pt.imread_unicode(p))      # ★ 학습과 동일한 전처리
```

> `%cd PROJECT_ROOT` 로 작업 폴더를 옮기는 기존 방식도 그대로 동작합니다.
> `setup()` 이 현재 폴더에서 `config.yaml` 을 찾아 저장소 루트로 인식합니다.

---

## 7. 하이퍼파라미터 전체 설명

`pt.describe_hparams()` 로 언제든 현재 값을 볼 수 있습니다.

### 7-1. 학습 예산 · 해상도

| 이름 | 값 | 의미 / 왜 이 값인가 |
|---|---|---|
| `DEFAULT_MODEL` | `yolo11s.pt` | 시작 가중치 |
| `DEFAULT_IMGSZ` | **1024** | ★ 각인(RE20, NR800…)이 클래스 정보라 해상도가 성능에 직결됩니다. 640에서는 `print_sensitive` 클래스의 박스 짧은 변이 60px 아래로 떨어져 각인이 안 읽힙니다. T4/L4 에서 batch 16 으로 도는 상한선 |
| `DEFAULT_BATCH` | 16 | ⚠️ **VRAM 부족 시 `IMGSZ` 가 아니라 이걸 먼저 내리세요** |
| `DEFAULT_EPOCHS` | 100 | 본 학습 상한 |
| `DEFAULT_PATIENCE` | 15 | 조기 종료. ⚠️ 홀드아웃이 없으면 `val = train` 이라 **과적합을 못 잡습니다** |
| `DEFAULT_SMOKE_EPOCHS` | 10 | 파이프라인 점검용 |
| `DEFAULT_WORKERS` | 0 | Windows·Colab 모두 0 이 안전 |
| `SEED` | 42 | random / numpy / torch 고정 |

### 7-2. 전처리 (학습·평가·추론 **전부** 동일)

| 이름 | 값 | 의미 |
|---|---|---|
| `USE_WHITE_BALANCE` | True | Shades-of-Gray. 채널별 p-norm 으로 색 캐스트 제거 → **배경색·조명색이 알약 색으로 새는 것을 막습니다** (`color1`이 클래스 정보) |
| `USE_CLAHE` | True | Lab 의 **L 채널만** 국소 평탄화 → 알약과 그림자의 대비를 벌립니다 |
| `CLAHE_CLIP` | 5.0 | 대비 강도. 노이즈가 뜨면 4.0 |
| `CLAHE_GRID` | 8 | 타일 격자 |

> Grayscale·Retinex 는 **안 씁니다.** 색이 클래스 정보라 색을 잃으면 안 됩니다.

### 7-3. 오프라인 기하 증강

| 이름 | 값 | 의미 |
|---|---|---|
| `DEFAULT_GEOM_MULT` | **3** | 원본 1장 → 최종 3장 (원본 1 + 증강 2) |
| `ROT_LIMIT` / `ROT_PROB` | 180 / 0.8 | 알약은 회전 불변 |
| `SCALE_RANGE` | (0.95, 1.05) | 실제 알약 크기가 클래스 단서라 거의 안 건드림 |
| `USE_FLIP` | **False** | ★ **각인 글자가 뒤집히면 클래스 단서가 파괴됩니다** |
| `MIN_VISIBILITY` | 0.2 | 잘린 뒤 원면적 20% 미만이면 박스 삭제 |
| `HSV_H_LIMIT` | 8 | ★ 색상(H)은 아주 조금만 |
| `HSV_S_LIMIT` / `HSV_V_LIMIT` | 25 / 30 | 조명 변화 대응 |
| `P_BLUR` / `P_NOISE` / `P_TONE` | 0.20 / 0.30 / 0.30 | **각인이 뭉개지면 `P_BLUR`를 0으로** |

### 7-4. Copy & Paste

| 이름 | 값 | 의미 |
|---|---|---|
| `DEFAULT_N_SYNTH` | **600** | 합성 이미지 장수 |
| `CP_MODE` | `"mix"` | 인공 배경 + 원본 사진 위 붙이기를 반반 |
| `DEFAULT_CP_WEIGHTED` | True | ★ **역빈도 가중** — 희소 클래스를 더 자주 |
| `PILLS_RANGE` | (3, 4) | 합성 1장당 알약 수 |
| `UNIQUE_CLASS_PER_IMAGE` | True | 한 장에 같은 종류 두 번 금지 |
| `CP_OVERLAP` / `CP_FEATHER` | 0.10 / 2 | 겹침 허용 / 경계 페더링(후광 보이면 1) |
| `MAX_CROPS_PER_CLASS` | 40 | 클래스당 재료 상한 |
| `BG_MODE` / `BG_HSV_DEFAULT` | `"fixed"` / (117, 62, 131) | 합성 배경을 원본 실측색으로 고정 |

### 7-5. `train_another` 얹기

| 이름 | 값 | 의미 |
|---|---|---|
| `ANOTHER_ONTO_N` | **200** | 원본 사진 위에 another 알약을 얹은 합성본 장수 |
| `ANOTHER_ONTO_BASE_PILLS` / `_ADD` | 3 / 1 | 알약 3개 원본 + 1개 = **4개** |
| `ANOTHER_ONTO_ROTATE` | True | 얹기 전 임의 회전 |

### 7-6. 컷아웃 (그림자 배제)

그림자는 배경과 **색은 같고 밝기만 어둡습니다.** 밝기로 가르면 그림자가 알약으로 오인되므로
Lab 의 `(a,b)` **색상 거리**로 판별합니다.

| 이름 | 값 | 조정 |
|---|---|---|
| `CUT_CHROMA_K` | 2.0 | Copy&Paste 가 0장이면 1.6 |
| `CUT_MIN_CHROMA` | 6.0 | 실패가 많으면 5.0 |

### 7-7. 온라인 증강 (Ultralytics 내장)

오프라인에서 이미 증강했으므로 내장 증강은 거의 끕니다. 둘 다 걸면 **이중 증강**입니다.

| 프리셋 | 내용 |
|---|---|
| **`"off"`** (기본) | `hsv_s=0.3` + `hsv_h=0.015`, `translate=0.1`, `shear=2.0` |
| `"strict_off"` | **`hsv_s=0.3` 만 켜고 나머지 전부 0** |
| `"light"` | 위 + 소각도 회전(15°) |
| `"full"` | 증강 없는 원본으로 학습할 때 (회전 180°) |

> ⚠️ **`"off"` 는 이름과 달리 완전히 꺼져 있지 않습니다.** `hsv_h·translate·shear` 가 살아 있습니다.
> 기존 실험과의 연속성 때문에 값을 그대로 뒀습니다. 문서대로 "hsv_s만"을 원하면
> `"strict_off"` 를 쓰고 **두 프리셋을 비교 실험**해 보세요.

---

## 8. 증강이 어떻게 되어 있나

```
train_images (원본 N장)
      │
      ├─→ train_another ──[배너 제거 → 컷아웃 → 976×1280 캔버스 배치]──┐
      │                                                                 │
      ▼                                                                 ▼
   RAW_DIR/images/train  =  train_images + train_another  (= M장)
      │
      │  build_augmented_yolo_dataset()
      ▼
   ┌────────────────────────────────────────────────────────┐
   │ [1] 원본 그대로 저장 (전처리만)               M장      │
   │ [2] 기하 증강  이미지당 (GEOM_MULT-1)장    M×2장       │
   │ [3] Copy & Paste                             600장     │
   │ [4] train_another 얹기 (원본 + 알약 1개)     200장     │
   └────────────────────────────────────────────────────────┘
      ▼
   AUG_DIR/  ← 학습에 쓰는 최종 데이터
```

파일 이름 접두사로 출처를 구분합니다.

| 접두사 | 뜻 |
|---|---|
| (없음) | `train_images` 원본 |
| `an_cv_` | `train_another` 캔버스 합성 |
| `aug_` | 기하 증강본 (`aug_an_cv_` = another 의 증강본) |
| `cp_` | Copy & Paste |
| `an_onto4_` | 원본 사진 + another 알약 1개 |

### 회전 시 박스 계산 — 내접 타원

회전한 박스를 축정렬 외접 사각형으로 감싸면 길쭉한 알약을 45° 돌렸을 때 **면적이 2.7배**로 부풉니다.
대신 **박스에 내접한 타원**을 회전시켜 그 외접 사각형을 씁니다.

```
반너비 = √( (a·cosθ)² + (b·sinθ)² )       a = w/2
반높이 = √( (a·sinθ)² + (b·cosθ)² )       b = h/2
```

### Copy & Paste 가 그림자 상관관계를 끊습니다

원본에서는 "알약이 있는 곳에 항상 그림자가 있다"는 상관이 생깁니다.
모델이 그림자를 단서로 삼으면 조명이 바뀐 테스트 이미지에서 무너집니다.
Copy&Paste 는 알약을 **마스크로 오려 내(그림자를 물리적으로 제거)** 새 배경에 놓고
그림자를 **별도 레이어로 재합성**합니다.

### 한계

기하 증강은 **같은 사진을 여러 번 보는 것**이라 원본 다양성이 늘지 않습니다.
표본이 3장인 클래스를 3배 증강해도 3장의 변형일 뿐입니다.
**각인이 640px에서 안 읽히는 문제는 증강으로 해결되지 않습니다** — `IMGSZ`를 올리거나 2-stage 로 가야 합니다.

---

## 9. 왜 Copy&Paste 증강이 총 800장인가

**800 = Copy&Paste 600장 + train_another 얹기 200장**

### 설계 의도: 기하 증강 : 합성 = 1 : 1

| 종류 | 늘어나는 장수 | 성격 |
|---|---|---|
| **기하 증강** | (원본+another) × (GEOM_MULT−1) | 같은 사진의 변형. **다양성은 안 늘어남** |
| **합성** (CP + another얹기) | 600 + 200 = **800** | 새 배경·새 조합. **다양성이 실제로 늘어남** |

한쪽으로 치우치면 문제가 생깁니다.

- **기하 증강만 많으면** → 같은 사진 반복 학습으로 과적합, 희소 클래스는 여전히 희소
- **합성만 많으면** → 합성 특유의 인공적 질감(경계·배경)을 학습해 진짜 사진에서 성능 저하

원본 200여 장 규모에서 `GEOM_MULT=3`, `N_SYNTH=600` 을 쓰면 기하 증강 증가분이 수백 장이 되므로,
여기에 **200장을 더해 800장**으로 균형을 맞춘 것입니다.

### 600 + 200 으로 나눈 이유 — 서로 다른 약점을 메웁니다

| | Copy&Paste 600장 | train_another 얹기 200장 |
|---|---|---|
| 배경 | **인공 배경**(고정 HSV) + 원본 사진 혼합 | **진짜 사진** 그대로 |
| 알약 재료 | `cropped_pills_review` + AI Hub 크롭 | `train_another` 크롭 |
| 알약 수 | 3~4개 | 3개(원본) + 1개 = **4개** |
| 강점 | 희소 클래스를 **역빈도 가중**으로 집중 보강 | **도메인 갭이 가장 작음** |
| 약점 | 인공 배경 질감이 진짜와 미묘하게 다름 | 원본 사진 수만큼만 생성 가능 |

즉 **600은 "클래스 불균형 해소"용, 200은 "도메인 갭 축소"용**입니다.
200이 더 적은 이유는 "알약 3개짜리 원본 사진"을 재료로 쓰기 때문에 만들 수 있는 장수가 원본 수에 묶여서입니다.

### 비율은 데이터가 바뀌면 달라집니다

원본 장수 N이 바뀌면 기하 증강 증가분도 바뀌므로 1:1이 자동 유지되지 않습니다.
증강 후 실제 비율을 이렇게 확인하세요.

```python
import glob, os
from collections import Counter

k = Counter()
for p in glob.glob(f'{P["AUG_DIR"]}/labels/train/*.txt'):
    n = os.path.basename(p)
    k['cp' if n.startswith('cp_') else
      'onto4' if n.startswith('an_onto4_') else
      'aug' if n.startswith('aug_') else
      'another' if n.startswith('an_') else 'orig'] += 1

ratio = (k['cp'] + k['onto4']) / max(k['aug'], 1)
print(f"기하 {k['aug']} vs 합성 {k['cp'] + k['onto4']}  → 비율 {ratio:.2f}")
print("✅ 1:1 근처" if 0.7 <= ratio <= 1.4 else "⚠️ 조정 검토")
```

> **끄려면** `ANOTHER_ONTO_N = 0` / `DEFAULT_N_SYNTH = 0`.
> 둘 다 `pill_transforms.py` 를 직접 고쳐야 합니다 (다음 절).

> ℹ️ `ANOTHER_ONTO_N` 이 0장으로 나온다면, `UNIQUE_CLASS_PER_IMAGE=True` 때문에
> 원본에 이미 있는 클래스를 제외하고 나면 얹을 클래스가 남지 않은 경우입니다.
> 클래스가 아주 적은 데이터에서만 생깁니다 (56종이면 정상 동작).

---

## 10. 설정 잠금 — 값을 바꾸려면

하이퍼파라미터는 **노트북에서 덮어쓸 수 없게 잠겨 있습니다.**

```python
pt.DEFAULT_EPOCHS = 50
# AttributeError: pill_transforms.DEFAULT_EPOCHS 은 잠겨 있습니다.
#   현재 값 : 100 / 시도한 값: 50
#   → pill_transforms.py 를 직접 고치고 커널을 재시작하세요.
```

**왜 막았나** — 팀원마다 노트북에서 다른 epochs·해상도로 돌리면 실험 비교가 무의미해집니다.

**바꾸는 방법**

1. `pill_transforms.py` 를 열어 값을 고칩니다
2. **런타임(커널) 재시작** — 안 하면 옛 값이 그대로 돕니다
3. 증강·전처리 값을 바꿨다면 **증강 데이터를 다시 만듭니다**

**잠긴 항목** — 학습 예산(`EPOCHS`, `PATIENCE`, `IMGSZ`, `BATCH`, `MODEL`, `WORKERS`, `SEED`),
추론 기본값(`CONF`, `IOU_NMS`, `MAX_DET`), 온라인 증강 프리셋,
증강량(`GEOM_MULT`, `N_SYNTH`, `ANOTHER_ONTO_N`), 전처리(`CLAHE_*`), 기하·CP·컷아웃 상수.

**경로는 잠겨 있지 않습니다** (환경마다 달라야 하므로).

같은 값 대입은 통과하므로, 데이터 생성 설정과 대조하는 코드는 그대로 동작합니다.

```python
pt.assert_config_matches(AUG_INFO)
# ⚠️ 데이터를 만든 설정과 지금 pill_transforms.py 가 다릅니다:
#      clahe_clip     데이터 6.0  ≠  현재 5.0
```

---

## 11. 자주 나는 오류

| 증상 | 원인 / 해결 |
|---|---|
| `ModuleNotFoundError: pill_transforms` | `sys.path` 에 **파일이 아니라 폴더**를 넣었는지 확인 → [2절](#2-import-하는-법-외부-ipynb에서) |
| `AttributeError: ... has no attribute 'setup'` | 모듈 캐시에 옛 버전이 남았습니다 → `sys.modules.pop` + `reload` 또는 세션 재시작 |
| `옛 버전입니다 (없는 기능: ...)` | 파일을 덮어쓴 뒤 **런타임 재시작**을 안 했습니다 |
| `원본 데이터 폴더를 찾지 못했습니다` | `pt.setup(pill_root="...")` 로 직접 지정 (train_images 의 **상위** 폴더) |
| `AttributeError: ... 은 잠겨 있습니다` | 의도된 동작입니다 → [10절](#10-설정-잠금--값을-바꾸려면) |
| `크롭 폴더를 찾을 수 없습니다` | `pt.resolve_crop_dirs()` 로 확인, 없으면 train 라벨에서 직접 컷아웃됩니다 |
| `클래스명 매칭 실패 폴더` | 폴더명이 `{category_id}_{약이름}` 형식이 아닙니다 |
| Copy&Paste 가 0장 | 컷아웃 전부 실패 → `CUT_MIN_CHROMA` 5.0, `CUT_CHROMA_K` 1.6 |
| 합성 경계에 후광 | `CP_FEATHER` 를 1로 |
| 각인이 뭉개짐 | `P_BLUR = 0.0`, `P_NOISE` ↓ |
| 학습이 너무 느림 | 드라이브 경로를 직접 읽고 있진 않은지 → `/content` 로 풀어서 쓰세요 |
| CUDA out of memory | `IMGSZ` 가 아니라 **`DEFAULT_BATCH`** 를 8 또는 4로 |
| 라벨이 □□□ | `pt.install_colab_deps()` (한글 폰트) |
| 세션이 끊겨 결과 소실 | `pt.save_outputs_to_drive()` 를 습관화 |
| mAP 는 높은데 제출 점수가 0 | `category_id` 역매핑 확인. YOLO 는 0..N-1, 제출은 원래 id(1900, 16548…) |
| 제출 점수가 유독 낮다 | **추론에 `pt.preprocess()` 를 걸었는지** 확인 — 가장 흔한 실패입니다 |
