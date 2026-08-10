# 알약(경구약제) Object Detection 프로젝트 — Colab 실행 가이드

이 저장소는 알약 이미지 데이터셋으로 YOLO 기반 Object Detection 모델을 학습하기 위한
파이프라인입니다. 노트북 4개 + 파이썬 모듈 1개로 구성되어 있고, **위에서 아래로 순서대로**
실행하면 됩니다.

## 0. 먼저 알아야 할 것

- 모든 노트북은 원래 로컬(Windows, `D:/PillData`)에서 작성되었습니다. **Colab에서 그대로
  실행하면 경로 에러가 납니다.** 아래 "Colab 실행 전 공통 체크리스트"를 반드시 먼저 하세요.
- 노트북 사이에 **파일로 결과를 주고받습니다** (JSON, `.py` 모듈, `data.yaml` 등). 중간 단계를
  건너뛰면 다음 노트북이 실행되지 않습니다.
- 아래 표의 순서를 반드시 지키세요.

## 1. 파일 구성 및 실행 순서

| 순서 | 파일 | 역할 | 입력 | 출력 |
|---|---|---|---|---|
| 1 | `02_crop_dataset.ipynb` | 원본 annotation의 bbox 기준으로 알약을 crop해 클래스별 폴더에 저장 (라벨 검수용) | `train_images/`, `train_annotations/` | `cropped_pills_review/` (검수용 crop 이미지) |
| 2 | `pill_detection_dataset.ipynb` | `PillDetectionDataset` 클래스 정의, train/val/test 분할, DataLoader 구성, **YOLO 형식(`pill_yolo_dataset/`)으로 변환·저장**, bbox 시각화 | 원본 이미지 + annotation | `pill_yolo_dataset/`(images/labels/train,val,test + `dataset.yaml`), `pill_dataset.py` |
| 3 | `pill_transforms.py` | Albumentations 기반 증강 파이프라인 모듈 (`get_train_transforms`, `get_valid_transforms` 등). 필요 시 `pill_transforms__1_.py`(albumentations 없이 cv2+numpy로 재구현한 버전)로 교체 가능 | — | 다른 노트북에서 `import` |
| 4 | `02_baseline.ipynb` | 전처리(화이트밸런스+CLAHE) → 클래스 가중치 기반 오프라인 증강 → copy-paste 합성 → YOLO 인코딩 → **YOLO 베이스라인 학습(epochs=10 스모크 테스트)** | `pill_dataset.py`, `splits.json`, `class_attributes.json` (※ 아래 4번 참고) | `data/aug/`, `data/pill_yolo/`, `runs/baseline/` |
| 5 | `03_train.ipynb` | 본 학습용 노트북. 모델/하이퍼파라미터를 바꿔가며 실험하고 결과를 `experiments/log.jsonl`에 자동 기록 | `data/pill_aug/data.yaml` | `runs/<실험명>/weights/best.pt`, `experiments/log.jsonl` |

> ⚠️ `02_baseline.ipynb`는 `01_eda.ipynb`(본 업로드에는 없음)가 만드는
> `outputs/splits.json`, `outputs/class_attributes.json`을 입력으로 기대합니다. 이 파일들이
> 없으면 `01_eda.ipynb`를 먼저 실행하거나, 직접 8:1:1 분할 JSON과 클래스 가중치 JSON을
> 같은 스키마로 만들어 넣어야 합니다.

---

## 2. Colab 실행 전 공통 체크리스트

### (1) Google Drive 마운트 + 데이터 배치

```python
from google.colab import drive
drive.mount("/content/drive")
```

원본 데이터(`train_images/`, `train_annotations/`)를 Drive의 원하는 위치에 올려두고,
아래 "경로 수정" 섹션의 `PROJECT_DIR` / `DATA_ROOT` / `dataset_root`를 그 위치로 바꿔주세요.

### (2) 한글 폰트 설치 (시각화용)

시각화 셀에서 약 이름을 한글로 표시하므로, Colab에서는 나눔고딕을 설치해야 합니다.

```bash
!apt-get -qq install -y fonts-nanum
!fc-cache -fv
```

`02_crop_dataset.ipynb`는 `/usr/share/fonts/truetype/nanum/NanumGothic.ttf` 경로를
이미 이렇게 가정하고 있어 별도 코드 수정이 필요 없습니다. `02_baseline.ipynb` /
`03_train.ipynb`의 `find_korean_font()`도 이 경로를 자동으로 찾으므로 위 설치만 해두면 됩니다.

### (3) 필요 패키지 설치

```bash
!pip install -q ultralytics albumentations
```

- `ultralytics`: YOLO 학습/추론 (`02_baseline.ipynb`, `03_train.ipynb`에서 사용)
- `albumentations`: `pill_transforms.py` 사용 시에만 필요 (알버멘테이션 없는 버전인
  `pill_transforms__1_.py`를 쓰면 설치하지 않아도 됩니다)

---

## 3. 노트북별로 반드시 고쳐야 하는 경로

### `02_crop_dataset.ipynb`

```python
PROJECT_DIR = Path("/content/drive/MyDrive/코드잇_파트2_3팀_프로젝트")
DATA_DIR = PROJECT_DIR / "project1-data"
```
이미 Colab(Drive) 경로로 되어 있어 **폴더명만 본인 팀 Drive 구조에 맞게 확인**하면 됩니다.

### `pill_detection_dataset.ipynb`

```python
dataset_root = "./sprint_ai_project1_data"   # ← 실제 데이터 폴더 절대경로로 변경
yolo_root = Path("./pill_yolo_dataset")      # ← 원하는 저장 위치로 변경 가능(기본값 그대로도 동작)
```

⚠️ **시각화 셀에 Windows 전용 폰트 경로가 하드코딩되어 있습니다.**
```python
font_path = r"C:\Windows\Fonts\malgun.ttf"
```
Colab에서는 이 줄을 아래처럼 바꿔야 시각화 셀이 에러 없이 동작합니다.
```python
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
```

⚠️ **`pill_dataset.py` 파일을 만들어야 합니다.** `02_baseline.ipynb`가
`from pill_dataset import PillDetectionDataset`로 이 클래스를 불러오는데, 지금 노트북에는
`%%writefile` 매직이 없어 `.py` 파일이 저장되지 않습니다. `PillDetectionDataset` 클래스가
정의된 코드 셀(0번 셀) 맨 위에 아래 한 줄을 추가하고 다시 실행하세요.
```python
%%writefile pill_dataset.py
```
그러면 같은 작업 디렉터리에 `pill_dataset.py`가 생성되고, 이후 노트북들이 이를 import할 수
있습니다. (Colab이라면 `02_baseline.ipynb`도 같은 세션/같은 작업 폴더에서 열어야 합니다.)

### `02_baseline.ipynb`

```python
DATA_ROOT = r"D:/PillData"   # ← Windows 로컬 경로. 반드시 Drive 경로로 교체
```
예:
```python
DATA_ROOT = "/content/drive/MyDrive/코드잇_파트2_3팀_프로젝트/PillData"
```
이 값 하나만 바꾸면 `PILL_ROOT`, `OUT_ROOT`, `AUG_DIR`, `YOLO_DIR`, `RUN_DIR` 등 하위 경로가
전부 자동으로 따라갑니다 (모두 `DATA_ROOT` 기반의 f-string이라서).

또한 이 노트북 맨 위 표에 나온 대로 `outputs/splits.json`, `outputs/class_attributes.json`이
`DATA_ROOT/outputs/` 아래에 미리 있어야 합니다.

YOLO 학습 셀 상단의 주석 처리된 설치 명령을 실행하세요.
```python
!pip install ultralytics
```

### `03_train.ipynb`

```python
DATA_ROOT = "D:/PillData"   # ← 위와 동일하게 Drive 경로로 교체
```
`WORKERS = 0` 은 "Windows는 0이 안전"이라는 이유로 설정된 값인데, Colab(Linux)에서는
`WORKERS`를 2~4 정도로 올려도 무방합니다(속도 향상). 필요 없다면 0 그대로 둬도 동작합니다.

`RESUME = False` 는 Colab 세션이 끊겼다가 다시 학습을 이어갈 때 `True`로 바꿔 쓰라고
만들어둔 옵션이니 참고하세요.

---

## 4. `pill_transforms.py` vs `pill_transforms__1_.py`

두 파일은 **같은 인터페이스**(`get_train_transforms`, `get_valid_transforms`,
`get_test_transforms`, `SafeAlbumentationsTransform`, `denormalize`)를 제공하므로
어느 쪽을 import해도 `PillDetectionDataset`이나 다른 노트북 코드를 고칠 필요가 없습니다.

| | `pill_transforms.py` | `pill_transforms__1_.py` |
|---|---|---|
| 의존성 | `albumentations`, `albumentations.pytorch` 필요 | **의존성 없음** (`cv2` + `numpy`만 사용, `torch`는 `to_tensor=True`일 때만 지연 import) |
| 장점 | albumentations의 검증된 구현 그대로 사용 | Colab/아나콘다 환경에서 `opencv-python-headless`가 기존 `cv2`를 덮어써 발생하는 충돌 위험이 없음 |
| 언제 쓰나 | `!pip install albumentations`가 문제없이 되는 환경 | albumentations 설치/충돌이 걱정되거나, 순수 cv2 파이프라인을 원할 때 |

Colab에서는 보통 `albumentations` 설치가 무난하므로 `pill_transforms.py`를 그대로 써도 되고,
설치 이슈가 생기면 `pill_transforms__1_.py`로 바꿔 끼우면 됩니다. 사용 예:

```python
from pill_transforms import get_train_transforms, get_valid_transforms
# 또는
# from pill_transforms__1_ import get_train_transforms, get_valid_transforms

train_tf = get_train_transforms(image_size=640)
valid_tf = get_valid_transforms(image_size=640)
```

---

## 5. 전체 실행 순서 요약 (Colab 기준)

1. Drive 마운트 + 한글 폰트 설치 + `pip install ultralytics albumentations`
2. `02_crop_dataset.ipynb` 실행 → 라벨/박스 검수
3. `pill_detection_dataset.ipynb` 실행
   - Windows 폰트 경로 수정
   - `PillDetectionDataset` 정의 셀에 `%%writefile pill_dataset.py` 추가 후 실행
   - YOLO 데이터셋(`pill_yolo_dataset/`) 생성 확인
4. (있다면) `01_eda.ipynb` 실행 → `splits.json`, `class_attributes.json` 생성
5. `02_baseline.ipynb` 실행
   - `DATA_ROOT`를 Drive 경로로 수정
   - `pill_transforms.py`(또는 `__1_` 버전) 같은 폴더에 위치시키기
   - epochs=10 스모크 테스트로 파이프라인이 끝까지 도는지 확인
6. `03_train.ipynb`로 본 학습(`EPOCHS=100`) 및 실험 반복
   - 실험마다 `EXP_NAME`, `NOTE`를 바꿔가며 `experiments/log.jsonl`에 기록 누적

각 노트북 상단의 마크다운 셀에 더 자세한 배경 설명(가중치 계산식, 정책 표, 다음 단계 가이드)이
있으니 실행 전에 한 번씩 읽어보시길 권장합니다.
