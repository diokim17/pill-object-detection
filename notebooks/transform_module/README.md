# pill_transforms.py

경구약제(알약) Object Detection 프로젝트용 전처리/증강(transform) 모듈입니다.
`01_eda.ipynb`, `02_crop_dataset.ipynb`, `pill_detection_dataset.ipynb`에서
확인한 데이터셋 특성(원본 976×1280 해상도, 작은 bbox 비중, 동일한 배경/조명,
70/75/90도 촬영 각도, 클래스 불균형 등)을 반영해 설계했습니다.

`PillDetectionDataset._apply_transforms`가 기대하는 Albumentations 스타일
호출(`transform(image=..., bboxes=..., labels=...)` → `dict(image, bboxes, labels)`)과
그대로 호환됩니다.

---

## 설치

```bash
pip install albumentations torch
```

Colab에서는 셀에 아래를 먼저 실행하세요.

```python
!pip install -q albumentations
```

> ⚠️ Windows/Anaconda 환경에서 `opencv` 관련 권한 오류(`WinError 5`)가 나면,
> `pip` 대신 `conda install -c conda-forge opencv`로 먼저 설치한 뒤
> `pip install albumentations`를 실행하세요.

---

## 빠른 시작

```python
from pill_transforms import get_train_transforms, get_valid_transforms

train_tf = get_train_transforms(image_size=640)
valid_tf = get_valid_transforms(image_size=640)

dataset = PillDetectionDataset(
    root=dataset_root,
    transforms=train_tf,   # 또는 valid_tf
    label_offset=1,
    strict=False,
    validate_image_size=True,
)
```

Colab에서 Google Drive에 올려둔 파일을 쓰는 경우:

```python
from google.colab import drive
drive.mount("/content/drive")

import sys
sys.path.append("/content/drive/MyDrive/코드잇_파트2_3팀_프로젝트")

from pill_transforms import get_train_transforms, get_valid_transforms
```

---

## API 목록

| 함수/클래스 | 용도 |
|---|---|
| [`get_train_transforms()`](#get_train_transforms) | 학습용 증강 파이프라인 |
| [`get_valid_transforms()`](#get_valid_transforms) | 검증용 파이프라인 (증강 없음) |
| [`get_test_transforms()`](#get_test_transforms) | 추론용, `get_valid_transforms()`의 별칭 |
| [`SafeAlbumentationsTransform`](#safealbumentationstransform) | bbox 전부 소실 방지 wrapper |
| [`denormalize()`](#denormalize) | 시각화용 역정규화 |
| `_bbox_params()` | 내부 헬퍼(직접 호출할 필요 없음) |

---

### `get_train_transforms()`

```python
get_train_transforms(
    image_size: int = 640,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    min_visibility: float = 0.2,
    to_tensor: bool = True,
)
```

학습 단계에서 쓰는 이미지+bbox 동시 증강 파이프라인을 반환합니다.

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `image_size` | `640` | 최종 정사각형 리사이즈 크기(px). 원본이 976×1280이므로 작은 bbox 보존을 위해 `896`~`1024` 상향도 고려해보세요. |
| `mean`, `std` | ImageNet 값 | 정규화(Normalize)에 쓰는 채널별 평균/표준편차 |
| `min_visibility` | `0.2` | 증강 후 원래 면적의 이 비율 미만으로 잘리면 해당 bbox 제거 |
| `to_tensor` | `True` | `True`: Normalize + PyTorch 텐서 변환까지 수행 / `False`: numpy 이미지로 반환(시각화용) |

**포함된 증강 순서**

| 순서 | 증강 | 기본 설정 | 목적 |
|---|---|---|---|
| 1 | `RandomSizedBBoxSafeCrop` | `erosion_rate=0.1`, `p=0.5` | bbox 보존하며 스케일 변화 학습 |
| 2 | `LongestMaxSize` + `PadIfNeeded` | `image_size` 기준 | 비율 왜곡 없이 정사각 캔버스로 리사이즈 |
| 3 | `HorizontalFlip` | `p=0.0` (현재 비활성) | 좌우 반전 |
| 4 | `Rotate` | `limit=15`, `p=0.4` | 70/75/90도 촬영 각도 변화 모사(각인 훼손 방지 위해 소각도만) |
| 5 | `OneOf`: BrightnessContrast / HueSaturationValue / CLAHE | `p=0.6` | 항상 동일했던 배경·조명 한계 보완 |
| 6 | `OneOf`: GaussNoise / MotionBlur | `p=0.2` | 실제 촬영 환경 노이즈 대비 |
| 7 | `Normalize` + `ToTensorV2` | `to_tensor=True`일 때만 | 모델 입력 형태로 변환 |

**반환값**: `albumentations.Compose` 객체

---

### `get_valid_transforms()`

```python
get_valid_transforms(
    image_size: int = 640,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    to_tensor: bool = True,
)
```

검증/추론용 파이프라인. **랜덤 증강 없이** 리사이즈 + 정규화만 수행하여 결과가 항상 동일합니다. bbox는 절대 제거되지 않도록 `min_visibility=0.0`, `min_area=0.0`로 고정되어 있습니다.

Test 데이터(842장)도 Train과 동일하게 976×1280 해상도이므로 별도 크기 보정 없이 그대로 사용 가능합니다.

---

### `get_test_transforms()`

```python
get_test_transforms(image_size=640, mean=IMAGENET_MEAN, std=IMAGENET_STD, to_tensor=True)
```

`get_valid_transforms()`를 그대로 호출하는 별칭 함수입니다. bbox 정답이 없는 추론 상황에서도 `labels=[]`를 함께 넘기면 동일하게 사용할 수 있습니다.

---

### `SafeAlbumentationsTransform`

```python
SafeAlbumentationsTransform(transform: A.Compose, max_retries: int = 3)
```

`RandomSizedBBoxSafeCrop`, `Rotate` 같은 기하 증강은 드물게 bbox를 전부 없앨 수 있습니다. 이 wrapper는 그런 경우 `max_retries`만큼 재시도합니다.

```python
from pill_transforms import get_train_transforms, SafeAlbumentationsTransform

safe_train_tf = SafeAlbumentationsTransform(
    get_train_transforms(image_size=640),
    max_retries=3,
)

dataset = PillDetectionDataset(root=dataset_root, transforms=safe_train_tf)
```

| 파라미터 | 설명 |
|---|---|
| `transform` | 감쌀 대상 파이프라인 (`get_train_transforms()`의 반환값) |
| `max_retries` | 재시도 최대 횟수. 다 실패하면 마지막 결과(빈 bbox일 수 있음)를 그대로 반환 |

호출 시그니처: `__call__(image, bboxes, labels)` → `dict(image, bboxes, labels, ...)`

---

### `denormalize()`

```python
denormalize(tensor: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray
```

`Normalize`로 정규화된 `(C, H, W)` 텐서를 시각화 가능한 `(H, W, C)` `uint8` 배열로 되돌립니다.

```python
import matplotlib.pyplot as plt
from pill_transforms import denormalize

image_tensor, target, metadata = dataset[0]
plt.imshow(denormalize(image_tensor))
plt.show()
```

---

## 자주 하는 실수 / FAQ

**Q. `image_size=640`이 원본(976×1280)보다 훨씬 작은데 괜찮나요?**
A. pill bbox 자체가 작은 객체가 많아(EDA 기준 area_ratio 평균 5.6%) 640으로 줄이면 작은 bbox가 더 작아져 검출이 어려워질 수 있습니다. 정확도가 중요하다면 `896`이나 `1024`(32의 배수)로 올려보고 실제 val mAP로 비교하는 것을 권장합니다.

**Q. `HorizontalFlip`이 `p=0.0`인데 왜 있나요?**
A. 현재 비활성화된 상태로 남아있습니다. 필요하면 `pill_transforms.py`에서 `p` 값을 조정하세요.

**Q. `_bbox_params`를 직접 import해서 써도 되나요?**
A. 이름 앞 `_`는 "내부 전용" 관례 표시입니다. import 자체는 가능하지만, `get_train_transforms()` / `get_valid_transforms()`가 내부적으로 이미 호출하고 있어 직접 쓸 일은 거의 없습니다.

**Q. 클래스 불균형(최대 153건 vs 최소 3건)은 이 모듈로 해결되나요?**
A. 아니요. transform은 기하/색상 변형만 담당하며, 클래스 불균형은 `WeightedRandomSampler` 등 Dataset/Sampler 단계에서 별도로 다뤄야 합니다.

---

## 요구 사항

- Python 3.8+
- `albumentations` (2.x 기준 작성/테스트; 1.x 사용 시 `fill` → `value` 파라미터명 차이 주의)
- `torch` (텐서 변환 및 `denormalize` 사용 시 필요)
- `numpy`
