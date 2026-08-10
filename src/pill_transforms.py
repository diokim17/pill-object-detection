"""
pill_transforms.py
===================

경구약제(알약) Object Detection 프로젝트용 전처리/증강(transform) 모듈입니다.

`01_eda.ipynb`, `02_crop_dataset.ipynb`, `pill_detection_dataset.ipynb`에서
확인한 데이터셋 특성을 반영하여 설계하였습니다.

반영한 EDA/Dataset 특성
------------------------
1. 모든 원본 이미지 해상도는 976 x 1280 (세로가 긴 형태)으로 동일함
   -> 종횡비를 크게 왜곡하지 않는 Resize(+ pad) 전략 사용
2. Bounding Box는 대부분 정사각형에 가까운 형태(Aspect Ratio 0.8~1.2, 약 64%)이며
   평균 area_ratio는 약 5.6%로 이미지 대비 작은 객체가 많음
   -> 과도한 축소를 피하고, 작은 객체 보존을 위해 RandomSizedBBoxSafeCrop 등에서
      erosion_rate를 낮게 설정
3. 배경(back_color)과 조명(light_color)은 전체 샘플에서 항상 동일(연회색 배경,
   주백색 조명)했음 -> 실제 배포 환경 일반화를 위해 색상/조명 변화(Brightness,
   Contrast, HueSaturationValue, CLAHE)를 증강에 반드시 포함
4. 촬영 각도(camera_la)가 70/75/90도 세 가지로 존재 -> 작은 각도의 회전/원근
   변화에는 강건해야 하므로 소폭의 Rotate/Affine을 사용하되, 각인(print_front/
   print_back) 식별성을 해치지 않도록 큰 각도 회전이나 과도한 왜곡은 배제
5. 클래스 불균형이 큼(최대 153건 vs 최소 3건, 약 51배 차이)
   -> transform만으로 해결할 수는 없지만, 소수 클래스가 상대적으로 더 큰 이득을
      보도록 기하 증강(flip/rotate/scale)의 강도를 다소 높게 설정. 실제 클래스
      균형은 Dataset/Sampler 단계(WeightedRandomSampler 등)에서 추가로 다뤄야 함
6. `PillDetectionDataset._apply_transforms`는 Albumentations 스타일 호출을
   최우선으로 시도함:

       transformed = transforms(image=np.ndarray, bboxes=[[x1,y1,x2,y2], ...],
                                 labels=[label, ...])
       transformed["image"], transformed["bboxes"], transformed.get("labels")

   따라서 이 모듈은 bbox 포맷 "pascal_voc"(x1, y1, x2, y2, 절대좌표)를 사용하는
   `albumentations.Compose`를 생성하는 팩토리 함수 위주로 구성합니다.

핵심 API
--------
- get_train_transforms(image_size=640, ...)  : 학습용 augmentation 파이프라인
- get_valid_transforms(image_size=640, ...)  : 검증/추론용 파이프라인(리사이즈+정규화만)
- get_test_transforms(image_size=640, ...)   : 추론용(검증과 동일하되 별칭 제공)
- SafeAlbumentationsTransform                : bbox가 전부 제거되는 예외 상황을
                                                방지하는 안전한 wrapper
- denormalize(tensor, mean, std)             : 시각화를 위한 역정규화 유틸
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# 코랩에서 반드시 !pip install-q albumentations 입력 엔터 설치 필요.
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "이 모듈은 albumentations 패키지가 필요합니다.\n"
        "설치: pip install albumentations"
    ) from exc


# EDA 결과: 모든 원본 이미지가 동일 배경/조명이므로 ImageNet 사전학습 통계를
# 그대로 쓰기보다, 실제 프로젝트에서는 아래 상수를 데이터셋에서 계산한 값으로
# 교체하는 것을 권장합니다. 기본값은 torchvision/ImageNet 표준값입니다.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# PillDetectionDataset이 기대하는 bbox 포맷
BBOX_FORMAT = "pascal_voc"  # [x1, y1, x2, y2], 절대 픽셀 좌표
LABEL_FIELDS = ["labels"]


def _bbox_params(min_visibility: float = 0.2, min_area: float = 4.0):

    return A.BboxParams(
        format=BBOX_FORMAT,
        label_fields=LABEL_FIELDS,
        min_visibility=min_visibility,
        min_area=min_area,
        clip=True,
    )


def get_train_transforms(
    image_size: int = 640, #원본976에서 640으로 리사이즈 하여 검출 난이도 상승시킴.
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    min_visibility: float = 0.2,
    to_tensor: bool = True,
):
    """
    학습용 전처리 + 증강 파이프라인을 생성합니다.

    구성 순서
    ---------
    1. RandomSizedBBoxSafeCrop  : bbox를 보존하면서 스케일 변화를 학습
    2. LongestMaxSize + PadIfNeeded : 976x1280 원본 비율을 왜곡 없이 정사각 캔버스에 맞춤
    3. HorizontalFlip           : 좌우 대칭 증강(알약은 좌우 반전에 강건함)
    4. Rotate(소각도)           : 70/75/90도의 촬영 각도 변화를 모사
    5. RandomBrightnessContrast, HueSaturationValue, CLAHE
                                 : EDA에서 확인된 "항상 동일한 배경/조명" 한계를
                                   보완하기 위한 색상·명암 증강
    6. GaussNoise / MotionBlur  : 실제 촬영 환경의 노이즈·흔들림 대비
    7. Normalize + ToTensorV2   : 모델 입력 형태로 변환
    """

    transforms = [
        A.RandomSizedBBoxSafeCrop(
            height=image_size,
            width=image_size,
            erosion_rate=0.1,
            p=0.5,
        ),
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,  # cv2.BORDER_CONSTANT
            fill=0,
        ),
        #상하좌우 반전
        A.HorizontalFlip(p=0.0),
        #회전
        A.Rotate(
            limit=15,
            border_mode=0,
            fill=0,
            p=0.4,
        ),
        A.OneOf(
            [
                #밝기
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0,
                ),
                #컬러
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=15,
                    p=1.0,
                ),
                #그림자
                A.CLAHE(clip_limit=2.0, p=1.0),
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
        transforms += [A.Normalize(mean=mean, std=std), ToTensorV2()]

    return A.Compose(transforms, bbox_params=_bbox_params(min_visibility=min_visibility))


def get_valid_transforms(
    image_size: int = 640,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    to_tensor: bool = True,
):
    """검증/추론용 전처리 파이프라인(증강 없이 리사이즈 + 정규화만 수행).

    Test 데이터셋(842장)도 Train과 동일하게 976x1280 해상도이므로
    별도의 크기 보정 없이 이 파이프라인을 그대로 사용할 수 있습니다.
    """

    transforms = [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,
            fill=0,
        ),
    ]

    if to_tensor:
        transforms += [A.Normalize(mean=mean, std=std), ToTensorV2()]

    return A.Compose(transforms, bbox_params=_bbox_params(min_visibility=0.0, min_area=0.0))


def get_test_transforms(
    image_size: int = 640,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    to_tensor: bool = True,
):
    """`get_valid_transforms`의 별칭입니다. (bbox 정답이 없는 추론 상황에서도
    labels=[] 를 함께 넘기면 동일하게 사용할 수 있습니다.)"""

    return get_valid_transforms(image_size=image_size, mean=mean, std=std, to_tensor=to_tensor)


class SafeAlbumentationsTransform:
    """PillDetectionDataset과 함께 쓸 때 bbox가 전부 사라지는 경우를 방지하는 wrapper.

    RandomSizedBBoxSafeCrop, Rotate 등 기하 변환은 이론상 모든 bbox를
    제거할 수 있습니다. `PillDetectionDataset._apply_transforms`는 이 경우
    빈 target을 그대로 반환하므로, 학습 루프에서 객체가 0개인 샘플이
    나오는 것을 피하고 싶다면 이 wrapper로 감싸 재시도 로직을 추가하세요.
    """

    def __init__(self, transform: A.Compose, max_retries: int = 3):
        self.transform = transform
        self.max_retries = max_retries

    def __call__(
        self,
        image: np.ndarray,
        bboxes: List[Sequence[float]],
        labels: List[int],
    ):
        last_result: Optional[Dict[str, Any]] = None

        for _ in range(self.max_retries):
            result = self.transform(image=image, bboxes=bboxes, labels=labels)
            last_result = result
            if len(result["bboxes"]) > 0 or len(bboxes) == 0:
                return result

        # 재시도 후에도 bbox가 모두 사라졌다면 마지막 결과를 그대로 반환합니다.
        # (호출부에서 빈 target 처리를 담당해야 합니다.)
        return last_result  # type: ignore[return-value]


def denormalize(
    tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
):
    """정규화된 (C, H, W) 텐서를 시각화용 (H, W, C) uint8 배열로 되돌립니다."""

    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("denormalize는 torch.Tensor 입력을 기대합니다.")

    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)

    image = tensor.detach().cpu() * std_t + mean_t
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return (image * 255).round().astype(np.uint8)


#import 할 시 실행되는 코드가 아닙니다.
if __name__ == "__main__":
    dummy_image = (np.random.rand(1280, 976, 3) * 255).astype(np.uint8)
    dummy_bboxes = [[100, 150, 300, 400], [500, 600, 700, 900]]
    dummy_labels = [1, 5]

    train_tf = get_train_transforms(image_size=640)
    valid_tf = get_valid_transforms(image_size=640)

    out_train = train_tf(image=dummy_image, bboxes=dummy_bboxes, labels=dummy_labels)
    out_valid = valid_tf(image=dummy_image, bboxes=dummy_bboxes, labels=dummy_labels)

    print("train image tensor shape:", tuple(out_train["image"].shape))
    print("train bboxes after transform:", out_train["bboxes"])
    print("valid image tensor shape:", tuple(out_valid["image"].shape))
    print("valid bboxes after transform:", out_valid["bboxes"])
