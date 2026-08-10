"""
pill_transforms.py  (albumentations-free 버전)
==============================================

경구약제(알약) Object Detection 프로젝트용 전처리/증강(transform) 모듈입니다.

이 버전은 **albumentations 를 사용하지 않고 `cv2` + `numpy` 만으로** 동일한
증강 파이프라인을 직접 구현했습니다.
(아나콘다 환경에서 `opencv-python-headless` 가 기존 `cv2` 를 덮어쓰며
 충돌하는 문제를 회피하기 위함 — `02_baseline.ipynb` 와 동일한 방침)

호출 규약은 albumentations 와 동일하게 유지했으므로
`PillDetectionDataset._apply_transforms` 를 수정할 필요가 없습니다.

    transformed = transforms(image=np.ndarray,          # (H, W, 3) uint8 RGB
                             bboxes=[[x1, y1, x2, y2], ...],   # pascal_voc, 절대좌표
                             labels=[label, ...])
    transformed["image"], transformed["bboxes"], transformed["labels"]

반영한 EDA/Dataset 특성
------------------------
1. 모든 원본 이미지 해상도는 976 x 1280 (세로가 긴 형태)으로 동일함
   -> 종횡비를 왜곡하지 않는 LongestMaxSize + Pad(letterbox) 전략 사용
2. Bounding Box는 대부분 정사각형에 가까우며(AR 0.8~1.2, 약 64%)
   평균 area_ratio 약 5.6% 로 작은 객체가 많음
   -> 과도한 축소를 피하고, RandomSizedBBoxSafeCrop 의 erosion_rate 를 낮게 설정
3. 배경(back_color)/조명(light_color)이 전 샘플 동일(연회색 배경, 주백색 조명)
   -> 일반화를 위해 Brightness/Contrast, HueSaturationValue, CLAHE 를 반드시 포함
4. 촬영 각도(camera_la)가 70/75/90도 -> 소폭 회전만 사용(각인 식별성 보존)
5. 클래스 불균형(최대 153 vs 최소 3) -> 기하 증강 강도를 다소 높게.
   실제 균형은 Sampler 단계에서 별도 처리 권장

핵심 API
--------
- get_train_transforms(image_size=640, ...)  : 학습용 augmentation 파이프라인
- get_valid_transforms(image_size=640, ...)  : 검증/추론용(리사이즈 + 정규화만)
- get_test_transforms(image_size=640, ...)   : 추론용 별칭
- SafeAlbumentationsTransform / SafeTransform : bbox가 전부 사라지는 상황 방지 wrapper
- denormalize(tensor, mean, std)             : 시각화용 역정규화 유틸

★ 회전 시 박스 계산 — 내접 타원 방식
------------------------------------
회전한 박스를 축정렬 외접 사각형으로 감싸면 길쭉한 알약을 45° 돌렸을 때
박스 면적이 크게 부풉니다. 대신 **박스에 내접한 타원**을 회전시켜 그 외접
사각형을 사용합니다.

    반너비 = sqrt((a·cosθ)² + (b·sinθ)²)      a = w/2
    반높이 = sqrt((a·sinθ)² + (b·cosθ)²)      b = h/2
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# torch 는 ToTensor 단계에서만 필요하므로 지연 import 합니다.
# (torch 없이 to_tensor=False 로도 이 모듈을 쓸 수 있게 하기 위함)


# EDA 결과: 배경/조명이 항상 동일하므로, 실제 프로젝트에서는 아래 상수를
# 데이터셋에서 계산한 값으로 교체하는 것을 권장합니다.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# PillDetectionDataset이 기대하는 bbox 포맷
BBOX_FORMAT = "pascal_voc"  # [x1, y1, x2, y2], 절대 픽셀 좌표
LABEL_FIELDS = ["labels"]


# ---------------------------------------------------------------------------
# bbox 유틸
# ---------------------------------------------------------------------------
def _as_bboxes(bboxes) -> np.ndarray:
    """어떤 형태로 들어오든 (N, 4) float32 배열로 정규화합니다."""
    if bboxes is None:
        return np.zeros((0, 4), dtype=np.float32)
    arr = np.asarray(bboxes, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return arr.reshape(-1, 4)


def _areas(bboxes: np.ndarray) -> np.ndarray:
    if len(bboxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    w = np.maximum(bboxes[:, 2] - bboxes[:, 0], 0.0)
    h = np.maximum(bboxes[:, 3] - bboxes[:, 1], 0.0)
    return w * h


def _clip_and_filter(
    bboxes: np.ndarray,
    labels: List[Any],
    height: int,
    width: int,
    areas_before: Optional[np.ndarray],
    min_visibility: float,
    min_area: float,
) -> Tuple[np.ndarray, List[Any]]:
    """이미지 경계로 clip 한 뒤, 가시성/면적 기준 미달 박스를 제거합니다.

    albumentations 의 BboxParams(min_visibility, min_area, clip=True) 와
    동일한 역할입니다.
    """
    if len(bboxes) == 0:
        return bboxes, list(labels)

    clipped = bboxes.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width)
    clipped[:, 2] = np.clip(clipped[:, 2], 0, width)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height)
    clipped[:, 3] = np.clip(clipped[:, 3], 0, height)

    areas_after = _areas(clipped)
    keep = areas_after >= max(min_area, 1e-6)

    if areas_before is not None and len(areas_before) == len(clipped):
        safe_before = np.maximum(areas_before, 1e-6)
        visibility = areas_after / safe_before
        keep &= visibility >= min_visibility

    keep_idx = np.nonzero(keep)[0]
    kept_labels = [labels[i] for i in keep_idx]
    return clipped[keep_idx], kept_labels


# ---------------------------------------------------------------------------
# Transform 기본 클래스
# ---------------------------------------------------------------------------
class BasicTransform:
    """모든 변환의 부모 클래스.

    서브클래스는 `apply(image, bboxes)` 를 구현하고 (image, bboxes) 를 반환합니다.
    `is_geometric = True` 인 변환 뒤에는 Compose 가 bbox clip/filter 를 수행합니다.
    """

    is_geometric: bool = False

    def __init__(self, p: float = 1.0):
        self.p = float(p)

    def apply(self, image: np.ndarray, bboxes: np.ndarray):
        raise NotImplementedError

    def __call__(self, image: np.ndarray, bboxes: np.ndarray):
        if self.p <= 0.0:
            return image, bboxes
        if self.p < 1.0 and random.random() >= self.p:
            return image, bboxes
        return self.apply(image, bboxes)


class OneOf(BasicTransform):
    """확률 p 로 내부 변환 중 하나를 (가중치에 따라) 선택해 적용합니다."""

    def __init__(self, transforms: Sequence[BasicTransform], p: float = 0.5):
        super().__init__(p=p)
        self.transforms = list(transforms)
        self.is_geometric = any(getattr(t, "is_geometric", False) for t in self.transforms)

    def apply(self, image, bboxes):
        if not self.transforms:
            return image, bboxes
        weights = [max(getattr(t, "p", 1.0), 1e-6) for t in self.transforms]
        chosen = random.choices(self.transforms, weights=weights, k=1)[0]
        # 선택된 변환은 무조건 적용 (albumentations OneOf 와 동일)
        return chosen.apply(image, bboxes)


# ---------------------------------------------------------------------------
# 기하 변환
# ---------------------------------------------------------------------------
class RandomSizedBBoxSafeCrop(BasicTransform):
    """모든 bbox 를 (erosion_rate 만큼 허용 오차를 두고) 포함하는 영역을
    랜덤 크기로 잘라낸 뒤 (width, height) 로 리사이즈합니다."""

    is_geometric = True

    def __init__(self, height: int, width: int, erosion_rate: float = 0.0, p: float = 0.5):
        super().__init__(p=p)
        self.height = int(height)
        self.width = int(width)
        self.erosion_rate = float(erosion_rate)

    def apply(self, image, bboxes):
        H, W = image.shape[:2]
        aspect = self.width / self.height  # crop_w / crop_h

        if len(bboxes) == 0:
            # 박스가 없으면 단순 랜덤 스케일 크롭
            scale = random.uniform(0.6, 1.0)
            crop_h = min(H, H * scale)
            crop_w = min(W, crop_h * aspect)
            crop_h = min(crop_h, crop_w / aspect)
            x0 = random.uniform(0, max(W - crop_w, 0))
            y0 = random.uniform(0, max(H - crop_h, 0))
        else:
            ux1, uy1 = float(bboxes[:, 0].min()), float(bboxes[:, 1].min())
            ux2, uy2 = float(bboxes[:, 2].max()), float(bboxes[:, 3].max())

            # erosion_rate 만큼 필수 영역을 축소(=박스 일부가 잘려도 허용)
            bw, bh = ux2 - ux1, uy2 - uy1
            ex1 = ux1 + bw * self.erosion_rate * 0.5
            ex2 = ux2 - bw * self.erosion_rate * 0.5
            ey1 = uy1 + bh * self.erosion_rate * 0.5
            ey2 = uy2 - bh * self.erosion_rate * 0.5

            req_w = max(ex2 - ex1, 1.0)
            req_h = max(ey2 - ey1, 1.0)

            min_h = max(req_h, req_w / aspect)
            max_h = min(H, W / aspect)
            if min_h > max_h:  # 필수 영역이 너무 커서 크롭 불가 -> 원본 유지
                return self._resize(image, bboxes, 0.0, 0.0, W, H)

            crop_h = random.uniform(min_h, max_h)
            crop_w = crop_h * aspect

            x0_lo = max(0.0, ex2 - crop_w)
            x0_hi = min(ex1, W - crop_w)
            y0_lo = max(0.0, ey2 - crop_h)
            y0_hi = min(ey1, H - crop_h)
            x0 = random.uniform(x0_lo, x0_hi) if x0_hi > x0_lo else max(x0_lo, 0.0)
            y0 = random.uniform(y0_lo, y0_hi) if y0_hi > y0_lo else max(y0_lo, 0.0)

        return self._resize(image, bboxes, x0, y0, crop_w, crop_h)

    def _resize(self, image, bboxes, x0, y0, crop_w, crop_h):
        H, W = image.shape[:2]
        x0i, y0i = int(round(x0)), int(round(y0))
        x1i = min(W, int(round(x0 + crop_w)))
        y1i = min(H, int(round(y0 + crop_h)))
        x0i, y0i = max(0, min(x0i, x1i - 1)), max(0, min(y0i, y1i - 1))

        cropped = image[y0i:y1i, x0i:x1i]
        out = cv2.resize(cropped, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        if len(bboxes):
            sx = self.width / max(x1i - x0i, 1)
            sy = self.height / max(y1i - y0i, 1)
            bboxes = bboxes.copy()
            bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - x0i) * sx
            bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - y0i) * sy
        return out, bboxes


class LongestMaxSizePad(BasicTransform):
    """LongestMaxSize + PadIfNeeded 를 합친 letterbox 변환.

    긴 변을 max_size 에 맞춰 비율 유지 리사이즈한 뒤, 정사각 캔버스 중앙에
    배치하고 나머지는 fill 값으로 채웁니다. (976x1280 원본의 종횡비 보존)
    """

    is_geometric = True

    def __init__(self, max_size: int, fill: int = 0, p: float = 1.0):
        super().__init__(p=p)
        self.max_size = int(max_size)
        self.fill = int(fill)

    def apply(self, image, bboxes):
        H, W = image.shape[:2]
        scale = self.max_size / max(H, W)
        new_w, new_h = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

        pad_x = (self.max_size - new_w) // 2
        pad_y = (self.max_size - new_h) // 2
        canvas = np.full(
            (self.max_size, self.max_size, image.shape[2]) if image.ndim == 3
            else (self.max_size, self.max_size),
            self.fill,
            dtype=image.dtype,
        )
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        if len(bboxes):
            bboxes = bboxes.copy()
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale + pad_x
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale + pad_y
        return canvas, bboxes


class HorizontalFlip(BasicTransform):
    is_geometric = True

    def apply(self, image, bboxes):
        W = image.shape[1]
        image = np.ascontiguousarray(image[:, ::-1])
        if len(bboxes):
            bboxes = bboxes.copy()
            x1 = W - bboxes[:, 2]
            x2 = W - bboxes[:, 0]
            bboxes[:, 0], bboxes[:, 2] = x1, x2
        return image, bboxes


class VerticalFlip(BasicTransform):
    is_geometric = True

    def apply(self, image, bboxes):
        H = image.shape[0]
        image = np.ascontiguousarray(image[::-1])
        if len(bboxes):
            bboxes = bboxes.copy()
            y1 = H - bboxes[:, 3]
            y2 = H - bboxes[:, 1]
            bboxes[:, 1], bboxes[:, 3] = y1, y2
        return image, bboxes


class Rotate(BasicTransform):
    """중심 기준 소각도 회전. bbox 는 내접 타원 방식으로 재계산합니다."""

    is_geometric = True

    def __init__(self, limit: float = 15.0, fill: int = 0, p: float = 0.4):
        super().__init__(p=p)
        self.limit = float(limit)
        self.fill = int(fill)

    def apply(self, image, bboxes):
        H, W = image.shape[:2]
        angle = random.uniform(-self.limit, self.limit)
        cx, cy = W / 2.0, H / 2.0

        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(self.fill,) * (image.shape[2] if image.ndim == 3 else 1),
        )

        if len(bboxes) == 0:
            return rotated, bboxes

        theta = math.radians(angle)
        cos_t, sin_t = abs(math.cos(theta)), abs(math.sin(theta))

        bcx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bcy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        a = (bboxes[:, 2] - bboxes[:, 0]) / 2.0   # 반너비
        b = (bboxes[:, 3] - bboxes[:, 1]) / 2.0   # 반높이

        # 중심 좌표 회전 (cv2 회전행렬 그대로 사용)
        ncx = M[0, 0] * bcx + M[0, 1] * bcy + M[0, 2]
        ncy = M[1, 0] * bcx + M[1, 1] * bcy + M[1, 2]

        # 내접 타원의 축정렬 외접 사각형
        half_w = np.sqrt((a * cos_t) ** 2 + (b * sin_t) ** 2)
        half_h = np.sqrt((a * sin_t) ** 2 + (b * cos_t) ** 2)

        out = np.stack([ncx - half_w, ncy - half_h, ncx + half_w, ncy + half_h], axis=1)
        return rotated, out.astype(np.float32)


# ---------------------------------------------------------------------------
# 색상 / 노이즈 변환 (bbox 불변)
# ---------------------------------------------------------------------------
class RandomBrightnessContrast(BasicTransform):
    """밝기/대비 변화. albumentations 와 동일한 식: img * (1+c) + b*255"""

    def __init__(self, brightness_limit: float = 0.2, contrast_limit: float = 0.2, p: float = 1.0):
        super().__init__(p=p)
        self.brightness_limit = float(brightness_limit)
        self.contrast_limit = float(contrast_limit)

    def apply(self, image, bboxes):
        beta = random.uniform(-self.brightness_limit, self.brightness_limit)
        alpha = 1.0 + random.uniform(-self.contrast_limit, self.contrast_limit)
        out = image.astype(np.float32) * alpha + beta * 255.0
        return np.clip(out, 0, 255).astype(np.uint8), bboxes


class HueSaturationValue(BasicTransform):
    """HSV 공간에서 색조/채도/명도를 shift 합니다. (OpenCV hue 범위 0~179)"""

    def __init__(
        self,
        hue_shift_limit: int = 10,
        sat_shift_limit: int = 20,
        val_shift_limit: int = 15,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.hue_shift_limit = int(hue_shift_limit)
        self.sat_shift_limit = int(sat_shift_limit)
        self.val_shift_limit = int(val_shift_limit)

    def apply(self, image, bboxes):
        if image.ndim != 3 or image.shape[2] != 3:
            return image, bboxes

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
        # albumentations 의 hue_shift_limit 은 0~255 스케일 기준이므로 179 스케일로 환산
        hue_shift = random.randint(-self.hue_shift_limit, self.hue_shift_limit)
        hue_shift = int(round(hue_shift * 180.0 / 255.0))
        sat_shift = random.randint(-self.sat_shift_limit, self.sat_shift_limit)
        val_shift = random.randint(-self.val_shift_limit, self.val_shift_limit)

        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat_shift, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + val_shift, 0, 255)

        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return out, bboxes


class CLAHE(BasicTransform):
    """Lab 의 L 채널에만 CLAHE 를 적용해 국소 명암을 평탄화합니다.
    (알약과 그림자의 대비를 벌리는 효과 — 02_baseline.ipynb 와 동일 방식)"""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8, p: float = 1.0):
        super().__init__(p=p)
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = int(tile_grid_size)

    def apply(self, image, bboxes):
        if image.ndim != 3 or image.shape[2] != 3:
            return image, bboxes
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_grid_size, self.tile_grid_size),
        )
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        return out, bboxes


class GaussNoise(BasicTransform):
    """가우시안 노이즈 추가. var_limit 는 분산 범위입니다."""

    def __init__(self, var_limit: Tuple[float, float] = (10.0, 50.0), p: float = 1.0):
        super().__init__(p=p)
        self.var_limit = var_limit

    def apply(self, image, bboxes):
        var = random.uniform(*self.var_limit)
        sigma = math.sqrt(var)
        noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
        out = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return out, bboxes


class MotionBlur(BasicTransform):
    """임의 방향의 선형 커널로 모션 블러를 적용합니다."""

    def __init__(self, blur_limit: int = 3, p: float = 1.0):
        super().__init__(p=p)
        self.blur_limit = max(3, int(blur_limit))

    def apply(self, image, bboxes):
        k = random.randrange(3, self.blur_limit + 1, 2) if self.blur_limit >= 5 else 3
        kernel = np.zeros((k, k), dtype=np.float32)
        angle = random.uniform(0, 180)
        c = (k - 1) / 2.0
        dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        p1 = (int(round(c - dx * c)), int(round(c - dy * c)))
        p2 = (int(round(c + dx * c)), int(round(c + dy * c)))
        cv2.line(kernel, p1, p2, 1.0, thickness=1)
        s = kernel.sum()
        if s <= 0:
            return image, bboxes
        kernel /= s
        return cv2.filter2D(image, -1, kernel), bboxes


# ---------------------------------------------------------------------------
# Normalize / ToTensor
# ---------------------------------------------------------------------------
class Normalize(BasicTransform):
    def __init__(self, mean: Sequence[float], std: Sequence[float], p: float = 1.0):
        super().__init__(p=p)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, -1)

    def apply(self, image, bboxes):
        out = image.astype(np.float32) / 255.0
        out = (out - self.mean) / self.std
        return out, bboxes


class ToTensor(BasicTransform):
    """(H, W, C) ndarray -> (C, H, W) torch.Tensor (albumentations 의 ToTensorV2)"""

    def apply(self, image, bboxes):
        import torch  # 지연 import

        arr = image
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        return tensor, bboxes


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------
class Compose:
    """albumentations.Compose 와 동일한 호출 규약을 갖는 경량 구현."""

    def __init__(
        self,
        transforms: Sequence[BasicTransform],
        min_visibility: float = 0.2,
        min_area: float = 4.0,
    ):
        self.transforms = list(transforms)
        self.min_visibility = float(min_visibility)
        self.min_area = float(min_area)

    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[Sequence[Sequence[float]]] = None,
        labels: Optional[Sequence[Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        img = np.asarray(image)
        boxes = _as_bboxes(bboxes)
        lbls: List[Any] = list(labels) if labels is not None else [0] * len(boxes)

        if len(lbls) != len(boxes):
            raise ValueError(
                f"bboxes({len(boxes)})와 labels({len(lbls)})의 길이가 다릅니다."
            )

        for t in self.transforms:
            geometric = getattr(t, "is_geometric", False)
            areas_before = _areas(boxes) if geometric else None
            img, boxes = t(img, boxes)
            if geometric and len(boxes):
                h, w = img.shape[:2]
                boxes, lbls = _clip_and_filter(
                    boxes, lbls, h, w, areas_before, self.min_visibility, self.min_area
                )

        return {
            "image": img,
            "bboxes": [[float(v) for v in box] for box in boxes],
            "labels": lbls,
        }


# ---------------------------------------------------------------------------
# 팩토리 함수 (기존 API 유지)
# ---------------------------------------------------------------------------
def get_train_transforms(
    image_size: int = 640,  # 원본 976 -> 640 리사이즈 (검출 난이도 상승)
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    min_visibility: float = 0.2,
    to_tensor: bool = True,
    hflip_p: float = 0.0,   # 각인 방향 보존을 위해 기본 비활성 (필요 시 0.5 권장)
    vflip_p: float = 0.0,
):
    """
    학습용 전처리 + 증강 파이프라인을 생성합니다.

    구성 순서
    ---------
    1. RandomSizedBBoxSafeCrop        : bbox를 보존하면서 스케일 변화를 학습
    2. LongestMaxSizePad              : 976x1280 비율을 왜곡 없이 정사각 캔버스에 맞춤
    3. HorizontalFlip / VerticalFlip  : 좌우/상하 대칭 증강 (기본 비활성)
    4. Rotate(소각도)                 : 70/75/90도 촬영 각도 변화를 모사
    5. OneOf(RandomBrightnessContrast / HueSaturationValue / CLAHE)
                                      : "항상 동일한 배경·조명" 한계를 보완
    6. OneOf(GaussNoise / MotionBlur) : 실제 촬영 환경의 노이즈·흔들림 대비
    7. Normalize + ToTensor           : 모델 입력 형태로 변환
    """

    transforms: List[BasicTransform] = [
        RandomSizedBBoxSafeCrop(
            height=image_size,
            width=image_size,
            erosion_rate=0.1,
            p=0.5,
        ),
        LongestMaxSizePad(max_size=image_size, fill=0),
        # 상하좌우 반전
        HorizontalFlip(p=hflip_p),
        VerticalFlip(p=vflip_p),
        # 회전
        Rotate(limit=15, fill=0, p=0.4),
        OneOf(
            [
                # 밝기
                RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0,
                ),
                # 컬러
                HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=15,
                    p=1.0,
                ),
                # 그림자 / 국소 대비
                CLAHE(clip_limit=2.0, p=1.0),
            ],
            p=0.6,
        ),
        OneOf(
            [
                GaussNoise(p=1.0),
                MotionBlur(blur_limit=3, p=1.0),
            ],
            p=0.2,
        ),
    ]

    if to_tensor:
        transforms += [Normalize(mean=mean, std=std), ToTensor()]

    return Compose(transforms, min_visibility=min_visibility, min_area=4.0)


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

    transforms: List[BasicTransform] = [
        LongestMaxSizePad(max_size=image_size, fill=0),
    ]

    if to_tensor:
        transforms += [Normalize(mean=mean, std=std), ToTensor()]

    return Compose(transforms, min_visibility=0.0, min_area=0.0)


def get_test_transforms(
    image_size: int = 640,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    to_tensor: bool = True,
):
    """`get_valid_transforms`의 별칭입니다. (bbox 정답이 없는 추론 상황에서도
    labels=[] 를 함께 넘기면 동일하게 사용할 수 있습니다.)"""

    return get_valid_transforms(image_size=image_size, mean=mean, std=std, to_tensor=to_tensor)


# ---------------------------------------------------------------------------
# 안전 wrapper
# ---------------------------------------------------------------------------
class SafeTransform:
    """PillDetectionDataset과 함께 쓸 때 bbox가 전부 사라지는 경우를 방지하는 wrapper.

    RandomSizedBBoxSafeCrop, Rotate 등 기하 변환은 이론상 모든 bbox를
    제거할 수 있습니다. 이 wrapper 는 bbox 가 하나도 남지 않으면
    max_retries 횟수만큼 변환을 다시 시도합니다.
    """

    def __init__(self, transform: Compose, max_retries: int = 3):
        self.transform = transform
        self.max_retries = int(max_retries)

    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[List[Sequence[float]]] = None,
        labels: Optional[List[int]] = None,
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

        # 재시도 후에도 bbox가 모두 사라졌다면 마지막 결과를 그대로 반환합니다.
        return last_result  # type: ignore[return-value]


# 기존 코드 호환용 별칭
SafeAlbumentationsTransform = SafeTransform


# ---------------------------------------------------------------------------
# 시각화 유틸
# ---------------------------------------------------------------------------
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


# import 할 시 실행되는 코드가 아닙니다.
if __name__ == "__main__":
    dummy_image = (np.random.rand(1280, 976, 3) * 255).astype(np.uint8)
    dummy_bboxes = [[100, 150, 300, 400], [500, 600, 700, 900]]
    dummy_labels = [1, 5]

    # torch 가 없는 환경에서도 테스트할 수 있도록 to_tensor 를 감지합니다.
    try:
        import torch  # noqa: F401
        use_tensor = True
    except ImportError:
        use_tensor = False
        print("[알림] torch 가 없어 to_tensor=False 로 테스트합니다.")

    train_tf = get_train_transforms(image_size=640, to_tensor=use_tensor)
    valid_tf = get_valid_transforms(image_size=640, to_tensor=use_tensor)

    out_train = train_tf(image=dummy_image, bboxes=dummy_bboxes, labels=dummy_labels)
    out_valid = valid_tf(image=dummy_image, bboxes=dummy_bboxes, labels=dummy_labels)

    print("train image shape:", tuple(out_train["image"].shape))
    print("train bboxes after transform:", out_train["bboxes"])
    print("train labels:", out_train["labels"])
    print("valid image shape:", tuple(out_valid["image"].shape))
    print("valid bboxes after transform:", out_valid["bboxes"])
