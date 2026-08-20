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
   ★ 이 "항상 동일한 촬영 조건" 문제를 더 강하게 깨기 위해 GAN 스타일 증강을
     추가했습니다(아래 GAN 섹션 참고).
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

★ 2026-08 개편 요약 (팀 공지)
----------------------------
1. 증강 효과를 **확률(p)이 아니라 스위치(True/False)로 켜고 끕니다.**
   예전: `A.Rotate(p=0.4)` → 40% 확률로 걸림(몇 장에 걸렸는지 알 수 없음)
   지금: `ROTATE_ENABLE=True` → 만드는 이미지마다 반드시 회전이 들어감
2. 밝기 / 색상 / CLAHE 를 묶던 `A.OneOf`(셋 중 하나만)를 없앴습니다.
   이제 셋을 각각 켜고 끄며, 켠 것은 전부 적용됩니다. 노이즈/블러도 동일.
3. 모든 설정은 노트북(`yolo11s_augmix.ipynb`)의 `AUG_CONFIG` 한 곳에서 조절합니다.
4. 증강 배수(`AUG_PER_IMAGE`)와 결과 통계·검수 이미지 저장을 지원합니다.
   자세한 사용법과 권장/한계값은 **README_AUGMENTATION.md** 를 보세요.

핵심 API
--------
- DEFAULT_AUG_CONFIG                         : 모든 증강 스위치의 기본값 딕셔너리
- merge_aug_config(user_config)              : 기본값 위에 사용자 설정 덮어쓰기
- describe_augmentation(config)              : 현재 설정을 표로 출력(학습 전 확인용)
- build_aug_transform_list(config)           : 설정 → Albumentations 변환 목록
- get_train_transforms(config=..., ...)      : 학습용 augmentation 파이프라인
- get_valid_transforms(image_size=640, ...)  : 검증/추론용 파이프라인(리사이즈+정규화만)
- get_test_transforms(image_size=640, ...)   : 추론용(검증과 동일하되 별칭 제공)
- count_dataset(root) / print_dataset_stats  : split별 이미지·라벨·bbox 개수 세기
- preview_augmentation(...)                  : 몇 장만 뽑아 증강 결과 미리보기
- augment_dataset(src, dst, n_aug=..., ...)  : ★ 오프라인 증강 데이터셋 생성 + 리포트
- SafeAlbumentationsTransform                : bbox가 전부 제거되는 예외 상황을
                                                방지하는 안전한 wrapper
- denormalize(tensor, mean, std)             : 시각화를 위한 역정규화 유틸

GAN 증강 API (★ 신규)
---------------------
- GANStyleTransform      : Albumentations 파이프라인에 끼우는 GAN 변환
                           (Faster R-CNN 등 이 모듈의 transform을 쓰는 경로용)
- gan_augment_dataset(...) : YOLO 데이터셋 폴더에 GAN 증강본을 **미리 만들어 저장**
                           (Ultralytics는 커스텀 transform을 못 받으므로 오프라인 방식)
- check_gan_model(...)   : 생성기가 제대로 로드/동작하는지 1장으로 점검
- describe_gan()         : 현재 GAN 설정을 표로 출력

⚠️ 이 모듈에는 GAN 가중치가 들어 있지 않고, GAN을 학습시키지도 않습니다.
   이미 학습된 image-to-image 생성기(CycleGAN / pix2pix 등)를 ONNX 또는
   TorchScript로 내보낸 파일 경로를 지정해야 동작합니다. 자세한 이유와
   사용법은 README.md 의 "GAN 증강" 절을 보세요.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# 코랩에서 반드시 !pip install -q albumentations 입력 엔터 설치 필요.
try:
    import albumentations as A
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "이 모듈은 albumentations 패키지가 필요합니다.\n"
        "설치: pip install albumentations"
    ) from exc

# ToTensorV2 는 torch 가 있어야 합니다. 오프라인 증강(이미지를 파일로 저장)만
# 할 때는 torch 가 필요 없으므로, 없으면 "쓸 때만" 에러를 내도록 미룹니다.
try:
    from albumentations.pytorch import ToTensorV2
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

    def ToTensorV2(*_a, **_kw):  # type: ignore[misc]
        raise ImportError(
            "to_tensor=True 를 쓰려면 torch 가 필요합니다.\n"
            "오프라인 증강(augment_dataset)만 할 거라면 to_tensor=False 로 두세요."
        )

# albumentations 1.x / 2.x 는 일부 인자 이름이 다릅니다(GaussNoise, fill 등).
# 아래 값으로 분기해 두 버전 모두에서 동작하게 만듭니다.
try:
    _ALBU_MAJOR = int(str(A.__version__).split(".")[0])
except Exception:  # pragma: no cover
    _ALBU_MAJOR = 2


# EDA 결과: 모든 원본 이미지가 동일 배경/조명이므로 ImageNet 사전학습 통계를
# 그대로 쓰기보다, 실제 프로젝트에서는 아래 상수를 데이터셋에서 계산한 값으로
# 교체하는 것을 권장합니다. 기본값은 torchvision/ImageNet 표준값입니다.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# PillDetectionDataset이 기대하는 bbox 포맷
BBOX_FORMAT = "pascal_voc"  # [x1, y1, x2, y2], 절대 픽셀 좌표
LABEL_FIELDS = ["labels"]

PathLike = Union[str, Path]


# ═══════════════════════════════════════════════════════════════════════════
#  ★★★ GAN 스타일 증강 설정 ★★★
# ═══════════════════════════════════════════════════════════════════════════
#  각 값의 의미 · 올리면/내리면 어떻게 되는지 · 권장 범위는 README.md 참고.
#
#  ▸ GAN은 "촬영 스타일"(조명·색감·질감)만 바꾸고 **알약의 위치는 바꾸지 않는**
#    image-to-image 모델이어야 합니다. 출력 크기가 입력과 달라도 내부에서 원래
#    크기로 되돌리므로 bbox 좌표는 그대로 유효합니다.
#  ▸ 알약의 "각인(print)"이 지워지거나 없던 각인이 생기면 그건 라벨 오염입니다.
#    GAN_STRENGTH로 원본과 섞어 그 위험을 낮춥니다.

GAN_MODEL_PATH: Optional[str] = None   # ★ .onnx 또는 TorchScript .pt 경로 (None이면 GAN 미사용)
GAN_BACKEND = "auto"        # "auto" | "onnx" | "torchscript"
GAN_DEVICE = "cpu"          # torchscript일 때만 의미 ("cpu" | "cuda")
GAN_INPUT_SIZE = 512        # 생성기에 넣을 정사각 크기 (속도·VRAM과 직결)
GAN_P = 0.30                # 이미지 1장당 GAN을 적용할 확률
GAN_STRENGTH = 0.50         # 원본과 섞는 비율 (0=원본 그대로, 1=GAN 결과 그대로)
GAN_INPUT_RANGE = "tanh"    # 생성기 입출력 범위: "tanh"(-1~1) | "sigmoid"(0~1)
GAN_CHANNEL_ORDER = "rgb"   # 생성기가 기대하는 채널 순서: "rgb" | "bgr"


# ---------------------------------------------------------------- 한글 경로 I/O
def imread_unicode(path: PathLike) -> Optional[np.ndarray]:
    """한글/공백이 든 경로도 안전하게 읽습니다 (cv2.imread 대체). 반환은 BGR."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: PathLike, img: np.ndarray) -> bool:
    """한글/공백이 든 경로도 안전하게 저장합니다 (cv2.imwrite 대체)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


# ---------------------------------------------------------------- 생성기 래퍼
class GANGenerator:
    """학습된 image-to-image 생성기를 감싸는 얇은 래퍼.

    ONNX(onnxruntime)와 TorchScript(torch.jit)를 모두 지원합니다.
    같은 경로에 대해서는 인스턴스를 재사용하므로(캐시) 매번 다시 로드하지 않습니다.

    Args:
        model_path: .onnx 또는 TorchScript .pt 경로. None이면 전역 GAN_MODEL_PATH.
        size:       생성기에 넣을 정사각 크기. None이면 전역 GAN_INPUT_SIZE.
        backend:    "auto"면 확장자로 판단합니다.
        device:     TorchScript 실행 장치.
        value_range: "tanh"(-1~1) 또는 "sigmoid"(0~1).
        channel_order: 생성기가 기대하는 채널 순서 "rgb" 또는 "bgr".
    """

    _cache: Dict[Tuple, "GANGenerator"] = {}
    _warned: set = set()

    def __init__(
        self,
        model_path: Optional[PathLike] = None,
        size: Optional[int] = None,
        backend: Optional[str] = None,
        device: Optional[str] = None,
        value_range: Optional[str] = None,
        channel_order: Optional[str] = None,
    ):
        self.model_path = str(model_path if model_path is not None else GAN_MODEL_PATH or "")
        self.size = int(size if size is not None else GAN_INPUT_SIZE)
        self.backend = (backend if backend is not None else GAN_BACKEND).lower()
        self.device = (device if device is not None else GAN_DEVICE).lower()
        self.value_range = (value_range if value_range is not None else GAN_INPUT_RANGE).lower()
        self.channel_order = (channel_order if channel_order is not None
                              else GAN_CHANNEL_ORDER).lower()
        self._sess = None
        self._kind: Optional[str] = None

    # ------------------------------------------------------------ 캐시 생성자
    @classmethod
    def get(cls, **kw) -> "GANGenerator":
        """같은 설정이면 이미 로드한 인스턴스를 재사용합니다."""
        key = (
            str(kw.get("model_path") or GAN_MODEL_PATH or ""),
            int(kw.get("size") or GAN_INPUT_SIZE),
            str(kw.get("backend") or GAN_BACKEND),
            str(kw.get("device") or GAN_DEVICE),
            str(kw.get("value_range") or GAN_INPUT_RANGE),
            str(kw.get("channel_order") or GAN_CHANNEL_ORDER),
        )
        if key not in cls._cache:
            cls._cache[key] = cls(**kw)
        return cls._cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        """모델을 바꿔 끼울 때 호출하세요."""
        cls._cache.clear()
        cls._warned.clear()

    # ------------------------------------------------------------ 로드
    @property
    def available(self) -> bool:
        """지금 이 생성기를 쓸 수 있는지."""
        return self._load()

    def _warn_once(self, msg: str) -> None:
        if msg not in GANGenerator._warned:
            print(f"⚠️ [GAN] {msg}")
            GANGenerator._warned.add(msg)

    def _load(self) -> bool:
        if self._sess is not None:
            return True
        if not self.model_path:
            self._warn_once("GAN_MODEL_PATH가 비어 있습니다 — GAN 증강을 건너뜁니다.")
            return False
        if not os.path.exists(self.model_path):
            self._warn_once(f"모델 파일이 없습니다: {self.model_path} — GAN 증강을 건너뜁니다.")
            return False

        kind = self.backend
        if kind == "auto":
            kind = "onnx" if self.model_path.lower().endswith(".onnx") else "torchscript"
        try:
            if kind == "onnx":
                import onnxruntime as ort

                providers = ["CPUExecutionProvider"]
                if self.device.startswith("cuda"):
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self._sess = ort.InferenceSession(self.model_path, providers=providers)
            else:
                import torch

                self._sess = torch.jit.load(self.model_path, map_location=self.device).eval()
            self._kind = kind
            return True
        except Exception as e:
            self._warn_once(f"모델 로드 실패({e}) — GAN 증강을 건너뜁니다.")
            return False

    # ------------------------------------------------------------ 추론
    def _to_model_input(self, rgb: np.ndarray) -> np.ndarray:
        x = rgb.astype(np.float32)
        x = x / 127.5 - 1.0 if self.value_range == "tanh" else x / 255.0
        return np.transpose(x, (2, 0, 1))[None]          # 1 x 3 x H x W

    def _from_model_output(self, y: np.ndarray) -> np.ndarray:
        y = np.squeeze(np.asarray(y), 0)
        if y.ndim == 3 and y.shape[0] in (1, 3):
            y = np.transpose(y, (1, 2, 0))               # CHW -> HWC
        if y.shape[-1] == 1:
            y = np.repeat(y, 3, axis=-1)
        if self.value_range == "tanh":
            y = (np.clip(y, -1.0, 1.0) + 1.0) * 127.5
        else:
            y = np.clip(y, 0.0, 1.0) * 255.0
        return y.astype(np.uint8)

    def __call__(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """BGR 이미지를 받아 GAN을 통과시킨 BGR 이미지를 돌려줍니다.

        실패하면 None을 반환합니다(호출부에서 원본을 그대로 씁니다).
        출력은 **항상 입력과 같은 크기**로 되돌립니다 → bbox 좌표가 유효합니다.
        """
        if not self._load():
            return None

        h, w = img_bgr.shape[:2]
        src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if self.channel_order == "rgb" else img_bgr
        small = cv2.resize(src, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = self._to_model_input(small)

        try:
            if self._kind == "onnx":
                name = self._sess.get_inputs()[0].name
                y = self._sess.run(None, {name: x})[0]
            else:
                import torch

                with torch.no_grad():
                    t = torch.from_numpy(x).to(self.device)
                    y = self._sess(t)
                    if isinstance(y, (list, tuple)):
                        y = y[0]
                    y = y.detach().cpu().numpy()
        except Exception as e:
            self._warn_once(f"추론 실패({e}) — 원본을 사용합니다.")
            return None

        out = self._from_model_output(y)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)   # ★ 원래 크기로 복원
        if self.channel_order == "rgb":
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out


def _blend(orig: np.ndarray, gen: np.ndarray, strength: float) -> np.ndarray:
    """원본과 GAN 결과를 strength 비율로 섞습니다 (각인 손상 완화)."""
    a = float(np.clip(strength, 0.0, 1.0))
    if a >= 1.0:
        return gen
    if a <= 0.0:
        return orig
    return cv2.addWeighted(gen, a, orig, 1.0 - a, 0.0)


def apply_gan(
    img_bgr: np.ndarray,
    strength: Optional[float] = None,
    **gen_kw,
) -> np.ndarray:
    """이미지 1장에 GAN을 적용합니다(확률 판정 없이 무조건). 실패 시 원본 반환."""
    gen = GANGenerator.get(**gen_kw)
    out = gen(img_bgr)
    if out is None:
        return img_bgr
    return _blend(img_bgr, out, GAN_STRENGTH if strength is None else strength)


# ---------------------------------------------------------------- Albumentations 변환
class GANStyleTransform(A.ImageOnlyTransform):
    """Albumentations 파이프라인에 끼우는 GAN 스타일 변환.

    ImageOnlyTransform이므로 **bbox/label을 건드리지 않습니다.**
    (이미지 픽셀만 바꾸고 크기도 그대로 유지하기 때문에 좌표가 안전합니다)

    Albumentations는 RGB를 다루므로 내부에서 BGR로 바꿔 생성기에 넣고 되돌립니다.

    Args:
        model_path: 생성기 경로. None이면 전역 GAN_MODEL_PATH.
        strength:   원본과 섞는 비율. None이면 전역 GAN_STRENGTH.
        p:          적용 확률. None이면 전역 GAN_P.
    """

    def __init__(
        self,
        model_path: Optional[PathLike] = None,
        strength: Optional[float] = None,
        size: Optional[int] = None,
        backend: Optional[str] = None,
        device: Optional[str] = None,
        value_range: Optional[str] = None,
        channel_order: Optional[str] = None,
        p: Optional[float] = None,
    ):
        super().__init__(p=GAN_P if p is None else p)
        self.model_path = model_path
        self.strength = strength
        self.size = size
        self.backend = backend
        self.device = device
        self.value_range = value_range
        self.channel_order = channel_order

    def _gen(self) -> GANGenerator:
        return GANGenerator.get(
            model_path=self.model_path, size=self.size, backend=self.backend,
            device=self.device, value_range=self.value_range,
            channel_order=self.channel_order,
        )

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out = self._gen()(bgr)
        if out is None:
            return img
        blended = _blend(bgr, out, GAN_STRENGTH if self.strength is None else self.strength)
        return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    def get_transform_init_args_names(self):
        return ("model_path", "strength", "size", "backend",
                "device", "value_range", "channel_order")


# ---------------------------------------------------------------- 점검 · 설명
def check_gan_model(
    model_path: Optional[PathLike] = None,
    sample_image: Optional[np.ndarray] = None,
    verbose: bool = True,
    **gen_kw,
) -> bool:
    """생성기가 제대로 로드되고 동작하는지 더미 이미지 1장으로 확인합니다.

    학습을 돌리기 **전에** 반드시 한 번 실행해 보세요. 여기서 실패하면
    학습 중에는 조용히 원본만 쓰게 되어 GAN이 안 걸린 줄도 모릅니다.
    """
    gen = GANGenerator.get(model_path=model_path, **gen_kw)
    if verbose:
        print(f"■ GAN 모델 점검: {gen.model_path or '(경로 없음)'}")
    if not gen.available:
        if verbose:
            print("  ❌ 로드 실패 — 위 경고 메시지를 확인하세요.")
        return False

    img = sample_image if sample_image is not None else \
        (np.random.rand(1280, 976, 3) * 255).astype(np.uint8)
    out = gen(img)
    if out is None:
        if verbose:
            print("  ❌ 추론 실패")
        return False

    same_shape = out.shape == img.shape
    diff = float(np.abs(out.astype(np.float32) - img.astype(np.float32)).mean())
    if verbose:
        print(f"  백엔드      {gen._kind}")
        print(f"  입력 크기   {gen.size} x {gen.size}")
        print(f"  입출력 shape {img.shape} → {out.shape}  {'✅' if same_shape else '❌ 크기 불일치'}")
        print(f"  평균 화소 변화 {diff:.2f}  (0에 가까우면 모델이 사실상 아무것도 안 바꾸는 중)")
        print(f"  값 범위     {out.min()} ~ {out.max()}")
        if diff < 1.0:
            print("  ⚠️ 변화가 거의 없습니다. GAN_INPUT_RANGE / GAN_CHANNEL_ORDER를 확인하세요.")
        print("  ✅ 사용 가능")
    return same_shape


def describe_gan() -> str:
    """현재 GAN 설정을 사람이 읽는 표로."""
    on = bool(GAN_MODEL_PATH)
    lines = [
        "■ GAN 스타일 증강 설정",
        f"   상태          {'ON' if on else 'off (GAN_MODEL_PATH 미지정)'}",
        f"   모델          {GAN_MODEL_PATH or '-'}",
        f"   백엔드/장치   {GAN_BACKEND} / {GAN_DEVICE}",
        f"   입력 크기     {GAN_INPUT_SIZE} x {GAN_INPUT_SIZE}",
        f"   적용 확률 p   {GAN_P}",
        f"   혼합 강도     {GAN_STRENGTH}  (0=원본, 1=GAN 그대로)",
        f"   값 범위       {GAN_INPUT_RANGE}      채널 순서 {GAN_CHANNEL_ORDER}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- YOLO 오프라인 증강
def gan_augment_dataset(
    src_root: PathLike,
    dst_root: Optional[PathLike] = None,
    *,
    split: str = "train",
    ratio: float = 0.5,
    strength: Optional[float] = None,
    keep_original: bool = True,
    prefix: str = "gan_",
    seed: int = 42,
    overwrite: bool = True,
    verbose: bool = True,
    **gen_kw,
) -> str:
    """YOLO 데이터셋에 **GAN 증강본을 미리 만들어 저장**하고 data.yaml 경로를 반환합니다.

    왜 오프라인인가
    ---------------
    Ultralytics(YOLO)는 이 모듈의 Albumentations 파이프라인을 받아 주지 않습니다.
    그래서 학습 전에 GAN을 먹인 이미지를 디스크에 만들어 두고, 그 폴더로 학습합니다.
    **라벨은 그대로 복사**합니다 — GAN은 픽셀만 바꾸고 위치는 안 바꾸기 때문입니다.

    Args:
        src_root: images/{split}, labels/{split}, data.yaml 이 있는 YOLO 폴더
        dst_root: 출력 폴더. None이면 `<src_root>_gan`
        split:    증강할 split (보통 "train")
        ratio:    원본 중 몇 %에 GAN을 적용할지 (0.5 = 절반)
        strength: 원본과 섞는 비율. None이면 전역 GAN_STRENGTH
        keep_original: True면 원본도 함께 복사(권장). False면 GAN본만 남습니다
        prefix:   생성 파일 접두어 (나중에 골라내기 쉽게)

    Returns:
        생성된 data.yaml 의 절대 경로 문자열
    """
    import random as _random

    rng = _random.Random(seed)
    src_root = Path(src_root).resolve()
    dst_root = Path(dst_root).resolve() if dst_root else src_root.parent / f"{src_root.name}_gan"

    img_dir = src_root / "images" / split
    lbl_dir = src_root / "labels" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"{img_dir} 가 없습니다. YOLO 데이터셋 폴더를 확인하세요.")

    gen = GANGenerator.get(**gen_kw)
    if not gen.available:
        raise RuntimeError(
            "GAN 생성기를 쓸 수 없습니다.\n"
            "→ pt.GAN_MODEL_PATH 를 지정하고 pt.check_gan_model() 로 먼저 점검하세요."
        )

    if overwrite and dst_root.exists():
        shutil.rmtree(dst_root)
    (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"{img_dir} 에 이미지가 없습니다.")

    n_target = int(round(len(files) * float(np.clip(ratio, 0.0, 1.0))))
    picked = set(rng.sample(range(len(files)), n_target)) if n_target else set()

    if verbose:
        print("=" * 62)
        print(f"  GAN 증강 데이터셋 생성")
        print(f"  입력 {src_root}")
        print(f"  출력 {dst_root}")
        print(f"  원본 {len(files):,}장 중 {n_target:,}장에 GAN 적용 (ratio={ratio})")
        print(f"  혼합 강도 {GAN_STRENGTH if strength is None else strength}")
        print("=" * 62)

    n_copy = n_gan = n_fail = 0
    for i, f in enumerate(files):
        lbl = lbl_dir / f"{f.stem}.txt"

        if keep_original:
            shutil.copy2(f, dst_root / "images" / split / f.name)
            if lbl.exists():
                shutil.copy2(lbl, dst_root / "labels" / split / lbl.name)
            n_copy += 1

        if i in picked:
            img = imread_unicode(f)
            if img is None:
                n_fail += 1
                continue
            out = gen(img)
            if out is None:
                n_fail += 1
                continue
            out = _blend(img, out, GAN_STRENGTH if strength is None else strength)
            new_stem = f"{prefix}{f.stem}"
            imwrite_unicode(dst_root / "images" / split / f"{new_stem}.png", out)
            if lbl.exists():
                shutil.copy2(lbl, dst_root / "labels" / split / f"{new_stem}.txt")
            n_gan += 1

        if verbose and (i + 1) % 100 == 0:
            print(f"    {i + 1:,}/{len(files):,} 처리")

    # 다른 split은 그대로 복사 (증강 없이)
    for other in ("val", "test"):
        s_img, s_lbl = src_root / "images" / other, src_root / "labels" / other
        if other == split or not s_img.is_dir():
            continue
        shutil.copytree(s_img, dst_root / "images" / other, dirs_exist_ok=True)
        if s_lbl.is_dir():
            shutil.copytree(s_lbl, dst_root / "labels" / other, dirs_exist_ok=True)

    # data.yaml 작성 (원본 것을 읽어 path만 교체)
    yaml_path = dst_root / "data.yaml"
    src_yaml = src_root / "data.yaml"
    try:
        import yaml as _yaml

        cfg = _yaml.safe_load(src_yaml.read_text(encoding="utf-8")) if src_yaml.exists() else {}
        cfg = cfg or {}
        cfg["path"] = str(dst_root)
        cfg["train"] = f"images/{split}"
        if (dst_root / "images" / "val").is_dir():
            cfg["val"] = "images/val"
        else:
            cfg["val"] = f"images/{split}"     # 홀드아웃이 없으면 train을 가리킵니다
        if (dst_root / "images" / "test").is_dir():
            cfg["test"] = "images/test"
        _yaml.safe_dump(cfg, open(yaml_path, "w", encoding="utf-8"),
                        allow_unicode=True, sort_keys=False)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"data.yaml 작성 실패({e}). 원본 data.yaml 을 확인하세요.") from e

    if verbose:
        print(f"\n■ 결과  원본 복사 {n_copy:,}장 + GAN 생성 {n_gan:,}장 "
              f"= {n_copy + n_gan:,}장" + (f"  (실패 {n_fail}장)" if n_fail else ""))
        print(f"★ data.yaml = {yaml_path}")
    return str(yaml_path)


# ═══════════════════════════════════════════════════════════════════════════
#  ★★★ [AUG] Albumentations 오프라인 증강 설정 ★★★
# ═══════════════════════════════════════════════════════════════════════════
#  이 블록의 값들은 **기본값**일 뿐입니다.
#  실제 사용은 `yolo11s_augmix.ipynb` 의 [D] 블록(AUG_CONFIG)에서
#  덮어쓰기 하세요. 노트북 값이 항상 우선합니다.
#
#  ◆ 설계 원칙 (기존 버전과 달라진 점)
#    1) 예전에는 p=0.4 / 0.6 / 0.2 처럼 **확률**로 "걸릴지 말지"를 랜덤하게
#       정했습니다. 그래서 몇 장에 어떤 효과가 들어갔는지 알 수 없었습니다.
#       → 이제는 `*_ENABLE` (True/False) 로 **켜고 끕니다.**
#         켜면 그 효과는 만드는 이미지마다 **반드시** 들어갑니다(p=1.0).
#         (효과의 "세기"는 여전히 지정한 범위 안에서 랜덤합니다. 예를 들어
#          ROTATE_LIMIT=15 면 매번 -15~+15도 중 하나가 뽑힙니다. 이건
#          "증강이 걸릴지 말지"의 랜덤이 아니라 "얼마나 걸릴지"의 랜덤이며,
#          이게 없으면 N장을 만들어도 전부 똑같은 이미지가 나옵니다.)
#    2) 예전에는 밝기 / 컬러 / 그림자(CLAHE) 를 `A.OneOf` 로 묶어
#       **셋 중 하나만** 적용했습니다.
#       → 이제는 셋을 각각 독립적으로 켜고 끕니다. 셋 다 켜면 셋 다 들어갑니다.
#         노이즈 / 블러도 마찬가지로 분리했습니다.
#    3) 그래도 예전처럼 확률로 굴리고 싶다면 `*_P` 값을 1.0 미만으로 주세요.
#       (기본은 전부 1.0 = 결정론적 ON. 초보자는 건드리지 마세요.)
#
#  ◆ 각 값의 의미 · 권장 범위 · 한계치는 README_AUGMENTATION.md 표 참고.
# ---------------------------------------------------------------------------

DEFAULT_AUG_CONFIG: Dict[str, Any] = {
    # ───────────────────────────────────────────────────────────────────
    # [0] 공통
    # ───────────────────────────────────────────────────────────────────
    # 증강 결과물의 한 변 크기(픽셀). RESIZE_ENABLE=True 일 때만 의미가 있습니다.
    #   권장 640 / 960      한계 320 ~ 1280
    #   ⚠️ 알약 각인(글자)을 봐야 하므로 640 미만으로 내리면 성능이 크게 떨어집니다.
    "IMAGE_SIZE": 640,

    # bbox가 잘려나갔을 때 몇 % 이상 남아야 라벨을 유지할지.
    #   권장 0.2 ~ 0.4      한계 0.0 ~ 1.0
    #   ↑ 올리면 잘린 알약 라벨이 잘 버려짐(라벨 품질↑ / 데이터 수↓)
    "MIN_VISIBILITY": 0.2,
    # 남은 bbox의 최소 픽셀 면적. 이보다 작으면 버립니다. 권장 4 ~ 64
    "MIN_AREA": 4.0,

    # ───────────────────────────────────────────────────────────────────
    # [1] RandomSizedBBoxSafeCrop — bbox를 살린 채 잘라내기(스케일 변화 학습)
    # ───────────────────────────────────────────────────────────────────
    #   ⚠️ 두 가지 이유로 **기본 OFF** 입니다.
    #      (1) 알약의 "실제 크기"가 클래스 단서인데 크롭은 그 크기를 바꿉니다.
    #      (2) 정사각형으로 잘라 내보내므로 **종횡비가 찌그러집니다.**
    #          → 동그란 알약이 타원이 되어 shape 정보가 오염됩니다.
    #      스케일 다양성이 꼭 필요할 때만 켜고, 켠 뒤 검수 이미지를 꼭 보세요.
    "CROP_ENABLE": False,
    "CROP_SIZE": None,        # None이면 IMAGE_SIZE 사용. 정수로 직접 지정 가능
    "CROP_EROSION_RATE": 0.1,  # 권장 0.0 ~ 0.2 / 한계 0.0 ~ 0.5
                               # ↑ 올리면 더 과감하게 잘라냄(작은 객체 소실 위험↑)
    "CROP_P": 1.0,

    # ───────────────────────────────────────────────────────────────────
    # [2] LongestMaxSize + PadIfNeeded — 비율 유지 리사이즈 + 검은 여백
    # ───────────────────────────────────────────────────────────────────
    #   ⚠️ YOLO(Ultralytics)는 학습할 때 자체적으로 letterbox 를 또 합니다.
    #      그래서 **오프라인 증강에서는 이걸 끄고(False) 원본 해상도를
    #      유지하는 쪽을 권장**합니다. (두 번 리사이즈 = 각인 뭉개짐)
    #      Faster R-CNN 처럼 이 모듈의 transform 을 직접 쓰는 경로에서는 True.
    "RESIZE_ENABLE": False,
    "PAD_VALUE": 0,            # 여백 색 (0=검정). 배경이 연회색이므로 114도 무난

    # ───────────────────────────────────────────────────────────────────
    # [3] 좌우 반전 / 상하 반전
    # ───────────────────────────────────────────────────────────────────
    #   ★ 알약 각인(글자/숫자)이 거울상이 되어 **없는 약처럼 보입니다.**
    #      이 프로젝트에서는 둘 다 OFF 를 강력 권장합니다.
    "HFLIP_ENABLE": False,
    "HFLIP_P": 1.0,
    "VFLIP_ENABLE": False,
    "VFLIP_P": 1.0,

    # ───────────────────────────────────────────────────────────────────
    # [4] Rotate — 회전
    # ───────────────────────────────────────────────────────────────────
    #   촬영 각도(camera_la 70/75/90도) 변화를 흉내냅니다.
    #   권장 10 ~ 20도      한계 0 ~ 45도
    #   ⚠️ 45도를 넘기면 bbox가 실제 알약보다 훨씬 커져(축 정렬 박스의 한계)
    #      라벨 품질이 떨어집니다. 알약이 원형이라 회전 불변이라면 180까지도
    #      가능하지만, 이 데이터셋은 캡슐/타원도 있어 15도 근처가 안전합니다.
    "ROTATE_ENABLE": True,
    "ROTATE_LIMIT": 15,
    "ROTATE_P": 1.0,

    # ───────────────────────────────────────────────────────────────────
    # [5] 밝기 / 대비  (RandomBrightnessContrast)   ※ 예전 "밝기"
    # ───────────────────────────────────────────────────────────────────
    #   EDA상 조명이 항상 동일(주백색)했으므로 실제 환경 일반화에 가장 중요.
    #   권장 0.15 ~ 0.30    한계 0.0 ~ 0.5
    #   ⚠️ 0.5를 넘기면 흰 알약이 배경에 묻히거나 검은 알약이 뭉개집니다.
    "BRIGHTNESS_ENABLE": True,
    "BRIGHTNESS_LIMIT": 0.2,
    "CONTRAST_LIMIT": 0.2,
    "BRIGHTNESS_P": 1.0,

    # ───────────────────────────────────────────────────────────────────
    # [6] 색상 / 채도  (HueSaturationValue)         ※ 예전 "컬러"
    # ───────────────────────────────────────────────────────────────────
    #   ⚠️⚠️ 알약의 **색(color1)이 곧 클래스 정보**입니다.
    #        HUE를 크게 흔들면 노란 약이 초록 약이 되어 라벨이 거짓말이 됩니다.
    #   HUE  권장 5 ~ 10    한계 0 ~ 15   (그 이상 절대 비권장)
    #   SAT  권장 10 ~ 25   한계 0 ~ 40
    #   VAL  권장 10 ~ 20   한계 0 ~ 40
    "HSV_ENABLE": True,
    "HUE_SHIFT_LIMIT": 10,
    "SAT_SHIFT_LIMIT": 20,
    "VAL_SHIFT_LIMIT": 15,
    "HSV_P": 1.0,

    # ───────────────────────────────────────────────────────────────────
    # [7] CLAHE — 국소 대비 평탄화                  ※ 예전 "그림자"
    # ───────────────────────────────────────────────────────────────────
    #   그림자로 어두워진 부분을 살려 각인을 또렷하게 만듭니다.
    #   권장 2.0 ~ 3.0      한계 1.0 ~ 4.0
    #   ⚠️ 4.0을 넘기면 노이즈까지 증폭되어 알약 표면이 지저분해집니다.
    "CLAHE_ENABLE": True,
    "CLAHE_CLIP_LIMIT": 2.0,
    "CLAHE_TILE_GRID": 8,      # 격자 수. 권장 8 (건드릴 일 거의 없음)
    "CLAHE_P": 1.0,

    # ───────────────────────────────────────────────────────────────────
    # [8] GaussNoise — 가우시안 노이즈
    # ───────────────────────────────────────────────────────────────────
    #   저조도 촬영의 센서 노이즈를 흉내냅니다.
    #   NOISE_STD 는 0~1로 정규화된 표준편차입니다.
    #   권장 0.02 ~ 0.10    한계 0.0 ~ 0.20
    #   ⚠️ 0.2를 넘기면 각인이 노이즈에 묻혀 사람 눈으로도 안 보입니다.
    "NOISE_ENABLE": True,
    "NOISE_STD_MIN": 0.03,
    "NOISE_STD_MAX": 0.08,
    "NOISE_P": 1.0,

    # ───────────────────────────────────────────────────────────────────
    # [9] MotionBlur — 흔들림 블러
    # ───────────────────────────────────────────────────────────────────
    #   손떨림·움직임을 흉내냅니다. 커널 크기이며 **홀수**여야 합니다.
    #   권장 3 ~ 5          한계 3 ~ 9
    #   ⚠️ 7 이상이면 각인이 사실상 사라집니다. 알약 프로젝트에서는 3 권장.
    "BLUR_ENABLE": True,
    "BLUR_LIMIT": 3,
    "BLUR_P": 1.0,

    # ───────────────────────────────────────────────────────────────────
    # [10] GAN 스타일 증강 (선택)
    # ───────────────────────────────────────────────────────────────────
    #   학습된 image-to-image 생성기가 있을 때만 동작합니다. 자세한 건
    #   README_AUGMENTATION.md 의 "GAN 증강" 절 참고.
    "GAN_ENABLE": False,
    "GAN_P": 1.0,
    "GAN_STRENGTH": None,      # None이면 전역 GAN_STRENGTH 사용
    "GAN_MODEL_PATH": None,    # None이면 전역 GAN_MODEL_PATH 사용
}


# 사람이 읽을 설명 표 (describe_augmentation / README 자동 생성에 사용)
AUG_DOC: List[Tuple[str, str, str, str, str]] = [
    # (스위치, 세부 파라미터, 무슨 효과, 권장, 한계/주의)
    ("CROP_ENABLE", "CROP_SIZE, CROP_EROSION_RATE",
     "bbox를 보존하며 랜덤 크롭 → 스케일 변화 학습",
     "OFF (알약 크기가 클래스 단서)", "erosion 0.0~0.5, 크면 작은 알약 소실"),
    ("RESIZE_ENABLE", "IMAGE_SIZE, PAD_VALUE",
     "비율 유지 리사이즈 + 검은 여백 패딩",
     "YOLO 오프라인 증강엔 OFF / Faster R-CNN엔 ON", "320~1280, 640 미만은 각인 소실"),
    ("HFLIP_ENABLE", "HFLIP_P", "좌우 반전", "OFF", "각인 글자가 거울상이 됨"),
    ("VFLIP_ENABLE", "VFLIP_P", "상하 반전", "OFF", "각인 글자가 뒤집힘"),
    ("ROTATE_ENABLE", "ROTATE_LIMIT",
     "±N도 회전 (촬영 각도 변화 모사)", "ON, 10~20도", "0~45도, 넘기면 bbox 부정확"),
    ("BRIGHTNESS_ENABLE", "BRIGHTNESS_LIMIT, CONTRAST_LIMIT",
     "밝기·대비 변화 (조명 일반화)", "ON, 0.15~0.30", "0.0~0.5, 넘기면 알약이 배경에 묻힘"),
    ("HSV_ENABLE", "HUE_SHIFT_LIMIT, SAT_SHIFT_LIMIT, VAL_SHIFT_LIMIT",
     "색상·채도·명도 변화", "ON, HUE 5~10", "HUE 0~15 ⚠️ 색이 곧 클래스"),
    ("CLAHE_ENABLE", "CLAHE_CLIP_LIMIT",
     "국소 대비 평탄화 (그림자 보정)", "ON, 2.0~3.0", "1.0~4.0, 넘기면 노이즈 증폭"),
    ("NOISE_ENABLE", "NOISE_STD_MIN, NOISE_STD_MAX",
     "가우시안 노이즈 (센서 노이즈)", "ON, 0.02~0.10", "0.0~0.20, 넘기면 각인 소실"),
    ("BLUR_ENABLE", "BLUR_LIMIT",
     "모션 블러 (흔들림)", "ON, 3", "3~9(홀수), 7 이상은 각인 소실"),
    ("GAN_ENABLE", "GAN_MODEL_PATH, GAN_STRENGTH",
     "GAN 스타일 변환 (촬영 환경 통째로 변경)", "모델 있을 때만 ON", "STRENGTH 0.3~0.6"),
]


def merge_aug_config(user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """DEFAULT_AUG_CONFIG 위에 사용자 설정을 덮어씁니다.

    노트북에서는 바꾸고 싶은 키만 적으면 되고, 나머지는 기본값이 채워집니다.
    오타로 없는 키를 넣으면 경고를 띄웁니다(조용히 무시되면 디버깅이 지옥이라서).
    """
    cfg = dict(DEFAULT_AUG_CONFIG)
    if user_config:
        unknown = [k for k in user_config if k not in DEFAULT_AUG_CONFIG]
        if unknown:
            print(f"⚠️ [AUG] 모르는 설정 키가 있습니다(오타?): {unknown}")
        cfg.update({k: v for k, v in user_config.items() if k in DEFAULT_AUG_CONFIG})
    return cfg


def _pad(text: str, width: int) -> str:
    """한글은 폭이 2칸이라 f-string 정렬이 깨집니다. 실제 표시 폭으로 맞춥니다."""
    import unicodedata

    w = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(0, width - w)


def describe_augmentation(config: Optional[Dict[str, Any]] = None) -> str:
    """현재 증강 설정을 사람이 읽는 표로 만듭니다. 학습 전에 한 번 출력하세요."""
    cfg = merge_aug_config(config)
    lines = [
        "■ Albumentations 증강 설정 (ON = 만드는 이미지마다 반드시 적용)",
        "-" * 78,
        f"  {_pad('효과', 22)}{_pad('상태', 10)}세부값",
        "-" * 78,
    ]
    rows = [
        ("크롭(BBoxSafeCrop)", "CROP_ENABLE",
         lambda c: f"size={c['CROP_SIZE'] or c['IMAGE_SIZE']}, erosion={c['CROP_EROSION_RATE']}"),
        ("리사이즈+패딩", "RESIZE_ENABLE",
         lambda c: f"size={c['IMAGE_SIZE']}, pad={c['PAD_VALUE']}"),
        ("좌우반전", "HFLIP_ENABLE", lambda c: f"p={c['HFLIP_P']}"),
        ("상하반전", "VFLIP_ENABLE", lambda c: f"p={c['VFLIP_P']}"),
        ("회전", "ROTATE_ENABLE",
         lambda c: f"±{c['ROTATE_LIMIT']}도, p={c['ROTATE_P']}"),
        ("밝기/대비", "BRIGHTNESS_ENABLE",
         lambda c: f"bright={c['BRIGHTNESS_LIMIT']}, contrast={c['CONTRAST_LIMIT']}, p={c['BRIGHTNESS_P']}"),
        ("색상(HSV)", "HSV_ENABLE",
         lambda c: f"h={c['HUE_SHIFT_LIMIT']}, s={c['SAT_SHIFT_LIMIT']}, v={c['VAL_SHIFT_LIMIT']}, p={c['HSV_P']}"),
        ("그림자(CLAHE)", "CLAHE_ENABLE",
         lambda c: f"clip={c['CLAHE_CLIP_LIMIT']}, p={c['CLAHE_P']}"),
        ("노이즈", "NOISE_ENABLE",
         lambda c: f"std={c['NOISE_STD_MIN']}~{c['NOISE_STD_MAX']}, p={c['NOISE_P']}"),
        ("블러", "BLUR_ENABLE",
         lambda c: f"limit={c['BLUR_LIMIT']}, p={c['BLUR_P']}"),
        ("GAN 스타일", "GAN_ENABLE",
         lambda c: f"strength={c['GAN_STRENGTH'] if c['GAN_STRENGTH'] is not None else GAN_STRENGTH}"),
    ]
    for label, key, detail in rows:
        state = "✅ ON " if cfg[key] else "⬜ off "
        lines.append(f"  {_pad(label, 22)}{_pad(state, 10)}"
                     f"{detail(cfg) if cfg[key] else '-'}")
    lines.append("-" * 78)
    on_count = sum(1 for _, k, _ in rows if cfg[k])
    lines.append(f"  켜진 효과 {on_count}개 / 전체 {len(rows)}개")
    return "\n".join(lines)


# ---------------------------------------------------------------- 버전 호환 헬퍼
def _make(cls, **kwargs):
    """albumentations 1.x / 2.x 인자 이름 차이를 흡수해서 생성합니다.

    (예: 2.x의 `fill=` 이 1.x에서는 `value=`, `std_range=` 가 `var_limit=`)
    한 팀 안에서도 버전이 제각각이라 여기서 한 번에 막아 둡니다.
    """
    alias = {"fill": "value", "fill_value": "value",
             "std_range": "var_limit", "tile_grid_size": "tile_grid_size"}
    try:
        return cls(**kwargs)
    except TypeError:
        kw = {}
        for k, v in kwargs.items():
            if k in alias:
                kw[alias[k]] = v
            else:
                kw[k] = v
        try:
            return cls(**kw)
        except TypeError:
            # 그래도 안 되면 문제되는 인자를 빼고 기본값으로 생성
            base = {k: v for k, v in kwargs.items() if k in ("p", "limit", "blur_limit")}
            print(f"⚠️ [AUG] {cls.__name__} 인자 호환 실패 → 기본값으로 생성합니다.")
            return cls(**base)


# ═══════════════════════════════════════════════════════════════════════════
#  Albumentations 파이프라인
# ═══════════════════════════════════════════════════════════════════════════

def _bbox_params(min_visibility: float = 0.2, min_area: float = 4.0):

    return A.BboxParams(
        format=BBOX_FORMAT,
        label_fields=LABEL_FIELDS,
        min_visibility=min_visibility,
        min_area=min_area,
        clip=True,
    )


def build_aug_transform_list(config: Optional[Dict[str, Any]] = None) -> List[Any]:
    """AUG_CONFIG 를 읽어 Albumentations 변환 **목록**을 만듭니다.

    ★ 이 함수가 이번 개편의 핵심입니다.
      - `*_ENABLE` 이 False 인 효과는 **아예 파이프라인에 들어가지 않습니다.**
        (예전처럼 p=0.0 으로 넣어 두지 않습니다 → 로그·재현성이 명확해집니다)
      - `A.OneOf` 를 쓰지 않습니다. 밝기/색상/CLAHE, 노이즈/블러가
        각각 독립적으로 적용됩니다.

    적용 순서(고정)
    ---------------
      1. 크롭            (기하)
      2. 리사이즈 + 패딩 (기하)
      3. 좌우/상하 반전  (기하)
      4. 회전            (기하)
      5. GAN 스타일      (픽셀)   ← 기하 변환 뒤에 두면 연산량이 적습니다
      6. 밝기/대비       (픽셀)
      7. 색상 HSV        (픽셀)
      8. CLAHE           (픽셀)
      9. 노이즈          (픽셀)
     10. 블러            (픽셀)   ← 노이즈 뒤에 두어야 "흔들린 사진"처럼 자연스럽습니다
    """
    cfg = merge_aug_config(config)
    size = int(cfg["IMAGE_SIZE"])
    tfs: List[Any] = []

    # ── 1. 크롭 ────────────────────────────────────────────────────────
    if cfg["CROP_ENABLE"]:
        crop_size = int(cfg["CROP_SIZE"] or size)
        tfs.append(_make(
            A.RandomSizedBBoxSafeCrop,
            height=crop_size, width=crop_size,
            erosion_rate=float(cfg["CROP_EROSION_RATE"]),
            p=float(cfg["CROP_P"]),
        ))

    # ── 2. 리사이즈 + 패딩 ─────────────────────────────────────────────
    if cfg["RESIZE_ENABLE"]:
        tfs.append(A.LongestMaxSize(max_size=size))
        tfs.append(_make(
            A.PadIfNeeded,
            min_height=size, min_width=size,
            border_mode=0,                  # cv2.BORDER_CONSTANT
            fill=int(cfg["PAD_VALUE"]),
        ))

    # ── 3. 반전 ────────────────────────────────────────────────────────
    if cfg["HFLIP_ENABLE"]:
        tfs.append(A.HorizontalFlip(p=float(cfg["HFLIP_P"])))
    if cfg["VFLIP_ENABLE"]:
        tfs.append(A.VerticalFlip(p=float(cfg["VFLIP_P"])))

    # ── 4. 회전 ────────────────────────────────────────────────────────
    if cfg["ROTATE_ENABLE"]:
        tfs.append(_make(
            A.Rotate,
            limit=int(cfg["ROTATE_LIMIT"]),
            border_mode=0,
            fill=int(cfg["PAD_VALUE"]),
            p=float(cfg["ROTATE_P"]),
        ))

    # ── 5. GAN 스타일 (픽셀만 바꾸므로 bbox 안전) ──────────────────────
    if cfg["GAN_ENABLE"]:
        tfs.append(GANStyleTransform(
            model_path=cfg["GAN_MODEL_PATH"],
            strength=cfg["GAN_STRENGTH"],
            p=float(cfg["GAN_P"]),
        ))

    # ── 6. 밝기 / 대비 ─────────────────────────────────────────────────
    if cfg["BRIGHTNESS_ENABLE"]:
        tfs.append(A.RandomBrightnessContrast(
            brightness_limit=float(cfg["BRIGHTNESS_LIMIT"]),
            contrast_limit=float(cfg["CONTRAST_LIMIT"]),
            p=float(cfg["BRIGHTNESS_P"]),
        ))

    # ── 7. 색상 (HSV) ──────────────────────────────────────────────────
    if cfg["HSV_ENABLE"]:
        tfs.append(A.HueSaturationValue(
            hue_shift_limit=int(cfg["HUE_SHIFT_LIMIT"]),
            sat_shift_limit=int(cfg["SAT_SHIFT_LIMIT"]),
            val_shift_limit=int(cfg["VAL_SHIFT_LIMIT"]),
            p=float(cfg["HSV_P"]),
        ))

    # ── 8. CLAHE (그림자 보정) ─────────────────────────────────────────
    if cfg["CLAHE_ENABLE"]:
        g = int(cfg["CLAHE_TILE_GRID"])
        tfs.append(_make(
            A.CLAHE,
            clip_limit=float(cfg["CLAHE_CLIP_LIMIT"]),
            tile_grid_size=(g, g),
            p=float(cfg["CLAHE_P"]),
        ))

    # ── 9. 노이즈 ──────────────────────────────────────────────────────
    if cfg["NOISE_ENABLE"]:
        lo, hi = float(cfg["NOISE_STD_MIN"]), float(cfg["NOISE_STD_MAX"])
        tfs.append(_make(
            A.GaussNoise,
            std_range=(lo, hi),                      # albumentations 2.x
            p=float(cfg["NOISE_P"]),
        ) if _ALBU_MAJOR >= 2 else _make(
            A.GaussNoise,
            var_limit=((lo * 255) ** 2, (hi * 255) ** 2),   # 1.x는 분산(0~255 스케일)
            p=float(cfg["NOISE_P"]),
        ))

    # ── 10. 블러 ───────────────────────────────────────────────────────
    if cfg["BLUR_ENABLE"]:
        bl = int(cfg["BLUR_LIMIT"])
        if bl % 2 == 0:                # MotionBlur 커널은 홀수여야 합니다
            bl += 1
            print(f"⚠️ [AUG] BLUR_LIMIT 는 홀수여야 해서 {bl} 로 올렸습니다.")
        tfs.append(A.MotionBlur(blur_limit=max(3, bl), p=float(cfg["BLUR_P"])))

    return tfs


def get_train_transforms(
    image_size: int = 640,   # 원본 976x1280 → 리사이즈 (RESIZE_ENABLE=True 일 때만)
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    min_visibility: Optional[float] = None,
    to_tensor: bool = True,
    config: Optional[Dict[str, Any]] = None,
    # ↓ 예전 시그니처 호환용 (새 코드에서는 config 로 넘기세요)
    use_gan: Optional[bool] = None,
    gan_model_path: Optional[PathLike] = None,
    gan_p: Optional[float] = None,
    gan_strength: Optional[float] = None,
):
    """학습용 전처리 + 증강 파이프라인을 만듭니다.

    무엇이 들어갈지는 **전부 `config` 딕셔너리가 결정**합니다.
    노트북에서 이렇게 쓰세요::

        tf = pt.get_train_transforms(config={"ROTATE_ENABLE": False,
                                             "BLUR_ENABLE": False})

    Args:
        image_size:     RESIZE_ENABLE=True 일 때의 출력 한 변 크기.
                        (config["IMAGE_SIZE"] 를 따로 주면 그 값이 우선)
        min_visibility: bbox 잔존 비율 하한. None이면 config 값 사용.
        to_tensor:      True면 Normalize+ToTensorV2 를 붙입니다.
                        ⚠️ **오프라인 증강(이미지를 파일로 저장)할 때는 반드시
                        False** 로 두세요. 텐서를 png로 저장할 수 없습니다.
        config:         DEFAULT_AUG_CONFIG 위에 덮어쓸 값들.
    """
    user_cfg: Dict[str, Any] = dict(config or {})
    user_cfg.setdefault("IMAGE_SIZE", image_size)
    if min_visibility is not None:
        user_cfg["MIN_VISIBILITY"] = min_visibility
    # 예전 인자 호환
    if use_gan is not None:
        user_cfg.setdefault("GAN_ENABLE", bool(use_gan))
    if gan_model_path is not None:
        user_cfg.setdefault("GAN_MODEL_PATH", str(gan_model_path))
    if gan_p is not None:
        user_cfg.setdefault("GAN_P", gan_p)
    if gan_strength is not None:
        user_cfg.setdefault("GAN_STRENGTH", gan_strength)

    cfg = merge_aug_config(user_cfg)
    transforms = build_aug_transform_list(cfg)

    if to_tensor:
        transforms += [A.Normalize(mean=mean, std=std), ToTensorV2()]

    return A.Compose(
        transforms,
        bbox_params=_bbox_params(min_visibility=float(cfg["MIN_VISIBILITY"]),
                                 min_area=float(cfg["MIN_AREA"])),
    )


def get_valid_transforms(
    image_size: int = 640,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    to_tensor: bool = True,
):
    """검증/추론용 전처리 파이프라인(증강 없이 리사이즈 + 정규화만 수행).

    Test 데이터셋(842장)도 Train과 동일하게 976x1280 해상도이므로
    별도의 크기 보정 없이 이 파이프라인을 그대로 사용할 수 있습니다.

    ★ GAN은 여기에 절대 넣지 않습니다. 검증/추론은 실제 입력 분포 그대로
      평가해야 하기 때문입니다.
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


def preview_gan(
    image_paths: Sequence[PathLike],
    out_dir: PathLike = "gan_preview",
    strength: Optional[float] = None,
    verbose: bool = True,
    **gen_kw,
) -> List[str]:
    """원본 | GAN 결과 | 혼합본을 가로로 이어 붙여 저장합니다.

    ★ 학습을 돌리기 전에 이걸로 **각인(글자)이 살아 있는지** 반드시 눈으로 보세요.
      각인이 뭉개지면 GAN_STRENGTH를 낮추거나 GAN 사용을 포기해야 합니다.
    """
    gen = GANGenerator.get(**gen_kw)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    a = GAN_STRENGTH if strength is None else strength

    for p in image_paths:
        img = imread_unicode(p)
        if img is None:
            continue
        out = gen(img)
        if out is None:
            continue
        panel = np.hstack([img, out, _blend(img, out, a)])
        dst = out_dir / f"gan_{Path(p).stem}.png"
        imwrite_unicode(dst, panel)
        saved.append(str(dst))

    if verbose:
        print(f"검수 이미지 {len(saved)}장 → {out_dir}   (원본 | GAN | 혼합 {a})")
    return saved


# ═══════════════════════════════════════════════════════════════════════════
#  ★★★ YOLO 오프라인 증강 (디스크에 증강본을 만들어 저장) ★★★
# ═══════════════════════════════════════════════════════════════════════════
#  왜 오프라인인가?
#  ----------------
#  Ultralytics(YOLO)는 이 모듈의 Albumentations 파이프라인을 인자로 받지
#  못합니다. 그래서 **학습 전에** 증강 이미지를 파일로 만들어 두고,
#  그 폴더를 가리키는 data.yaml 로 학습합니다.
#  장점: 몇 장이 만들어졌는지 정확히 세어지고, 눈으로 검수할 수 있습니다.
#  단점: 디스크를 먹습니다(원본 x (1 + N배)).

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _yolo_to_pascal(lines: List[str], w: int, h: int
                    ) -> Tuple[List[List[float]], List[int]]:
    """YOLO 라벨(class cx cy bw bh, 0~1 정규화) → pascal_voc(x1,y1,x2,y2 픽셀)."""
    bboxes: List[List[float]] = []
    labels: List[int] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w), x2), min(float(h), y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        bboxes.append([x1, y1, x2, y2])
        labels.append(c)
    return bboxes, labels


def _pascal_to_yolo(bboxes: Sequence[Sequence[float]], labels: Sequence[int],
                    w: int, h: int) -> List[str]:
    """pascal_voc(픽셀) → YOLO 라벨 문자열 목록."""
    out: List[str] = []
    for (x1, y1, x2, y2), c in zip(bboxes, labels):
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        if bw <= 0 or bh <= 0:
            continue
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        bw, bh = min(bw, 1.0), min(bh, 1.0)
        out.append(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return out


def count_dataset(root: PathLike, splits: Sequence[str] = ("train", "val", "test")
                  ) -> Dict[str, Dict[str, int]]:
    """데이터셋 폴더의 split별 **이미지 수 / 라벨 파일 수 / bbox 개수**를 셉니다.

    "전체 train 데이터가 몇 개인지" 를 확인할 때 쓰세요::

        stat = pt.count_dataset(DATASET_DIR)
        print(stat["train"]["images"])
    """
    root = Path(root)
    stat: Dict[str, Dict[str, int]] = {}
    for sp in splits:
        img_dir, lbl_dir = root / "images" / sp, root / "labels" / sp
        if not img_dir.is_dir():
            continue
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        n_box, n_lbl = 0, 0
        for p in imgs:
            f = lbl_dir / f"{p.stem}.txt"
            if f.exists():
                n_lbl += 1
                n_box += sum(1 for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())
        stat[sp] = {"images": len(imgs), "labels": n_lbl, "boxes": n_box}
    return stat


def print_dataset_stats(root: PathLike, title: str = "데이터셋 현황") -> Dict[str, Dict[str, int]]:
    """count_dataset 결과를 표로 출력하고 그대로 돌려줍니다."""
    stat = count_dataset(root)
    print(f"■ {title}  ({root})")
    print(f"  {_pad('split', 10)}{_pad('이미지', 12)}{_pad('라벨파일', 12)}bbox")
    print("  " + "-" * 40)
    for sp, v in stat.items():
        print(f"  {_pad(sp, 10)}{_pad(f'{v["images"]:,}', 12)}"
              f"{_pad(f'{v["labels"]:,}', 12)}{v['boxes']:,}")
    total = sum(v["images"] for v in stat.values())
    print("  " + "-" * 40)
    print(f"  {_pad('합계', 10)}{total:,}")
    return stat


def _draw_boxes(img_bgr: np.ndarray, bboxes: Sequence[Sequence[float]],
                labels: Sequence[int]) -> np.ndarray:
    """검수용으로 bbox를 그려 넣은 복사본을 돌려줍니다."""
    out = img_bgr.copy()
    for (x1, y1, x2, y2), c in zip(bboxes, labels):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, (0, 255, 0), 2)
        cv2.putText(out, str(int(c)), (p1[0], max(0, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def _panel(images: Sequence[np.ndarray], titles: Sequence[str],
           tile_h: int = 420) -> np.ndarray:
    """여러 장을 같은 높이로 맞춰 가로로 이어 붙이고 제목을 얹습니다."""
    tiles = []
    for im, t in zip(images, titles):
        h, w = im.shape[:2]
        nw = max(1, int(w * tile_h / h))
        r = cv2.resize(im, (nw, tile_h))
        bar = np.full((28, nw, 3), 30, np.uint8)
        cv2.putText(bar, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(np.vstack([bar, r]))
    return np.hstack(tiles)


def augment_dataset(
    src_root: PathLike,
    dst_root: Optional[PathLike] = None,
    *,
    n_aug: int = 2,
    split: str = "train",
    config: Optional[Dict[str, Any]] = None,
    keep_original: bool = True,
    prefix: str = "aug",
    out_ext: str = ".png",
    report_dir: Optional[PathLike] = None,
    n_preview: int = 8,
    save_previews: bool = True,
    copy_other_splits: bool = True,
    seed: int = 42,
    overwrite: bool = True,
    max_images: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """YOLO 데이터셋에 Albumentations 증강본을 만들어 저장합니다.

    한 장당 `n_aug` 개의 증강본을 만들며, `keep_original=True` 면 원본도 함께
    복사하므로 **최종 학습 데이터 = 원본 x (1 + n_aug)** 가 됩니다.

    Args:
        src_root:  images/{split}, labels/{split}, data.yaml 이 있는 폴더
        dst_root:  출력 폴더. None이면 `<src_root>_aug`
        n_aug:     원본 1장당 만들 증강본 수. **증강 배수는 이 값으로 조절합니다.**
                   권장 1~4 / 한계 0~10 (0이면 증강 없이 복사만)
                   ⚠️ 디스크와 학습 시간이 (1+n_aug)배로 늘어납니다.
        config:    AUG_CONFIG. 어떤 효과를 켤지 여기서 정합니다.
        keep_original: 원본도 학습에 포함할지. **True 권장**
                   (False면 모델이 "증강된 그림"만 보게 되어 실제 입력과 어긋납니다)
        prefix:    증강본 파일 접두어. `aug1_원본이름.png` 형태로 저장됩니다.
        report_dir: 검수 이미지·리포트 저장 폴더. None이면 `<dst_root>/_report`
        n_preview: 원본|증강 비교 이미지를 몇 장 저장할지. 0이면 저장 안 함
        max_images: 앞의 N장만 처리(파이프라인 테스트용). None이면 전체

    Returns:
        통계 딕셔너리. 예::

            {"n_original": 1234, "n_augmented": 2468, "n_total": 3702,
             "multiplier": 3.0, "data_yaml": ".../data.yaml", ...}
    """
    import json as _json
    import random as _random
    import time as _time

    rng = _random.Random(seed)
    np.random.seed(seed)

    src_root = Path(src_root).resolve()
    dst_root = Path(dst_root).resolve() if dst_root else src_root.parent / f"{src_root.name}_aug"
    report_dir = Path(report_dir).resolve() if report_dir else dst_root / "_report"

    img_dir, lbl_dir = src_root / "images" / split, src_root / "labels" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"{img_dir} 가 없습니다. YOLO 데이터셋 폴더를 확인하세요.")

    cfg = merge_aug_config(config)
    # 오프라인 저장이므로 텐서 변환은 절대 하지 않습니다(png로 저장해야 하므로).
    transform = get_train_transforms(config=cfg, to_tensor=False)

    if overwrite and dst_root.exists():
        shutil.rmtree(dst_root)
    (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (report_dir / "samples").mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"{img_dir} 에 이미지가 없습니다.")
    if max_images:
        files = files[:max_images]

    if verbose:
        print("=" * 70)
        print("  Albumentations 오프라인 증강")
        print(f"  입력      {src_root}")
        print(f"  출력      {dst_root}")
        print(f"  리포트    {report_dir}")
        print(f"  원본      {len(files):,}장 (split={split})")
        print(f"  증강 배수 1장당 {n_aug}개" + (" + 원본 유지" if keep_original else " (원본 미포함)"))
        print("=" * 70)
        print(describe_augmentation(cfg))
        print()

    t0 = _time.time()
    n_orig = n_aug_made = n_fail = n_dropped = 0
    box_orig = box_aug = 0
    previews: List[str] = []

    for i, f in enumerate(files):
        img = imread_unicode(f)
        if img is None:
            n_fail += 1
            print(f"⚠️ 읽기 실패: {f.name}")
            continue
        h, w = img.shape[:2]

        lbl_file = lbl_dir / f"{f.stem}.txt"
        lines = (lbl_file.read_text(encoding="utf-8").splitlines()
                 if lbl_file.exists() else [])
        bboxes, labels = _yolo_to_pascal(lines, w, h)

        # ── 원본 복사 ────────────────────────────────────────────────
        if keep_original:
            shutil.copy2(f, dst_root / "images" / split / f.name)
            (dst_root / "labels" / split / f"{f.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n_orig += 1
            box_orig += len(bboxes)

        # ── 증강본 생성 ──────────────────────────────────────────────
        made_imgs, made_titles = [], []
        for k in range(int(n_aug)):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                res = transform(image=rgb, bboxes=bboxes, labels=labels)
            except Exception as e:                      # pragma: no cover
                n_fail += 1
                print(f"⚠️ 증강 실패({f.name} #{k}): {e}")
                continue

            a_img = cv2.cvtColor(res["image"], cv2.COLOR_RGB2BGR)
            a_bb, a_lb = list(res["bboxes"]), list(res.get("labels", []))

            if bboxes and not a_bb:
                # 기하 변환으로 모든 bbox가 사라진 경우 → 라벨 없는 쓰레기 데이터
                n_dropped += 1
                continue

            ah, aw = a_img.shape[:2]
            new_stem = f"{prefix}{k + 1}_{f.stem}"
            imwrite_unicode(dst_root / "images" / split / f"{new_stem}{out_ext}", a_img)
            out_lines = _pascal_to_yolo(a_bb, a_lb, aw, ah)
            (dst_root / "labels" / split / f"{new_stem}.txt").write_text(
                "\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
            n_aug_made += 1
            box_aug += len(out_lines)

            if save_previews and len(previews) < n_preview:
                made_imgs.append(_draw_boxes(a_img, a_bb, a_lb))
                made_titles.append(f"aug{k + 1}")

        # ── 검수용 비교 이미지 ───────────────────────────────────────
        if save_previews and made_imgs and len(previews) < n_preview:
            panel = _panel([_draw_boxes(img, bboxes, labels)] + made_imgs,
                           ["original"] + made_titles)
            dst = report_dir / "samples" / f"sample_{len(previews) + 1:02d}_{f.stem}.png"
            imwrite_unicode(dst, panel)
            previews.append(str(dst))

        if verbose and (i + 1) % 200 == 0:
            print(f"    {i + 1:,}/{len(files):,} 처리  "
                  f"(원본 {n_orig:,} + 증강 {n_aug_made:,})")

    # ── 다른 split은 증강 없이 그대로 복사 ─────────────────────────────
    #    ⚠️ val/test 를 증강하면 성능 지표가 거짓말이 됩니다. 절대 하지 마세요.
    if copy_other_splits:
        for other in ("val", "test"):
            s_img, s_lbl = src_root / "images" / other, src_root / "labels" / other
            if other == split or not s_img.is_dir():
                continue
            shutil.copytree(s_img, dst_root / "images" / other, dirs_exist_ok=True)
            if s_lbl.is_dir():
                shutil.copytree(s_lbl, dst_root / "labels" / other, dirs_exist_ok=True)

    # ── data.yaml 작성 (원본 것을 읽어 path만 교체) ───────────────────
    yaml_path = dst_root / "data.yaml"
    src_yaml = src_root / "data.yaml"
    try:
        import yaml as _yaml
        base = _yaml.safe_load(src_yaml.read_text(encoding="utf-8")) if src_yaml.exists() else {}
        base = base or {}
        base["path"] = str(dst_root)
        base["train"] = f"images/{split}"
        base["val"] = "images/val" if (dst_root / "images" / "val").is_dir() else f"images/{split}"
        if (dst_root / "images" / "test").is_dir():
            base["test"] = "images/test"
        with open(yaml_path, "w", encoding="utf-8") as fp:
            _yaml.safe_dump(base, fp, allow_unicode=True, sort_keys=False)
    except Exception as e:                              # pragma: no cover
        raise RuntimeError(f"data.yaml 작성 실패({e}). 원본 data.yaml 을 확인하세요.") from e

    elapsed = _time.time() - t0
    n_total = n_orig + n_aug_made
    stats: Dict[str, Any] = {
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "report_dir": str(report_dir),
        "data_yaml": str(yaml_path),
        "split": split,
        "n_source_images": len(files),
        "n_original_kept": n_orig,
        "n_augmented": n_aug_made,
        "n_total": n_total,
        "n_aug_per_image": int(n_aug),
        "multiplier": round(n_total / max(1, len(files)), 3),
        "boxes_original": box_orig,
        "boxes_augmented": box_aug,
        "boxes_total": box_orig + box_aug,
        "n_failed": n_fail,
        "n_dropped_no_bbox": n_dropped,
        "elapsed_sec": round(elapsed, 1),
        "preview_images": previews,
        "enabled_effects": [k for k in cfg if k.endswith("_ENABLE") and cfg[k]],
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "other_splits": count_dataset(dst_root),
    }

    # ── 리포트 저장 ───────────────────────────────────────────────────
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "aug_config.json").write_text(
        _json.dumps(stats["config"], indent=2, ensure_ascii=False), encoding="utf-8")
    (report_dir / "augmentation_report.json").write_text(
        _json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (report_dir / "augmentation_report.md").write_text(
        _render_report_md(stats, cfg), encoding="utf-8")

    if verbose:
        print()
        print("■ 증강 완료")
        print(f"  원본 이미지        {len(files):,} 장")
        print(f"  ├ 그대로 유지      {n_orig:,} 장")
        print(f"  └ 새로 만든 증강본 {n_aug_made:,} 장  (1장당 {n_aug}개 요청)")
        print(f"  ─────────────────────────────────")
        print(f"  ★ 총 학습 데이터   {n_total:,} 장   (원본의 {stats['multiplier']}배)")
        print(f"  ★ 총 bbox          {stats['boxes_total']:,} 개")
        if n_dropped:
            print(f"  ⚠️ bbox가 전부 사라져 버린 증강본 {n_dropped}장 "
                  f"(회전/크롭이 과할 때 발생 → ROTATE_LIMIT 를 낮춰 보세요)")
        if n_fail:
            print(f"  ⚠️ 실패 {n_fail}장")
        print(f"  소요 {elapsed / 60:.1f}분")
        print(f"  검수 이미지 {len(previews)}장 → {report_dir / 'samples'}")
        print(f"  리포트          → {report_dir / 'augmentation_report.md'}")
        print(f"★ 학습에 쓸 data.yaml = {yaml_path}")
    return stats


def _render_report_md(stats: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """증강 결과를 마크다운 리포트로 만듭니다(팀 공유·제출용)."""
    from datetime import datetime

    on = [label for label, key, _ in [
        ("크롭", "CROP_ENABLE", None), ("리사이즈+패딩", "RESIZE_ENABLE", None),
        ("좌우반전", "HFLIP_ENABLE", None), ("상하반전", "VFLIP_ENABLE", None),
        ("회전", "ROTATE_ENABLE", None), ("밝기/대비", "BRIGHTNESS_ENABLE", None),
        ("색상(HSV)", "HSV_ENABLE", None), ("그림자(CLAHE)", "CLAHE_ENABLE", None),
        ("노이즈", "NOISE_ENABLE", None), ("블러", "BLUR_ENABLE", None),
        ("GAN", "GAN_ENABLE", None),
    ] if cfg.get(key)]

    lines = [
        "# 증강(Augmentation) 리포트",
        "",
        f"- 생성 시각: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- 입력 데이터셋: `{stats['src_root']}`",
        f"- 출력 데이터셋: `{stats['dst_root']}`",
        f"- 학습용 data.yaml: `{stats['data_yaml']}`",
        "",
        "## 1. 데이터 수량",
        "",
        "| 항목 | 개수 |",
        "|---|---|",
        f"| 원본 train 이미지 | {stats['n_source_images']:,} 장 |",
        f"| 유지한 원본 | {stats['n_original_kept']:,} 장 |",
        f"| 새로 만든 증강본 | {stats['n_augmented']:,} 장 |",
        f"| **총 학습 이미지** | **{stats['n_total']:,} 장** |",
        f"| **증강 배수** | **{stats['multiplier']} 배** (원본 1장 → {stats['n_aug_per_image']}개 증강) |",
        f"| 총 bbox | {stats['boxes_total']:,} 개 |",
        f"| bbox 소실로 버린 증강본 | {stats['n_dropped_no_bbox']:,} 장 |",
        f"| 처리 실패 | {stats['n_failed']:,} 장 |",
        f"| 소요 시간 | {stats['elapsed_sec']}초 |",
        "",
        "### split별 최종 현황",
        "",
        "| split | 이미지 | 라벨파일 | bbox |",
        "|---|---|---|---|",
    ]
    for sp, v in stats["other_splits"].items():
        lines.append(f"| {sp} | {v['images']:,} | {v['labels']:,} | {v['boxes']:,} |")

    lines += [
        "",
        "> val / test 는 **증강하지 않고 그대로 복사**했습니다. "
        "검증 데이터를 증강하면 성능 지표를 신뢰할 수 없기 때문입니다.",
        "",
        "## 2. 적용한 증강 효과",
        "",
        f"켜진 효과: **{', '.join(on) if on else '없음'}**",
        "",
        "| 파라미터 | 값 |",
        "|---|---|",
    ]
    for k in sorted(cfg):
        lines.append(f"| `{k}` | `{cfg[k]}` |")

    lines += [
        "",
        "## 3. 검수 이미지",
        "",
        f"`samples/` 폴더에 원본과 증강본을 나란히 놓고 bbox를 그린 이미지 "
        f"{len(stats['preview_images'])}장을 저장했습니다.",
        "",
        "**학습을 돌리기 전에 반드시 눈으로 확인하세요.**",
        "",
        "- 초록 박스가 알약을 정확히 감싸고 있나요? (아니면 ROTATE_LIMIT ↓)",
        "- 알약의 각인(글자)이 읽히나요? (아니면 BLUR/NOISE ↓)",
        "- 알약 색이 원본과 비슷한가요? (아니면 HUE_SHIFT_LIMIT ↓)",
        "",
    ]
    return "\n".join(lines)


def preview_augmentation(
    src_root: PathLike,
    out_dir: PathLike = "aug_preview",
    *,
    split: str = "train",
    n_images: int = 4,
    n_aug: int = 3,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    verbose: bool = True,
) -> List[str]:
    """**데이터셋 전체를 만들기 전에** 몇 장만 뽑아 증강 결과를 미리 봅니다.

    전체 증강은 수십 분이 걸릴 수 있으므로, 파라미터를 바꿀 때마다
    이 함수로 먼저 3~4장만 확인하고 마음에 들면 `augment_dataset` 을 돌리세요.
    """
    import random as _random

    rng = _random.Random(seed)
    np.random.seed(seed)
    src_root, out_dir = Path(src_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir, lbl_dir = src_root / "images" / split, src_root / "labels" / split
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"{img_dir} 에 이미지가 없습니다.")
    picked = rng.sample(files, min(n_images, len(files)))

    cfg = merge_aug_config(config)
    transform = get_train_transforms(config=cfg, to_tensor=False)
    if verbose:
        print(describe_augmentation(cfg))

    saved: List[str] = []
    for f in picked:
        img = imread_unicode(f)
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl_file = lbl_dir / f"{f.stem}.txt"
        lines = lbl_file.read_text(encoding="utf-8").splitlines() if lbl_file.exists() else []
        bboxes, labels = _yolo_to_pascal(lines, w, h)

        imgs, titles = [_draw_boxes(img, bboxes, labels)], ["original"]
        for k in range(n_aug):
            res = transform(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                            bboxes=bboxes, labels=labels)
            a = cv2.cvtColor(res["image"], cv2.COLOR_RGB2BGR)
            imgs.append(_draw_boxes(a, list(res["bboxes"]), list(res.get("labels", []))))
            titles.append(f"aug{k + 1}")

        dst = out_dir / f"preview_{f.stem}.png"
        imwrite_unicode(dst, _panel(imgs, titles))
        saved.append(str(dst))

    if verbose:
        print(f"\n검수 이미지 {len(saved)}장 → {out_dir}")
        print("★ 각인이 읽히는지 / bbox가 알약을 감싸는지 눈으로 확인하세요.")
    return saved


#import 할 시 실행되는 코드가 아닙니다.
if __name__ == "__main__":
    dummy_image = (np.random.rand(1280, 976, 3) * 255).astype(np.uint8)
    dummy_bboxes = [[100, 150, 300, 400], [500, 600, 700, 900]]
    dummy_labels = [1, 5]

    print(describe_augmentation())
    print()

    # RESIZE_ENABLE 을 켜서 640x640 텐서가 나오는지 확인합니다.
    train_tf = get_train_transforms(image_size=640, config={"RESIZE_ENABLE": True})
    valid_tf = get_valid_transforms(image_size=640)

    out_train = train_tf(image=dummy_image, bboxes=dummy_bboxes, labels=dummy_labels)
    out_valid = valid_tf(image=dummy_image, bboxes=dummy_bboxes, labels=dummy_labels)

    print("train image tensor shape:", tuple(out_train["image"].shape))
    print("train bboxes after transform:", out_train["bboxes"])
    print("valid image tensor shape:", tuple(out_valid["image"].shape))
    print("valid bboxes after transform:", out_valid["bboxes"])

    print()
    print(describe_gan())
    if GAN_MODEL_PATH:
        check_gan_model()
    else:
        print("\n(GAN_MODEL_PATH 가 없어 GAN 점검은 건너뜁니다)")
