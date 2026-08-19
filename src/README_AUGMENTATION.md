# 증강(Augmentation) 사용 설명서

경구약제 Object Detection 프로젝트 · `pill_transforms.py` + `yolo11s_augmix.ipynb`

> **처음 보는 사람은 [3. 5분 만에 돌려보기](#3-5분-만에-돌려보기)만 읽으면 됩니다.**
> 파라미터를 만지고 싶어지면 [5. 파라미터 사전](#5-파라미터-사전-권장값--한계값)을 보세요.

---

## 1. 이번 개편에서 달라진 것

| | 예전 | 지금 |
|---|---|---|
| **켜고 끄기** | `A.Rotate(p=0.4)` → 40% 확률로 랜덤하게 걸림 | `ROTATE_ENABLE=True/False` → **확실하게** 켜고 끔 |
| **밝기 / 컬러 / 그림자** | `A.OneOf([...], p=0.6)` → **셋 중 하나만**, 그것도 60% 확률 | 셋을 **각각 독립**으로 켜고 끔. 켠 건 전부 적용 |
| **노이즈 / 블러** | `A.OneOf([...], p=0.2)` → 둘 중 하나만, 20% 확률 | 각각 독립으로 켜고 끔 |
| **설정 위치** | `pill_transforms.py` 안의 코드를 직접 고쳐야 함 | 노트북의 `AUG_CONFIG` 한 곳 |
| **증강 배수** | 개념 자체가 없었음 | `AUG_PER_IMAGE` 로 조절 |
| **결과 확인** | 몇 장에 무엇이 걸렸는지 알 수 없었음 | 장수·bbox 수·리포트·검수 이미지 자동 생성 |

### "확률(p)을 없앴다"는 게 무슨 뜻인가요?

예전 코드는 `p=0.4`, `p=0.6`, `p=0.2` 처럼 **"이 증강이 걸릴지 말지"를 주사위로**
정했습니다. 그래서 학습이 끝난 뒤에도 *어떤 이미지에 무엇이 적용됐는지* 알 수
없었고, 실험을 비교할 수도 없었습니다.

지금은 `*_ENABLE = True` 로 켠 효과는 **만드는 증강본마다 반드시** 들어갑니다.

다만 **효과의 세기는 여전히 매번 랜덤**입니다. 예를 들어 `ROTATE_LIMIT=15` 면
매번 -15˚ ~ +15˚ 중 하나가 뽑힙니다. 이건 없앨 수 없습니다 — 세기까지 고정하면
`AUG_PER_IMAGE=3` 으로 3장을 만들어도 **완전히 똑같은 이미지 3장**이 나와서
증강의 의미가 사라지기 때문입니다.

> 정리: **무엇을 적용할지 = 내가 결정 / 얼마나 세게 = 범위 안에서 랜덤**

굳이 예전처럼 확률로 굴리고 싶다면 `ROTATE_P` 같은 `*_P` 값을 1.0 미만으로
주면 됩니다. (기본은 전부 1.0. 초보자는 건드리지 마세요.)

---

## 2. 증강이 두 종류인 이유 (중요)

이 프로젝트에는 **성격이 다른 두 개의 증강**이 있습니다. 헷갈리면 과증강으로
성능이 떨어지니 꼭 구분하세요.

| | **[B] YOLO 내장 증강** | **[D] Albumentations 오프라인 증강** |
|---|---|---|
| 누가 | Ultralytics 내부 | `pill_transforms.py` |
| 어디서 조절 | 노트북 `YOLO_AUG` | 노트북 `AUG_CONFIG` |
| 언제 | 학습 중 **매 배치마다** | 학습 **전에 한 번** |
| 파일이 생기나 | ❌ 메모리에서만 | ✅ 디스크에 이미지로 저장 |
| 데이터 수가 늘어나나 | ❌ (매번 다르게 보일 뿐) | ✅ 원본 x (1 + N) |
| 눈으로 검수 가능 | 어려움 | ✅ 쉬움 |
| 항목 | hsv / degrees / mosaic / mixup 등 14개 | 회전 · 밝기 · 색상 · CLAHE · 노이즈 · 블러 · GAN |

**왜 [D]는 미리 파일로 만드나요?**
Ultralytics(YOLO)는 외부 Albumentations 파이프라인을 인자로 받지 못합니다.
그래서 학습 전에 증강 이미지를 디스크에 만들어 두고, 그 폴더를 가리키는
`data.yaml` 로 학습합니다. 대신 **몇 장이 만들어졌는지 정확히 세어지고 눈으로
검수할 수 있다**는 큰 장점이 생깁니다.

> ⚠️ **역할이 겹치면 끄세요.**
> `[D] ROTATE_ENABLE=True` 라면 `[B] degrees=0.0`,
> `[D] HSV_ENABLE=True` 라면 `[B] hsv_s`, `hsv_v` 는 낮게.
> 기본 설정이 이미 그렇게 맞춰져 있습니다.

---

## 3. 5분 만에 돌려보기

1. `pill_transforms.py` 를 `PROJECT_ROOT/src/` 에 둡니다.
2. `yolo11s_augmix.ipynb` 를 위에서부터 실행합니다.
3. **4-1 셀**에서 두 줄만 보면 됩니다.

```python
AUG_PER_IMAGE = 2          # 원본 1장당 증강본 2개 → 총 3배
AUG_CONFIG = dict(
    ROTATE_ENABLE     = True,    # 회전 켬
    BRIGHTNESS_ENABLE = True,    # 밝기/대비 켬
    HSV_ENABLE        = True,    # 색상 켬
    CLAHE_ENABLE      = True,    # 그림자 보정 켬
    NOISE_ENABLE      = True,    # 노이즈 켬
    BLUR_ENABLE       = True,    # 블러 켬
    ...
)
```

4. **4-1-b 셀**로 미리보기 → 각인이 읽히는지 눈으로 확인
5. **4-1-c 셀**로 증강 데이터셋 생성 → 장수가 출력됩니다
6. **4-2 셀**로 학습

### 증강을 아예 끄고 싶다면

```python
ALBU_ENABLE = False      # 이 한 줄이면 원본 데이터셋으로 그대로 학습합니다
```

### 특정 효과만 끄고 싶다면

```python
AUG_CONFIG["BLUR_ENABLE"]  = False   # 블러만 끄기
AUG_CONFIG["NOISE_ENABLE"] = False   # 노이즈만 끄기
```

### 밝기만 켜고 나머지는 다 끄고 싶다면

```python
AUG_CONFIG = dict(
    ROTATE_ENABLE=False, BRIGHTNESS_ENABLE=True, HSV_ENABLE=False,
    CLAHE_ENABLE=False, NOISE_ENABLE=False, BLUR_ENABLE=False,
    CROP_ENABLE=False, RESIZE_ENABLE=False,
    HFLIP_ENABLE=False, VFLIP_ENABLE=False, GAN_ENABLE=False,
    BRIGHTNESS_LIMIT=0.25, CONTRAST_LIMIT=0.25,
)
```

> 적지 않은 키는 `DEFAULT_AUG_CONFIG` 의 기본값이 자동으로 채워집니다.
> 없는 키를 적으면 `⚠️ 모르는 설정 키가 있습니다(오타?)` 경고가 뜹니다.

---

## 4. 증강 배수와 데이터 개수

```
최종 학습 이미지 수 = 원본 x (1 + AUG_PER_IMAGE)      # AUG_KEEP_ORIGINAL=True
                    = 원본 x AUG_PER_IMAGE            # AUG_KEEP_ORIGINAL=False
```

| `AUG_PER_IMAGE` | 원본 1,000장 기준 | 디스크 | 1에포크 시간 |
|---|---|---|---|
| 0 | 1,000장 (증강 없음) | x1 | x1 |
| 1 | 2,000장 | x2 | x2 |
| **2 (권장)** | **3,000장** | x3 | x3 |
| 3 | 4,000장 | x4 | x4 |
| 5 이상 | 6,000장~ | x6~ | x6~ |

**권장 1~3.** 5를 넘기면 비슷비슷한 이미지만 늘어나서 성능은 그대로인데
학습 시간만 몇 배가 됩니다. 데이터가 부족하다고 느끼면 배수를 올리기 전에
**소수 클래스만 더 증강**하는 쪽을 먼저 고민하세요(클래스 불균형 51배).

`AUG_KEEP_ORIGINAL=True` 를 권장합니다. False면 모델이 "증강된 그림"만 보게
되는데, 실제 테스트 이미지는 증강되지 않은 원본이라 분포가 어긋납니다.

### 지금 데이터가 몇 장인지 확인만 하고 싶다면

```python
import pill_transforms as pt
pt.print_dataset_stats(DATASET_DIR)      # split별 이미지·라벨·bbox 개수
```

---

## 5. 파라미터 사전 (권장값 · 한계값)

> **한계값** = 넘기면 라벨이 오염되거나 성능이 떨어지기 시작하는 선입니다.
> 이 데이터셋의 특성(각인으로 약을 구분 / 색이 클래스 정보 / 배경·조명이 항상 동일)을
> 반영한 값입니다.

### 5-1. 기하 변환

| 파라미터 | 하는 일 | 기본 | 권장 | 한계 | 넘기면 |
|---|---|---|---|---|---|
| `CROP_ENABLE` | bbox 보존 랜덤 크롭 | `False` | **OFF** | — | 정사각으로 잘라 **종횡비 왜곡**(동그란 약→타원). 알약 크기 단서도 깨짐 |
| `CROP_EROSION_RATE` | 얼마나 과감히 자를지 | `0.1` | 0.0~0.2 | 0.5 | 작은 알약이 잘려 나감 |
| `RESIZE_ENABLE` | 리사이즈 + 검은 여백 | `False` | **OFF**(YOLO용) | — | YOLO가 학습 때 또 letterbox → **두 번 리사이즈로 각인 뭉개짐** |
| `IMAGE_SIZE` | 리사이즈 크기 | `640` | 640~960 | 320~1280 | 640 미만이면 각인이 안 보임 |
| `PAD_VALUE` | 여백 색 | `0` | 0 또는 200~220 | 0~255 | 이 데이터셋 배경이 연회색이라 200~220이 더 자연스러움 |
| `HFLIP_ENABLE` | 좌우 반전 | `False` | **OFF 고정** | — | **각인 글자가 거울상** → 존재하지 않는 약이 됨 |
| `VFLIP_ENABLE` | 상하 반전 | `False` | **OFF 고정** | — | 같은 이유 |
| `ROTATE_ENABLE` | 회전 | `True` | **ON** | — | 촬영각 70/75/90˚ 변화 대응에 유용 |
| `ROTATE_LIMIT` | ±N도 | `15` | 10~20 | 45 | 축정렬 bbox가 알약보다 훨씬 커져 라벨 부정확 |

### 5-2. 색·명암 (예전에는 `OneOf`로 셋 중 하나만 걸렸던 부분)

| 파라미터 | 하는 일 | 기본 | 권장 | 한계 | 넘기면 |
|---|---|---|---|---|---|
| `BRIGHTNESS_ENABLE` | 밝기·대비 | `True` | **ON** | — | EDA상 조명이 항상 동일 → 일반화에 **가장 중요** |
| `BRIGHTNESS_LIMIT` | 밝기 변동폭 | `0.2` | 0.15~0.30 | 0.5 | 흰 알약이 배경에 묻히고 검은 알약이 뭉개짐 |
| `CONTRAST_LIMIT` | 대비 변동폭 | `0.2` | 0.15~0.30 | 0.5 | 같음 |
| `HSV_ENABLE` | 색상·채도·명도 | `True` | **ON** | — | |
| `HUE_SHIFT_LIMIT` | 색상(색조) 이동 | `10` | **5~10** | **15** | ⚠️⚠️ **알약 색(color1)이 곧 클래스**. 노란 약이 초록 약이 되면 라벨이 거짓말 |
| `SAT_SHIFT_LIMIT` | 채도 이동 | `20` | 10~25 | 40 | 색이 과하게 바래거나 형광색이 됨 |
| `VAL_SHIFT_LIMIT` | 명도 이동 | `15` | 10~20 | 40 | 밝기 파라미터와 중복되니 둘 다 크게 주지 말 것 |
| `CLAHE_ENABLE` | 그림자 보정(국소 대비 평탄화) | `True` | **ON** | — | 그림자에 묻힌 각인을 살려 줌 |
| `CLAHE_CLIP_LIMIT` | 평탄화 강도 | `2.0` | 2.0~3.0 | 4.0 | 노이즈까지 증폭되어 알약 표면이 지저분해짐 |

### 5-3. 화질 열화 (예전에는 `OneOf`로 둘 중 하나만 걸렸던 부분)

| 파라미터 | 하는 일 | 기본 | 권장 | 한계 | 넘기면 |
|---|---|---|---|---|---|
| `NOISE_ENABLE` | 가우시안 노이즈 | `True` | ON | — | 저조도 촬영의 센서 노이즈 대응 |
| `NOISE_STD_MIN` / `MAX` | 노이즈 세기(0~1 표준편차) | `0.03` / `0.08` | 0.02~0.10 | **0.20** | **각인이 노이즈에 묻혀 사람 눈으로도 안 보임** |
| `BLUR_ENABLE` | 모션 블러(흔들림) | `True` | ON | — | 손떨림 대응 |
| `BLUR_LIMIT` | 커널 크기(**홀수만**) | `3` | **3** | 9 | 7 이상이면 각인이 사실상 사라짐. 짝수를 넣으면 자동으로 +1 |

### 5-4. bbox 필터

| 파라미터 | 하는 일 | 기본 | 권장 | 한계 |
|---|---|---|---|---|
| `MIN_VISIBILITY` | 잘린 뒤 몇 % 남아야 라벨 유지 | `0.2` | 0.2~0.4 | 0.0~1.0 |
| `MIN_AREA` | 최소 bbox 픽셀 면적 | `4.0` | 4~64 | — |

↑ 올리면 잘린 알약 라벨이 잘 버려집니다(라벨 품질 ↑, 학습 데이터 수 ↓).

### 5-5. GAN 스타일 증강 (선택)

학습된 image-to-image 생성기(CycleGAN / pix2pix 등)를 ONNX 또는 TorchScript로
내보낸 파일이 **있을 때만** 동작합니다. 이 저장소에는 가중치가 없습니다.

| 파라미터 | 하는 일 | 권장 | 주의 |
|---|---|---|---|
| `GAN_ENABLE` | GAN 사용 여부 | 모델 있을 때만 ON | 없으면 경고만 찍고 원본 통과 |
| `GAN_MODEL_PATH` | `.onnx` 또는 TorchScript `.pt` 경로 | — | |
| `GAN_STRENGTH` | 원본과 섞는 비율 | 0.3~0.6 | 1.0이면 GAN 결과 그대로 → **각인이 지워질 위험** |
| `pt.GAN_INPUT_SIZE` | 생성기 입력 크기 | 256~512 | 크면 느리고 VRAM 먹음 |
| `pt.GAN_INPUT_RANGE` | `"tanh"`(-1~1) / `"sigmoid"`(0~1) | 모델에 맞춰 | 틀리면 색이 이상해짐 |
| `pt.GAN_CHANNEL_ORDER` | `"rgb"` / `"bgr"` | 모델에 맞춰 | 틀리면 빨강↔파랑 뒤집힘 |

켜기 전에 반드시:

```python
pt.check_gan_model()      # 로드·추론이 되는지 1장으로 점검
pt.preview_gan([...])     # 원본 | GAN | 혼합 을 눈으로 확인
```

**각인이 뭉개지면 `GAN_STRENGTH`를 낮추거나 GAN을 포기하세요.** 없던 각인이
생기거나 있던 각인이 지워지면 그건 라벨 오염입니다.

---

## 6. 만들어지는 폴더 구조

```
PROJECT_ROOT/
├── src/
│   └── pill_transforms.py                  # 증강 모듈
├── data/
│   ├── processed/                          # 원본 YOLO 데이터셋 (입력, 안 건드림)
│   │   ├── images/{train,val,test}/
│   │   ├── labels/{train,val,test}/
│   │   └── data.yaml
│   └── processed_albu_x3/                  # ★ 증강 데이터셋 (학습에 사용)
│       ├── images/train/                   #   원본 + aug1_*.png + aug2_*.png
│       ├── images/{val,test}/              #   증강 없이 그대로 복사
│       ├── labels/{train,val,test}/
│       └── data.yaml                       #   ← TRAIN_YAML 이 이걸 가리킴
└── outputs/
    ├── augmentation/albu_x3/               # ★ 사람이 보는 산출물
    │   ├── preview/                        #   파라미터 조정용 미리보기
    │   ├── samples/                        #   원본 | 증강본 비교 (bbox 표시)
    │   ├── augmentation_report.md          #   수량·설정 리포트 (팀 공유/제출용)
    │   ├── augmentation_report.json        #   같은 내용의 기계 판독용
    │   └── aug_config.json                 #   이 실험의 AUG_CONFIG 전체 (재현용)
    └── yolo/yolo11s_mix8_albu_x3/          # YOLO 학습 결과 (weights/best.pt)
```

- 폴더 이름의 `albu_x3` 는 `AUG_TAG` 변수로 자동 생성됩니다
  (`AUG_PER_IMAGE=2` → `1+2=3` → `albu_x3`).
  실험을 여러 개 돌려도 서로 덮어쓰지 않습니다.
- 증강본 파일명은 `aug1_원본이름.png`, `aug2_원본이름.png` 형태라
  나중에 골라내거나 지우기 쉽습니다.
- `augment_dataset(..., overwrite=True)` 는 출력 폴더를 **지우고 새로 만듭니다.**
  결과를 남기고 싶으면 `AUG_TAG` 를 바꾸세요.

> 팀 표준 폴더 구조 문서가 따로 있다면 노트북 4-1 셀의
> `AUG_DATASET_DIR` / `AUG_REPORT_DIR` 두 줄만 고치면 그대로 맞출 수 있습니다.

### `augmentation_report.md` 에 들어가는 것

- 원본 train 이미지 수 / 유지한 원본 수 / 새로 만든 증강본 수
- **총 학습 이미지 수**와 **증강 배수**
- 총 bbox 수, bbox 소실로 버린 증강본 수, 실패 수, 소요 시간
- split별 최종 현황 표 (train / val / test)
- 켠 효과 목록과 전체 파라미터 값 표
- 검수 체크리스트

---

## 7. 검수 체크리스트 (학습 전에 꼭)

`outputs/augmentation/<태그>/samples/` 의 이미지를 열어 세 가지를 봅니다.

| 볼 것 | 이상하면 |
|---|---|
| 초록 bbox가 알약을 정확히 감싸나 | `ROTATE_LIMIT` ↓ / `CROP_ENABLE=False` |
| **알약 각인(글자)이 읽히나** | `BLUR_LIMIT` ↓ , `NOISE_STD_MAX` ↓ , `CLAHE_CLIP_LIMIT` ↓ |
| 알약 색이 원본과 비슷한가 | `HUE_SHIFT_LIMIT` ↓ , `SAT_SHIFT_LIMIT` ↓ |
| 알약 모양(원/타원/캡슐)이 유지되나 | `CROP_ENABLE=False` , `ROTATE_LIMIT` ↓ |

**각인이 안 읽히면 사람도 못 맞히는 데이터입니다. 모델도 못 맞힙니다.**

---

## 8. 자주 겪는 문제

| 증상 | 원인 · 해결 |
|---|---|
| `⚠️ 모르는 설정 키가 있습니다(오타?)` | `AUG_CONFIG` 키 오타. 대문자·언더스코어 확인 |
| `bbox가 전부 사라져 버린 증강본 N장` | 회전/크롭이 과함 → `ROTATE_LIMIT` ↓ , `MIN_VISIBILITY` ↓ |
| 증강이 너무 느림 | `AUG_PER_IMAGE` ↓ , 또는 `AUG_MAX_IMAGES=50` 으로 먼저 테스트 |
| 디스크가 꽉 참 | `AUG_PER_IMAGE` ↓ , 예전 `processed_albu_x*` 폴더 삭제 |
| 학습이 원본 데이터로 돌아감 | `model.train(data=str(TRAIN_YAML))` 인지 확인 (`RUNTIME_YAML` 아님) |
| `to_tensor=True` 인데 torch 없다고 에러 | 오프라인 증강에는 torch가 필요 없습니다. `to_tensor=False` (기본) 사용 |
| albumentations 버전 때문에 인자 에러 | 1.x / 2.x 차이는 `_make()` 가 흡수합니다. 그래도 나면 `pip install -U albumentations` |
| 한글 경로에서 이미지가 안 열림 | `pt.imread_unicode` / `pt.imwrite_unicode` 를 쓰고 있습니다. `cv2.imread` 직접 호출 금지 |
| 성능이 오히려 떨어짐 | 과증강 의심. `[B] YOLO_AUG` 와 `[D] AUG_CONFIG` 가 겹치는지 확인 |

---

## 9. 모듈 API 요약 (직접 코드로 쓰고 싶을 때)

```python
import pill_transforms as pt

# 설정 확인
print(pt.describe_augmentation(AUG_CONFIG))       # 표로 출력
cfg = pt.merge_aug_config({"BLUR_ENABLE": False}) # 기본값 + 내 설정

# 데이터 개수 세기
stat = pt.count_dataset("data/processed")         # {"train": {"images":..,"boxes":..}, ...}
pt.print_dataset_stats("data/processed")          # 표로 출력

# 몇 장만 미리보기 (파일 생성 없음)
pt.preview_augmentation("data/processed", "out/preview",
                        n_images=3, n_aug=2, config=AUG_CONFIG)

# 오프라인 증강 데이터셋 생성
stats = pt.augment_dataset("data/processed", "data/processed_aug",
                           n_aug=2, config=AUG_CONFIG,
                           keep_original=True, report_dir="outputs/augmentation/x3")
print(stats["n_total"], stats["multiplier"], stats["data_yaml"])

# Faster R-CNN 등 Albumentations 파이프라인을 직접 쓰는 경로
tf = pt.get_train_transforms(config={**AUG_CONFIG, "RESIZE_ENABLE": True},
                             to_tensor=True)
out = tf(image=rgb_ndarray, bboxes=[[x1,y1,x2,y2]], labels=[3])
```

`augment_dataset` 이 돌려주는 딕셔너리의 주요 키:

| 키 | 뜻 |
|---|---|
| `n_source_images` | 원본 train 이미지 수 |
| `n_original_kept` | 그대로 복사한 원본 수 |
| `n_augmented` | 새로 만든 증강본 수 |
| `n_total` | **총 학습 이미지 수** |
| `multiplier` | **증강 배수** |
| `boxes_total` | 총 bbox 수 |
| `n_dropped_no_bbox` | bbox가 전부 사라져 버린 증강본 수 |
| `data_yaml` | 학습에 넣을 data.yaml 절대경로 |
| `report_dir` | 리포트·검수 이미지 폴더 |

---

## 10. 팀 규칙

1. **val / test 는 절대 증강하지 않습니다.** (모듈이 자동으로 그대로 복사합니다)
   증강된 검증 데이터로 잰 mAP는 아무 의미가 없습니다.
2. 실험을 바꿀 때는 `AUG_TAG` 도 바꿔서 이전 결과를 덮어쓰지 마세요.
3. 학습을 돌리기 전에 **검수 이미지를 반드시 눈으로 봅니다.**
4. W&B에 `AUG_CONFIG` 전체가 자동으로 기록되므로, 나중에
   "그때 뭘 켰더라?"는 W&B config 또는 `aug_config.json` 을 보면 됩니다.
5. `[B] YOLO_AUG` 와 `[D] AUG_CONFIG` 중 **한쪽만 세게** 켭니다.
