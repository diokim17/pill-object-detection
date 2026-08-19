# 증강(Augmentation) 리포트

- 생성 시각: 2026-08-19 03:01:48
- 입력 데이터셋: `/tmp/ds`
- 출력 데이터셋: `/tmp/proj/data/processed_albu_x3`
- 학습용 data.yaml: `/tmp/proj/data/processed_albu_x3/data.yaml`

## 1. 데이터 수량

| 항목 | 개수 |
|---|---|
| 원본 train 이미지 | 12 장 |
| 유지한 원본 | 12 장 |
| 새로 만든 증강본 | 24 장 |
| **총 학습 이미지** | **36 장** |
| **증강 배수** | **3.0 배** (원본 1장 → 2개 증강) |
| 총 bbox | 69 개 |
| bbox 소실로 버린 증강본 | 0 장 |
| 처리 실패 | 0 장 |
| 소요 시간 | 4.3초 |

### split별 최종 현황

| split | 이미지 | 라벨파일 | bbox |
|---|---|---|---|
| train | 36 | 36 | 69 |
| val | 4 | 4 | 6 |
| test | 3 | 3 | 3 |

> val / test 는 **증강하지 않고 그대로 복사**했습니다. 검증 데이터를 증강하면 성능 지표를 신뢰할 수 없기 때문입니다.

## 2. 적용한 증강 효과

켜진 효과: **회전, 밝기/대비, 색상(HSV), 그림자(CLAHE), 노이즈, 블러**

| 파라미터 | 값 |
|---|---|
| `BLUR_ENABLE` | `True` |
| `BLUR_LIMIT` | `3` |
| `BLUR_P` | `1.0` |
| `BRIGHTNESS_ENABLE` | `True` |
| `BRIGHTNESS_LIMIT` | `0.2` |
| `BRIGHTNESS_P` | `1.0` |
| `CLAHE_CLIP_LIMIT` | `2.0` |
| `CLAHE_ENABLE` | `True` |
| `CLAHE_P` | `1.0` |
| `CLAHE_TILE_GRID` | `8` |
| `CONTRAST_LIMIT` | `0.2` |
| `CROP_ENABLE` | `False` |
| `CROP_EROSION_RATE` | `0.1` |
| `CROP_P` | `1.0` |
| `CROP_SIZE` | `None` |
| `GAN_ENABLE` | `False` |
| `GAN_MODEL_PATH` | `/tmp/proj/models/generator.onnx` |
| `GAN_P` | `1.0` |
| `GAN_STRENGTH` | `0.5` |
| `HFLIP_ENABLE` | `False` |
| `HFLIP_P` | `1.0` |
| `HSV_ENABLE` | `True` |
| `HSV_P` | `1.0` |
| `HUE_SHIFT_LIMIT` | `10` |
| `IMAGE_SIZE` | `640` |
| `MIN_AREA` | `4.0` |
| `MIN_VISIBILITY` | `0.2` |
| `NOISE_ENABLE` | `True` |
| `NOISE_P` | `1.0` |
| `NOISE_STD_MAX` | `0.08` |
| `NOISE_STD_MIN` | `0.03` |
| `PAD_VALUE` | `0` |
| `RESIZE_ENABLE` | `False` |
| `ROTATE_ENABLE` | `True` |
| `ROTATE_LIMIT` | `15` |
| `ROTATE_P` | `1.0` |
| `SAT_SHIFT_LIMIT` | `20` |
| `VAL_SHIFT_LIMIT` | `15` |
| `VFLIP_ENABLE` | `False` |
| `VFLIP_P` | `1.0` |

## 3. 검수 이미지

`samples/` 폴더에 원본과 증강본을 나란히 놓고 bbox를 그린 이미지 8장을 저장했습니다.

**학습을 돌리기 전에 반드시 눈으로 확인하세요.**

- 초록 박스가 알약을 정확히 감싸고 있나요? (아니면 ROTATE_LIMIT ↓)
- 알약의 각인(글자)이 읽히나요? (아니면 BLUR/NOISE ↓)
- 알약 색이 원본과 비슷한가요? (아니면 HUE_SHIFT_LIMIT ↓)
