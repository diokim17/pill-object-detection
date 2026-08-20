# 💊 Pill Object Detection

사진 속 알약의 종류와 위치를 검출하는 Object Detection 프로젝트입니다.

> Codeit AI Engineer Bootcamp Team Project  
> Project Period: **2026.08.05 ~ 2026.08.21**

## 📌 Project Overview

헬스케어 스타트업 **Health Eat**의 AI 엔지니어링 팀이라는 가정하에 진행했습니다. 사용자가 모바일 애플리케이션으로 복용 중인 약을 촬영하면, 이미지 인식 기술로 알약을 식별하고 관련 정보를 제공하는 서비스를 목표로 합니다.

주요 목표는 다음과 같습니다.

- 한 이미지에서 최대 **4개의 알약(Object)** 검출
- 알약별 **Bounding Box** 예측
- 알약별 **Class** 분류
- 데이터와 모델 실험을 통한 Object Detection 성능 개선

## 👥 Team

| Name | GitHub | 협업 일지 |
| --- | --- | --- |
| 김도영 | [@diokim17](https://github.com/diokim17) | [도영님 협업일지](https://docs.google.com/document/d/1NbZmXKJG7K_dmMsLEOKm8Qg8OaqXq9PPsOr-C1_MVJc/edit?tab=t.0) |
| 김민협 | [@seolwoom](https://github.com/seolwoom) | [민협님 협업일지](https://docs.google.com/document/d/1NbZmXKJG7K_dmMsLEOKm8Qg8OaqXq9PPsOr-C1_MVJc/edit?tab=t.4gadpvsip6lf) |
| 이원영 | [@1young-codes](https://github.com/1young-codes) | [원영님 협업일지](https://docs.google.com/document/d/1NbZmXKJG7K_dmMsLEOKm8Qg8OaqXq9PPsOr-C1_MVJc/edit?tab=t.qt241c3bv39) |
| 이재웅 | [@mitguchin](https://github.com/mitguchin) | [재웅님 협업일지](https://docs.google.com/document/d/1NbZmXKJG7K_dmMsLEOKm8Qg8OaqXq9PPsOr-C1_MVJc/edit?tab=t.x7hx0cls3eoj) |
| 황세희 | [@hwangsiiii](https://github.com/hwangsiiii) | [세희님 협업일지](https://docs.google.com/document/d/1NbZmXKJG7K_dmMsLEOKm8Qg8OaqXq9PPsOr-C1_MVJc/edit?tab=t.phfoaioiz36c) |

### 회의록 및 일정

- [회의록 및 협업일지](https://docs.google.com/document/d/1NbZmXKJG7K_dmMsLEOKm8Qg8OaqXq9PPsOr-C1_MVJc/edit?tab=t.fwuvz6mbikcq)
- [일정 관리](https://docs.google.com/spreadsheets/d/1tYkRh30PQU-JUwD4WYt2OkT3x0hBXk1jCB_lgWlEViM/edit?gid=193608105#gid=193608105)

## 🗂️ Project Structure

```text
pill-object-detection/
├── notebooks/
│   ├── 01_eda.ipynb                 # 데이터 탐색 및 시각화
│   ├── pill_detection_dataset.ipynb # 원본 데이터를 YOLO 형식으로 변환
│   ├── yolo11_unified.ipynb         # 데이터 준비·학습·추론 통합 파이프라인
│   └── yolo11s_yolo11m WBF.ipynb    # YOLO11s·YOLO11m WBF 앙상블
├── data/
│   ├── dataset/
│   │   ├── raw_data/                # Kaggle 원본 데이터
│   │   └── cleaning_data/           # 정제 데이터
│   ├── copy_paste_annotations/      # Copy & Paste 데이터 구조
│   └── processed/                   # YOLO 형식 데이터
│       ├── images/{train,val,test}/
│       ├── labels/{train,val,test}/
│       └── data.yaml.example        # 데이터 설정 예시
├── src/
│   ├── PillDetectionDataset.py       # 알약 Object Detection Dataset 클래스
│   ├── eda_utils.py                  # EDA 및 데이터 시각화 유틸리티
│   ├── pill_transforms.py            # 알약 이미지·Bounding Box 변환 모듈
│   ├── pill_transforms_cp.py         # Copy & Paste 데이터 증강 모듈
│   ├── utils.py                      # 데이터 분할·검증 및 W&B 공통 유틸리티
│   └── yolo/
│       ├── __init__.py
│       ├── dataset_pipeline.py       # YOLO 데이터셋 준비 파이프라인
│       ├── evaluation.py             # 모델 평가 및 mAP 계산
│       ├── runtime.py                # 설정·시드·디바이스·경로 관리
│       ├── submission.py             # 추론 및 Kaggle 제출 파일 생성
│       ├── training.py               # YOLO 학습 및 W&B 연동
│       ├── visualization.py          # GT·예측 결과 시각화
│       ├── yolo_dataset.py           # 데이터를 YOLO 형식으로 변환
│       ├── yolo_dataset_single_object.py # Single-object 데이터셋 생성
│       └── yolo_mapping.py           # YOLO 클래스와 Category ID 매핑
├── outputs/
│   ├── submission/                  # Kaggle 제출 파일
│   ├── predictions/                 # 추론 결과
│   ├── checkpoints/                 # 학습된 모델 가중치
│   └── yolo/                        # Ultralytics 실행 결과
├── wandb/                           # Weights & Biases 실험 로그
├── yolo11s.pt                       # YOLO11s 사전 학습 가중치
├── yolo11m.pt                       # YOLO11m 사전 학습 가중치
├── requirements.txt
└── README.md
```

`data/`, `outputs/`, `wandb/`, 모델 가중치는 실행 과정에서 생성되는 대용량 파일을 포함하므로 Git에서 제외합니다. 저장소에는 데이터 폴더 구조와 `data.yaml.example`만 제공합니다.

## 🛠️ Development Environment

| Category | Stack |
| --- | --- |
| Language | Python |
| Model | YOLO11s, YOLO11m |
| Framework | PyTorch, Ultralytics |
| Environment | Google Colab |
| Dataset | Kaggle |
| Experiment Tracking | Weights & Biases |
| Version Control | Git, GitHub |

## 🚀 Getting Started

프로젝트는 **Google Colab**에서 실행하는 것을 기준으로 구성했습니다. 데이터셋 변환 노트북으로 `data/processed`를 생성한 뒤 통합 노트북에서 학습, 평가, 추론과 제출 파일 생성을 재현할 수 있습니다.

### 1. 저장소 복제 및 의존성 설치

```bash
git clone git@github.com:diokim17/pill-object-detection.git
cd pill-object-detection
pip install -r requirements.txt
```

### 2. 원본 데이터 배치

Kaggle에서 데이터를 내려받은 뒤 다음 구조에 맞게 배치합니다. 실제 이미지와 annotation은 Git에 포함되지 않습니다.

```text
data/dataset/cleaning_data/
├── train_images/
├── train_annotations/
└── test_images/
```

### 3. YOLO 데이터셋 생성

[`notebooks/pill_detection_dataset.ipynb`](notebooks/pill_detection_dataset.ipynb)을 위에서부터 실행합니다. 원본 이미지와 annotation을 분할·변환해 다음 파일을 생성합니다.

```text
data/processed/
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── data.yaml
```

### 4. 최종 모델 학습·평가·추론

`data/processed` 생성이 완료되면 [`notebooks/yolo11_unified.ipynb`](notebooks/yolo11_unified.ipynb)을 위에서부터 실행합니다.

- 데이터 분석: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)
- YOLO 데이터셋 생성: [`notebooks/pill_detection_dataset.ipynb`](notebooks/pill_detection_dataset.ipynb)
- 최종 모델 학습·평가·추론: [`notebooks/yolo11_unified.ipynb`](notebooks/yolo11_unified.ipynb)
- WBF 앙상블: [`notebooks/yolo11s_yolo11m WBF.ipynb`](notebooks/yolo11s_yolo11m%20WBF.ipynb)

실행 순서는 다음과 같습니다.

1. `pill_detection_dataset.ipynb` 실행 및 `data/processed` 생성
2. `yolo11_unified.ipynb` 실행 및 YOLO11 최종 모델 학습
3. Validation/Test 평가와 Test 이미지 추론
4. Kaggle 제출 파일 생성
5. 필요시 `yolo11s_yolo11m WBF.ipynb`으로 앙상블

## 📊 Dataset

- Source: Kaggle
- Task: Object Detection
- Classes: **98**
- Maximum Objects: **4 Pills / Image**
- Label Format: YOLO Bounding Box Format

실제 데이터는 저장소에 포함하지 않습니다. 동일한 디렉터리 구조를 만든 뒤 `data.yaml.example`을 환경에 맞게 수정해 사용합니다.

## 🧪 Experiments

실험은 가능한 한 한 번에 하나의 독립변수만 변경하고, 개별 효과를 확인한 뒤 유효한 요소를 조합하는 방식으로 진행했습니다. 아래 표는 전체 실험 중 성능 변화와 최종 모델 선정 과정을 대표하는 결과입니다.

| Date | Experiment | Model | Key Change | Image Size | Epoch | Kaggle Score | Result |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-08-10 | BASE-003 | YOLO11s | 기본 증강을 끈 YOLO 기준선 | 640 | 100 | **0.34942** | 초기 YOLO 베이스라인 |
| 2026-08-14 | EXP-015 | YOLO11s | 학습 데이터 확대 | 640 | 20 | **0.60291** | 데이터 양과 분포 개선으로 기준선 대비 **+0.25349** 상승 |
| 2026-08-19 | EXP-040 | YOLO11s + YOLO11m | Weighted Box Fusion 앙상블 | 960 | s: 20, m:50 | **0.62409** | 두 모델의 예측 박스를 결합해 단일 모델보다 안정적인 예측 확인 |
| 2026-08-19 | EXP-045 | YOLO11s | CP v3, 1024 입력, ±5° 회전·CLAHE·Gaussian Noise | 1024 | 50 | **0.62623** | 전체 실험 중 최고 Kaggle Score, 최종 모델로 선정 |

### 주요 실험 인사이트

- 모델 크기나 강한 증강보다 **실제 학습 데이터의 양과 분포**가 초기 성능의 핵심 병목이었습니다.
- 학습 데이터를 확대한 EXP-015에서 Kaggle Score가 `0.34942`에서 `0.60291`로 크게 상승했습니다.
- 입력 크기는 `640 → 800 → 960` 구간에서 성능이 개선됐지만 `1280`에서는 하락해, 해상도를 무조건 높이는 방식은 유효하지 않았습니다.
- Copy & Paste는 버전에 따라 효과가 달랐습니다. CP v1·v2는 성능이 감소했지만 CP v3는 개선에 기여했습니다.
- 강한 회전 증강은 성능을 크게 떨어뜨렸고, 최종 실험에서는 ±5°의 약한 회전과 CLAHE·Gaussian Noise를 적용했습니다.
- YOLO11s와 YOLO11m의 WBF 앙상블은 `0.62409`를 기록했으나, 최종 단일 YOLO11s 모델이 `0.62623`으로 가장 높은 점수를 기록했습니다.

## 🏆 Results

최종 모델은 **YOLO11s**를 기반으로 `CP v3` 데이터셋과 기하·화질 증강 데이터를 사용해 `imgsz=1024`, 최대 `50 epochs`, `batch=16`, `seed=42`로 학습했습니다.

| Model | Internal Test mAP@0.5:0.95 | Validation mAP@0.5:0.95 | Kaggle Score |
| --- | ---: | ---: | ---: |
| YOLO11s Final (EXP-045) | **0.9907** | **0.9891** | **0.62623** |

베이스라인 대비 Kaggle Score는 `0.34942 → 0.62623`으로 **0.27681 상승**했습니다.

> EXP-045에서는 입력 해상도와 데이터 증강 조건을 함께 변경했기 때문에, 성능 향상을 특정 변수 하나의 효과로만 해석하지 않습니다.

## 📎 Reports

- [보고서](https://drive.google.com/file/d/1mxFMISxxXrZ0ZjhitfhUiJpCU0IPF8Uy/view?usp=sharing)
- [발표 자료](https://drive.google.com/file/d/1AwIcisf1xv2PL0m_gXmOwdHjnLNsHX3S/view?usp=sharing)
- [실험 기록표](https://docs.google.com/spreadsheets/d/1J9_dE4SCrI2IT4dy5duMgjTgHTYyrKubmzDz2v298zg/edit?gid=1828834460#gid=1828834460)
