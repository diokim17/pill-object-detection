# YOLO 모듈화 구조

```text
pill-object-detection/
├── config_yolo_cp.yaml
├── notebooks/
│   └── yolo11s_cp_modular.ipynb
└── src/
    └── yolo/
        ├── __init__.py
        ├── runtime.py
        ├── dataset_pipeline.py
        ├── training.py
        ├── evaluation.py
        ├── submission.py
        ├── yolo_dataset.py
        └── yolo_mapping.py
```

## 책임 분리

## 책임 분리

- `runtime.py`
  - config 로드
  - seed 고정
  - CUDA/CPU device 결정
  - 프로젝트 공통 경로 관리 및 검증

- `dataset_pipeline.py`
  - 원본 데이터를 YOLO 학습 포맷으로 변환
  - 기본 YOLO 데이터셋 생성
  - Copy&Paste 증강 데이터셋 생성
  - train/val/test 데이터셋 구조 검증

- `training.py`
  - W&B 설정
  - YOLO 학습 파라미터 구성
  - 모델 학습 실행
  - `best.pt` 경로 관리

- `evaluation.py`
  - validation/test split 평가
  - mAP@0.50, mAP@0.75 등 YOLO 평가 지표 계산
  - 대회 평가 기준인 mAP@0.75:0.95 계산

- `submission.py`
  - Kaggle test 이미지 추론
  - YOLO class ID를 원본 `category_id`로 역매핑
  - submission CSV 생성
  - 제출 파일 형식 및 값 검증
  - 추론 중 GPU/RAM 메모리 관리

- `yolo_dataset.py`
  - 원본 Pill Detection 데이터셋을 YOLO 학습 포맷으로 변환
  - COCO 형식 annotation의 bbox/class 정보를 YOLO label 형식으로 변환
  - Faster R-CNN과 동일한 train/val/test split을 기준으로 YOLO 데이터셋 구성
  - YOLO 학습에 필요한 `images/`, `labels/`, `data.yaml` 생성

- `yolo_mapping.py`
  - 원본 `category_id`와 YOLO class ID 간 매핑 관리
  - YOLO 학습용 연속 class ID 생성
  - 추론 결과의 YOLO class ID를 원본 `category_id`로 복원
  - 학습과 Kaggle 제출 사이의 클래스 ID 일관성 보장

- `notebook`
  - config 로드
  - 데이터셋 준비
  - 학습
  - 평가
  - 추론 및 submission 생성
  - 위 모듈들을 호출하는 실험 실행 순서만 담당
