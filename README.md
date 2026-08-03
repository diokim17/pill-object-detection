# 💊 Pill Object Detection

이미지 인식 기술을 활용하여 사진 속 알약의 종류와 위치를 검출하는 Object Detection 프로젝트입니다.

> Codeit AI Engineer Bootcamp Team Project  
> Project Period : **2026.08.05 ~ 2026.08.21**

---

# 📌 Project Overview

헬스케어 스타트업 **Health Eat**의 AI 엔지니어링 팀이라는 가정하에 진행하는 프로젝트입니다.

사용자가 모바일 애플리케이션으로 복용 중인 약을 촬영하면,
이미지 인식 기술을 이용하여 사진 속 알약을 인식하고 정보를 제공할 수 있는 AI 모델을 개발합니다.

본 프로젝트의 목표는

- 사진 속 최대 **4개의 알약(Object)** 검출
- 각 알약의 **Bounding Box** 예측
- 각 알약의 **Class** 분류
- 지속적인 성능 개선을 통한 높은 Object Detection 성능 달성

입니다.

---

# Team

| Name | GitHub | Role | 협업 일지 |
|------|---------|------|--------|
| 김도영 | [@diokim17](https://github.com/diokim17) |  |[도영님 협업일지](https://docs.google.com/document/d/1NbZmXKJG7K_dmMsLEOKm8Qg8OaqXq9PPsOr-C1_MVJc/edit?tab=t.0) |
| 김민협 | [@seolwoom](https://github.com/seolwoom) |  |[민협님 협업일지](https://docs.google.com/document/d/1NbZmXKJG7K_dmMsLEOKm8Qg8OaqXq9PPsOr-C1_MVJc/edit?tab=t.4gadpvsip6lf) |
| 이원영 | [@1young-codes](https://github.com/1young-codes) |  |[원영님 협업일지](https://docs.google.com/document/d/1NbZmXKJG7K_dmMsLEOKm8Qg8OaqXq9PPsOr-C1_MVJc/edit?tab=t.qt241c3bv39)  |
| 이재웅 | [@mitguchin](https://github.com/mitguchin) |  |[재웅님 협업일지](https://docs.google.com/document/d/1NbZmXKJG7K_dmMsLEOKm8Qg8OaqXq9PPsOr-C1_MVJc/edit?tab=t.x7hx0cls3eoj)|
| 황세희 | [@hwangsiiii](https://github.com/hwangsiiii) |  |[세희님 협업일지](https://docs.google.com/document/d/1NbZmXKJG7K_dmMsLEOKm8Qg8OaqXq9PPsOr-C1_MVJc/edit?tab=t.phfoaioiz36c)|

## 회의록 및 일정 
[회의록](https://docs.google.com/document/d/1NbZmXKJG7K_dmMsLEOKm8Qg8OaqXq9PPsOr-C1_MVJc/edit?tab=t.fwuvz6mbikcq)

[일정 관리](https://docs.google.com/spreadsheets/d/1tYkRh30PQU-JUwD4WYt2OkT3x0hBXk1jCB_lgWlEViM/edit?gid=193608105#gid=193608105)

---

# 프로젝트 구조

다음은 예시입니다.

```text
pill-object-detection/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_train.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_inference.ipynb
│
├── outputs/
│   ├── figures/
│   ├── predictions/
│   └── submissions/
│
├── docs/
│
├── requirements.txt
└── README.md
```

---

# 개발 환경

| Category | Stack |
|----------|------|
| Language | Python |
| Framework | PyTorch |
| Environment | Google Colab |
| Dataset | Kaggle |
| Version Control | Git, GitHub |

---

# 🚀 Getting Started

프로젝트는 **Google Colab** 환경에서 실행하는 것을 기준으로 개발합니다.

모든 팀원이 동일한 환경에서 실행할 수 있도록 구성하며,

- Kaggle Dataset 다운로드
- 모델 학습
- 추론
- 결과 저장

과정을 Colab Notebook 하나로 재현 가능하도록 개발하는 것을 목표로 합니다.

---

# 📊 Dataset

- Source : Kaggle
- Task : Object Detection
- Maximum Objects : 4 Pills / Image

데이터에 대한 자세한 내용은 프로젝트 진행 중 업데이트 예정입니다.

---

# 📈 Experiments

프로젝트 진행 중 실험 내용을 지속적으로 기록합니다.

| Date | Model | Description | Score |
|------|--------|------------|-------|
| - | - | - | - |

---

# 🏆 Results

프로젝트 종료 후 업데이트 예정




