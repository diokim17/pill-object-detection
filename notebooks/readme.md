# Notebooks

이 폴더는 데이터 탐색(EDA), 실험, 모델 학습 및 결과 분석을 위한 Jupyter Notebook을 관리합니다.

## 목적
1. 데이터 EDA 자료를 종합해서 데이터 탐색, 실험, 모델 학습 및 결과 분석에 대한 전반적인 초기 구조들(.ipynb) 전체 작성
2. 데이터 증강
- Flip을 제외한 train_images파일내 이미지를 통한 기하 증강 
- Copy & Paste를 위해 데이터 불균형을 방지하기 위해 cropped_pills_review 파일 내 이미지들을 이용.
3. 모델
- COCO 데이터 합성에 이어 YOLO 인코더, 디코더를 적용.
- Epochs = 100, patience = 15, IMGSZ = 960. batch size = 8 기본 설정
(03_train.ipynb에서 수정 가능)
[최소 학습 소요 시간은 210분 예상] 
### 초기 01_EDA 실행 시, Colab GPU 연결 설정 필수

4. 결과값
experiments 파일 내 result.csv에 모델의 최종 mAP값 산출됨.