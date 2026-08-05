# Outputs

이 폴더는 **모델 추론 결과 및 Kaggle 제출 파일**을 저장합니다.

## 목적

* Kaggle 제출용 CSV 파일 관리
* 제출 이력 및 결과 비교
* 최종 제출 파일 보관

## 제출 파일 형식

제출 파일은 **CSV 형식**이며, 아래 컬럼을 포함해야 합니다.

| Column          | Description          |
| --------------- | -------------------- |
| `annotation_id` | 객체별 고유 ID (임의의 고유 값) |
| `image_id`      | 이미지 파일명의 숫자          |
| `category_id`   | 예측 클래스 ID            |
| `bbox_x`        | Bounding Box의 x 좌표   |
| `bbox_y`        | Bounding Box의 y 좌표   |
| `bbox_w`        | Bounding Box의 너비     |
| `bbox_h`        | Bounding Box의 높이     |
| `score`         | 예측 Confidence Score  |

### 예시

```csv
annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score
1,1,1,156,247,211,456,0.91
2,1,24,498,40,460,474,0.78
3,1,11,579,700,260,473,0.27
4,1,69,527,83,398,416,0.27
5,3,1,143,236,204,135,0.89
6,3,24,512,41,388,432,0.78
7,3,11,556,677,257,435,0.20
```

## 작성 규칙

* 제출 파일은 **CSV 형식**으로 저장합니다.
* `image_id`는 이미지 파일명의 숫자를 사용합니다.
* `annotation_id`는 각 객체마다 고유한 값을 사용합니다.
* 하나의 객체는 하나의 행(Row)으로 작성합니다.
* Bounding Box는 `(bbox_x, bbox_y, bbox_w, bbox_h)` 형식을 사용합니다.
* `score`는 모델이 예측한 Confidence Score를 입력합니다.

## 파일명 규칙

제출 이력을 쉽게 관리할 수 있도록 다음 형식을 권장합니다.

```text
submission_baseline.csv
submission_yolo11s.csv
submission_ssd.csv
submission_faster_rcnn.csv
submission_ensemble.csv
submission_best.csv
```

또는 실험 번호를 포함하여 관리할 수 있습니다.

```text
submission_exp01.csv
submission_exp02.csv
submission_exp03.csv
```

> **참고:** 제출 파일 형식을 준수하지 않거나 컬럼이 누락될 경우 Kaggle에서 정상적으로 채점되지 않을 수 있습니다.
