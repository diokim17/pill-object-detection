# Baseline

## Faster R CNN
### Dataset 클래스 구현 요청
현재 annotation 데이터는 **COCO 형식의 개별 JSON 파일**로 구성되어 있습니다.

각 JSON 파일에는 다음 정보가 포함되어 있습니다.

- `images`: 이미지 정보 1건
- `annotations`: 해당 이미지에 포함된 객체 annotation 1건
- `categories`: 해당 객체의 클래스 정보 1건

하나의 실제 이미지에는 여러 개의 알약 객체가 존재하며, 각 객체에 대응하는 JSON 파일이 별도로 존재합니다.

따라서 **JSON 파일 하나를 Dataset 샘플 하나로 처리하지 않고, 이미지 1장을 Dataset 샘플 하나로 구성**해야 합니다.

동일한 `file_name`을 가진 여러 JSON 파일을 이미지 기준으로 묶고, 각 JSON에 포함된 annotation을 하나의 `target` dictionary에 누적해 주세요.

---

## 반환 형식

`__getitem__()`은 다음 형식으로 반환해 주세요.

```python
image, target
```

`target`에는 다음 정보가 포함되어야 합니다.

```python
target = {
    "boxes": boxes,
    "labels": labels,
    "image_id": image_id,
    "area": area,
    "iscrowd": iscrowd,
}
```

---

## boxes

JSON의 `annotations["bbox"]`는 COCO 형식입니다.

```text
[x_min, y_min, width, height]
```

Faster R-CNN에서 사용할 수 있도록 다음 형식으로 변환해 주세요.

```text
[x_min, y_min, x_max, y_max]
```

변환 예시

```python
x_min, y_min, width, height = bbox

x_max = x_min + width
y_max = y_min + height
```

예를 들어

```python
[644, 845, 189, 190]
```

이라면

```python
[644, 845, 833, 1035]
```

로 변환됩니다.

조건

- shape : `[객체 수, 4]`
- dtype : `torch.float32`

---

## labels

각 객체의 클래스는 `category_id`를 사용합니다.

다만 현재 원본 `category_id`는

```text
1900
3351
4120
...
```

처럼 큰 숫자이므로 그대로 사용하지 말고,

**1부터 시작하는 연속된 class id로 재매핑**해 주세요.

예시

```python
category_mapping = {
    1900: 1,
    3351: 2,
    4120: 3,
}
```

조건

- shape : `[객체 수]`
- dtype : `torch.int64`

> **YOLO와 Faster R-CNN 모두 동일한 원본 category mapping을 사용해 주세요.**
>
> 단,
>
> - Faster R-CNN : background = 0, 실제 클래스 = 1부터 시작
> - YOLO : 실제 클래스 = 0부터 시작

---

## image_id

같은 이미지에 속한 annotation은 동일한 `image_id`를 사용해야 합니다.

만약 JSON 내부의 `image_id`가 이미지마다 고유하지 않거나 동일 이미지에서도 다르다면,

Dataset 내부에서 이미지 기준의 고유한 index를 생성하여 사용해 주세요.

예시

```python
target["image_id"] = torch.tensor(
    [image_index],
    dtype=torch.int64,
)
```

---

## area

JSON의 `area` 값을 사용해 주세요.

조건

- shape : `[객체 수]`
- dtype : `torch.float32`

---

## iscrowd

JSON의 `iscrowd` 값을 사용해 주세요.

현재 데이터에서는 모두 `0`으로 처리하면 됩니다.

조건

- shape : `[객체 수]`
- dtype : `torch.int64`

---

## 반환 예시

이미지 한 장에 객체가 4개라면 다음과 같은 형태가 되어야 합니다.

```python
target = {
    "boxes": torch.tensor(
        [
            [100.0, 150.0, 220.0, 280.0],
            [300.0, 200.0, 430.0, 340.0],
            [500.0, 400.0, 650.0, 550.0],
            [644.0, 845.0, 833.0, 1035.0],
        ],
        dtype=torch.float32,
    ),
    "labels": torch.tensor(
        [1, 3, 2, 4],
        dtype=torch.int64,
    ),
    "image_id": torch.tensor(
        [0],
        dtype=torch.int64,
    ),
    "area": torch.tensor(
        [
            15600.0,
            18200.0,
            22500.0,
            35910.0,
        ],
        dtype=torch.float32,
    ),
    "iscrowd": torch.tensor(
        [0, 0, 0, 0],
        dtype=torch.int64,
    ),
}
```

---

## DataLoader용 collate_fn

이미지마다 객체 수가 다르므로 다음 `collate_fn`도 함께 구현해 주세요.

```python
def collate_fn(batch):
    return tuple(zip(*batch))
```

---

## 데이터 검증 항목

구현 과정에서 아래 항목도 함께 확인해 주세요.

- 동일한 `file_name`의 JSON이 하나의 이미지 샘플로 묶이는지
- JSON 하나가 Dataset 샘플 하나로 생성되지 않는지
- `bbox`가 이미지 범위를 벗어나지 않는지
- `width > 0`, `height > 0`인지
- 변환 후 `x_max > x_min`, `y_max > y_min`인지
- `category_id`가 mapping에 존재하는지
- `boxes`, `labels`, `area`, `iscrowd`의 길이가 모두 동일한지

---

# 가장 중요한 요구사항

- 동일한 `file_name`을 가진 여러 JSON 파일을 **하나의 이미지 샘플로 묶어 주세요.**
- 여러 JSON의 annotation 정보를 하나의 `target` dictionary에 누적해 주세요.
- bbox는 COCO 형식 `[x, y, width, height]`에서 Faster R-CNN 형식 `[x_min, y_min, x_max, y_max]`로 변환해 주세요.
- 원본 `category_id`는 그대로 사용하지 말고 **background를 제외한 1부터 시작하는 연속 class id**로 재매핑해 주세요.
- 이미지 한 장에 객체가 4개라면 `boxes`, `labels`, `area`, `iscrowd`에도 각각 4개의 객체 정보가 들어가야 합니다.