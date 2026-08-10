

from PillDetectionDataset import PillDetectionDataset, detection_collate_fn
from pathlib import Path

from collections import defaultdict

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

print("현재 작업 디렉터리:", Path.cwd())
print("스크립트 위치:", Path(__file__).resolve())

dataset_root = "./data/dataset/cleaning_data" #현재 git 폴더 구조에 맞도록 변경

dataset = PillDetectionDataset(
    root=dataset_root,
    transforms=None,
    label_offset=1,  # torchvision detection 기준. YOLO용이면 0 권장.
    strict=False,  # 일부 annotation 누락/불일치가 있어도 경고 후 진행
    validate_image_size=True,
)

print(f"Number of images: {len(dataset)}")
print(f"Number of pill classes: {dataset.num_classes}")
print("First sample summary:")
print(dataset.get_sample_summary(0))

image, target, metadata = dataset[0]
print(f"Image size: {image.size}")
print(f"Boxes shape: {target['boxes'].shape}")
print(f"Boxes: {target['boxes']}")
print(f"Labels: {target['labels'].tolist()}")
print(f"Pill IDs: {metadata['pill_ids']}")
print(f"First object metadata: {metadata['objects'][0]}")
print(f"Second object metadata: {metadata['objects'][1]}")
print(f"Third object metadata: {metadata['objects'][2]}")
print(f"Fourth object metadata: {metadata['objects'][3]}")

dataset.print_class_statistics()





# ============================================================
# 1. 기본 설정
# ============================================================
#dataset_root = "./sprint_ai_project1_data"
dataset_root = "./data/dataset/cleaning_data"  #현재 git 폴더 구조에 맞도록 변경
random_seed = 42
batch_size = 4

# Albumentations 등을 사용하는 경우 여기에 지정합니다.
# 예:
# train_transforms = A.Compose(
#     [...],
#     bbox_params=A.BboxParams(
#         format="pascal_voc",
#         label_fields=["labels"],
#     ),
# )
#
# valid/test에는 resize 및 tensor 변환 등 평가에 필요한 변환만 지정합니다.
train_transforms = None
eval_transforms = None


# ============================================================
# 2. 분할 기준으로 사용할 데이터셋 생성
# ============================================================
# transforms=None으로 생성하여 데이터 분할 정보만 확인합니다.
split_base_dataset = PillDetectionDataset(
    root=dataset_root,
    transforms=None,
    label_offset=1,
    strict=False,
    validate_image_size=True,
)

print(f"전체 이미지 수: {len(split_base_dataset)}")
print(f"알약 클래스 수: {split_base_dataset.num_classes}")


# ============================================================
# 3. combination_key 단위로 이미지 인덱스 그룹화
# ============================================================
group_to_indices = defaultdict(list)

for index, sample in enumerate(split_base_dataset.samples):
    combination_key = sample["metadata"]["combination_key"]
    group_to_indices[combination_key].append(index)

group_keys = sorted(group_to_indices.keys())

print(f"전체 combination_key 수: {len(group_keys)}")


# ============================================================
# 4. combination_key를 Train / Validation / Test = 8:1:1로 분할
# ============================================================
# 먼저 전체 그룹의 80%를 train, 20%를 validation+test로 분리합니다.
train_groups, temp_groups = train_test_split(
    group_keys,
    test_size=0.2,
    random_state=random_seed,
    shuffle=True,
)

# 남은 20%를 절반으로 나눠 validation 10%, test 10%로 만듭니다.
valid_groups, test_groups = train_test_split(
    temp_groups,
    test_size=0.5,
    random_state=random_seed,
    shuffle=True,
)

train_groups = set(train_groups)
valid_groups = set(valid_groups)
test_groups = set(test_groups)


def groups_to_indices(groups, group_mapping):
    """combination_key 집합을 원본 데이터셋 인덱스 목록으로 변환합니다."""
    return sorted(
        index
        for group in groups
        for index in group_mapping[group]
    )


train_indices = groups_to_indices(train_groups, group_to_indices)
valid_indices = groups_to_indices(valid_groups, group_to_indices)
test_indices = groups_to_indices(test_groups, group_to_indices)


# ============================================================
# 5. combination_key 누수 여부 검증
# ============================================================
assert train_groups.isdisjoint(valid_groups), (
    "Train과 Validation에 중복 combination_key가 있습니다."
)
assert train_groups.isdisjoint(test_groups), (
    "Train과 Test에 중복 combination_key가 있습니다."
)
assert valid_groups.isdisjoint(test_groups), (
    "Validation과 Test에 중복 combination_key가 있습니다."
)

# 모든 이미지가 정확히 한 번씩 분배되었는지 확인
all_split_indices = train_indices + valid_indices + test_indices

assert len(all_split_indices) == len(split_base_dataset), (
    "분할된 이미지 수와 전체 이미지 수가 일치하지 않습니다."
)
assert len(set(all_split_indices)) == len(split_base_dataset), (
    "두 개 이상의 데이터셋에 중복된 이미지 인덱스가 있습니다."
)


# ============================================================
# 6. Train용 / 평가용 데이터셋을 별도로 생성
# ============================================================
train_base_dataset = PillDetectionDataset(
    root=dataset_root,
    transforms=train_transforms, #none
    label_offset=1,
    strict=False,
    validate_image_size=True,
)

eval_base_dataset = PillDetectionDataset(
    root=dataset_root,
    transforms=eval_transforms, #none
    label_offset=1,
    strict=False,
    validate_image_size=True,
)


# 세 데이터셋의 샘플 순서가 같은지 검증합니다.
assert len(split_base_dataset) == len(train_base_dataset)
assert len(split_base_dataset) == len(eval_base_dataset)

for index in range(len(split_base_dataset)):
    split_file = split_base_dataset.samples[index]["metadata"]["file_name"]
    train_file = train_base_dataset.samples[index]["metadata"]["file_name"]
    eval_file = eval_base_dataset.samples[index]["metadata"]["file_name"]

    assert split_file == train_file == eval_file, (
        f"데이터셋의 샘플 순서가 다릅니다: index={index}, "
        f"split={split_file}, train={train_file}, eval={eval_file}"
    )


# ============================================================
# 7. 동일한 인덱스를 적용해 Subset 생성
# ============================================================
train_dataset = Subset(
    train_base_dataset,
    train_indices,
)

valid_dataset = Subset(
    eval_base_dataset,
    valid_indices,
)

test_dataset = Subset(
    eval_base_dataset,
    test_indices,
)


# ============================================================
# 8. 분할 결과 출력
# ============================================================
total_images = len(split_base_dataset)

print("\n===== 데이터셋 분할 결과 =====")
print(
    f"Train      : {len(train_dataset):4d}장 "
    f"({len(train_dataset) / total_images:.1%}), "
    f"{len(train_groups)}개 조합"
)
print(
    f"Validation : {len(valid_dataset):4d}장 "
    f"({len(valid_dataset) / total_images:.1%}), "
    f"{len(valid_groups)}개 조합"
)
print(
    f"Test       : {len(test_dataset):4d}장 "
    f"({len(test_dataset) / total_images:.1%}), "
    f"{len(test_groups)}개 조합"
)


# ============================================================
# 9. DataLoader 생성
# ============================================================
# Windows + Jupyter 환경에서는 우선 num_workers=0을 권장합니다.
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=detection_collate_fn,
    pin_memory=torch.cuda.is_available(),
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=detection_collate_fn,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=detection_collate_fn,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# 10. DataLoader 동작 확인
# ============================================================
train_images, train_targets, train_metadata = next(iter(train_loader))

print("\n===== Train batch 확인 =====")
print(f"배치 이미지 수: {len(train_images)}")
print(f"첫 이미지 크기: {train_images[0].size}")
print(f"첫 target boxes 크기: {train_targets[0]['boxes'].shape}")
print(f"첫 target labels: {train_targets[0]['labels'].tolist()}")
print(f"첫 이미지 combination_key: {train_metadata[0]['combination_key']}")