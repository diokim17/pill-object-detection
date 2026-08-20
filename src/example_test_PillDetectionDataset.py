

from PillDetectionDataset import PillDetectionDataset, detection_collate_fn, prepare_ultralytics_dataset_from_dataset
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
    image_dir_name="train_images",
    annotation_dir_name="train_annotations",
    label_offset=0,
    strict=False,
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
#print(f"First object metadata: {metadata['objects'][0]}")
#print(f"Second object metadata: {metadata['objects'][1]}")
#print(f"Third object metadata: {metadata['objects'][2]}")
#print(f"Fourth object metadata: {metadata['objects'][3]}")

dataset.print_class_statistics()

result = prepare_ultralytics_dataset_from_dataset(
    dataset=dataset,
    output_dir="./data/processed",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
)

print(result["yaml_path"])
print(result["statistics"])