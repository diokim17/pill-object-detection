from collections import defaultdict

import wandb
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split


# ============================================================
# combination_key 기준 그룹 생성
# ============================================================

def build_group_mapping(dataset):
    """
    Dataset의 combination_key를 기준으로
    이미지 index를 그룹화합니다.

    Returns:
        group_to_indices:
            {
                combination_key: [dataset_index, ...]
            }

        group_keys:
            정렬된 combination_key 목록
    """

    group_to_indices = defaultdict(list)

    for index, sample in enumerate(
        dataset.samples
    ):

        combination_key = (
            sample["metadata"]
            ["combination_key"]
        )

        group_to_indices[
            combination_key
        ].append(index)

    group_keys = sorted(
        group_to_indices.keys()
    )

    return (
        group_to_indices,
        group_keys,
    )


# ============================================================
# combination_key 기준 Train / Val / Test Split
# ============================================================

def make_group_split(
    group_keys,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    """
    combination_key 단위로 Train / Validation / Test를 분할합니다.

    동일 combination_key가 여러 split에 들어가는 것을 방지합니다.

    Args:
        group_keys:
            전체 combination_key 목록

        train_ratio:
            Train 비율

        val_ratio:
            Validation 비율

        test_ratio:
            Test 비율

        seed:
            random seed

    Returns:
        train_groups
        valid_groups
        test_groups
    """

    # ========================================================
    # 1. Split 비율 검증
    # ========================================================

    assert abs(
        train_ratio
        + val_ratio
        + test_ratio
        - 1.0
    ) < 1e-8, (
        "train_ratio + val_ratio + test_ratio의 합은 "
        "1.0이어야 합니다."
    )

    # ========================================================
    # 2. Train / Temp 분할
    # ========================================================

    temp_ratio = (
        val_ratio
        + test_ratio
    )

    train_groups, temp_groups = train_test_split(
        group_keys,
        test_size=temp_ratio,
        random_state=seed,
        shuffle=True,
    )

    # ========================================================
    # 3. Validation / Test 분할
    # ========================================================

    relative_test_ratio = (
        test_ratio
        / temp_ratio
    )

    valid_groups, test_groups = train_test_split(
        temp_groups,
        test_size=relative_test_ratio,
        random_state=seed,
        shuffle=True,
    )

    return (
        set(train_groups),
        set(valid_groups),
        set(test_groups),
    )


# ============================================================
# Group → Dataset Index 변환
# ============================================================

def groups_to_indices(
    groups,
    group_mapping,
):
    """
    combination_key 집합을 Dataset index 목록으로 변환합니다.
    """

    return sorted(
        index
        for group in groups
        for index in group_mapping[group]
    )


# ============================================================
# Split 검증
# ============================================================

def validate_split(
    dataset,
    train_groups,
    valid_groups,
    test_groups,
    train_indices,
    valid_indices,
    test_indices,
):
    """
    Train / Validation / Test split에
    중복이나 누락이 없는지 검증합니다.
    """

    # ========================================================
    # combination_key 중복 검증
    # ========================================================

    assert train_groups.isdisjoint(
        valid_groups
    ), (
        "Train과 Validation에 "
        "중복 combination_key가 있습니다."
    )

    assert train_groups.isdisjoint(
        test_groups
    ), (
        "Train과 Test에 "
        "중복 combination_key가 있습니다."
    )

    assert valid_groups.isdisjoint(
        test_groups
    ), (
        "Validation과 Test에 "
        "중복 combination_key가 있습니다."
    )

    # ========================================================
    # 이미지 index 중복 검증
    # ========================================================

    train_index_set = set(
        train_indices
    )

    valid_index_set = set(
        valid_indices
    )

    test_index_set = set(
        test_indices
    )

    assert train_index_set.isdisjoint(
        valid_index_set
    )

    assert train_index_set.isdisjoint(
        test_index_set
    )

    assert valid_index_set.isdisjoint(
        test_index_set
    )

    # ========================================================
    # 전체 Dataset 분배 검증
    # ========================================================

    all_split_indices = (
        train_indices
        + valid_indices
        + test_indices
    )

    assert len(
        all_split_indices
    ) == len(
        dataset
    ), (
        "분할된 이미지 개수와 "
        "전체 Dataset 이미지 개수가 일치하지 않습니다."
    )

    assert len(
        set(all_split_indices)
    ) == len(
        dataset
    ), (
        "하나의 이미지가 여러 split에 "
        "중복 포함되어 있습니다."
    )

    return True


# ============================================================
# W&B 초기화
# ============================================================

def init_wandb(cfg):
    """
    config.yaml 설정을 기반으로
    Weights & Biases Run을 초기화합니다.

    cfg.wandb.enabled=False이면
    None을 반환합니다.
    """

    if not cfg.wandb.enabled:
        return None

    # 최초 실행 시 API Key 입력 가능
    wandb.login()

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        tags=list(cfg.wandb.tags),

        config=OmegaConf.to_container(
            cfg,
            resolve=True,
        ),
    )

    print(
        "W&B Run:",
        run.name,
    )

    return run