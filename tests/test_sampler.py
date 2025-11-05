"""
测试 PKBatchSampler 功能
"""

import pytest
import torch
from torch.utils.data import Dataset

from train import PKBatchSampler


class DummyDataset(Dataset):
    def __init__(self, num_classes=3, samples_per_class=10):
        self.video_files = []
        for class_idx in range(num_classes):
            for _ in range(samples_per_class):
                self.video_files.append((f"dummy_{class_idx}_{_}.mp4", class_idx))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        return self.video_files[idx]


def test_pk_sampler_initialization():
    """测试采样器的初始化"""
    dataset = DummyDataset(num_classes=3, samples_per_class=10)
    sampler = PKBatchSampler(dataset, p=2, k=4)

    assert sampler.p == 2
    assert sampler.k == 4
    assert len(sampler.labels) == 3  # 应该找到所有类别


def test_pk_sampler_batch_composition():
    """测试批次的组成"""
    dataset = DummyDataset(num_classes=3, samples_per_class=10)
    p, k = 2, 4
    sampler = PKBatchSampler(dataset, p=p, k=k)

    for batch_indices in sampler:
        # 检查批次大小
        assert len(batch_indices) == p * k

        # 获取这个批次中的标签
        batch_labels = [dataset.video_files[i][1] for i in batch_indices]
        unique_labels = set(batch_labels)

        # 验证类别数量
        assert len(unique_labels) == p

        # 验证每个类别的样本数
        for label in unique_labels:
            assert batch_labels.count(label) == k


def test_pk_sampler_deterministic():
    """测试采样器的确定性（使用相同种子）"""
    dataset = DummyDataset(num_classes=3, samples_per_class=10)

    # 设置 Python 的随机种子和 numpy 的随机种子
    import random

    import numpy as np

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    sampler1 = PKBatchSampler(dataset, p=2, k=4)
    sampler1_batches = list(sampler1)

    # 重置所有随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    sampler2 = PKBatchSampler(dataset, p=2, k=4)
    sampler2_batches = list(sampler2)

    # 比较完整的批次列表而不是单个批次
    assert len(sampler1_batches) == len(sampler2_batches)
    for batch1, batch2 in zip(sampler1_batches, sampler2_batches):
        # 比较排序后的批次，因为类别的选择顺序可能不同
        assert sorted(batch1) == sorted(batch2)


def test_pk_sampler_with_insufficient_samples():
    """测试样本数不足的情况"""
    # 创建一个类别样本数不均衡的数据集
    dataset = DummyDataset(num_classes=3, samples_per_class=3)  # 每类只有3个样本

    # 当 k=4 时应该抛出异常，因为每类只有3个样本
    with pytest.raises(ValueError, match="Not enough samples"):
        sampler = PKBatchSampler(dataset, p=2, k=4)
        next(iter(sampler))


def test_pk_sampler_coverage():
    """测试采样器的覆盖范围"""
    dataset = DummyDataset(num_classes=3, samples_per_class=10)
    sampler = PKBatchSampler(dataset, p=2, k=4)

    # 收集所有批次的索引
    all_indices = set()
    for batch_indices in sampler:
        all_indices.update(batch_indices)

    # 验证是否使用了大部分数据点
    assert len(all_indices) > len(dataset) * 0.5  # 至少使用了50%的数据
