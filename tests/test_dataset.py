"""
测试数据集的基本功能
"""

import os

import numpy as np
import pytest
import torch

from tool.dataset import VideoDataset


def test_dataset_creation(dummy_dataset_path):
    """测试数据集初始化"""
    # 训练集
    train_dataset = VideoDataset(dummy_dataset_path, mode="train")
    assert train_dataset.mode == "train"
    assert hasattr(train_dataset, "video_files")
    assert hasattr(train_dataset, "class_to_idx")
    assert len(train_dataset.class_to_idx) == 3  # 3个类别

    # 验证集
    val_dataset = VideoDataset(dummy_dataset_path, mode="val")
    assert val_dataset.mode == "val"
    assert len(val_dataset.class_to_idx) == 3


def test_invalid_dataset_path():
    """测试无效的数据集路径"""
    with pytest.raises(FileNotFoundError):
        VideoDataset("/nonexistent/path", mode="train")


def test_dataset_transforms(dummy_dataset_path):
    """测试数据变换功能"""
    dataset = VideoDataset(dummy_dataset_path, mode="train", resize_shape=(112, 112))

    # 检查变换组件
    assert hasattr(dataset, "base_transform")
    assert hasattr(dataset, "pil_transform")

    # 测试单帧变换
    dummy_frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    transformed = dataset.base_transform(dummy_frame)

    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (3, 112, 112)  # C,H,W

    # ImageNet 标准化后的范围通常在 [-2.5, 2.5] 左右
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    assert transformed.min() >= -3 and transformed.max() <= 3


def test_class_mapping(dummy_dataset_path):
    """测试类别到索引的映射"""
    dataset = VideoDataset(dummy_dataset_path, mode="train")

    # 检查类别映射
    expected_classes = ["举手", "玩手机", "正常"]
    for cls in expected_classes:
        assert cls in dataset.class_to_idx

    # 验证索引的唯一性
    indices = list(dataset.class_to_idx.values())
    assert len(set(indices)) == len(indices)  # 所有索引都是唯一的


def test_train_val_consistency(dummy_dataset_path):
    """测试训练集和验证集的一致性"""
    train_dataset = VideoDataset(dummy_dataset_path, mode="train")
    val_dataset = VideoDataset(dummy_dataset_path, mode="val")

    # 类别映射应该相同
    assert train_dataset.class_to_idx == val_dataset.class_to_idx


@pytest.mark.skip(reason="需要实际的视频文件")
def test_video_loading_and_processing(test_video_path):
    """测试视频加载和处理（需要实际视频文件）"""
    dataset = VideoDataset(os.path.dirname(os.path.dirname(test_video_path)), mode="train", num_frames=30)

    frames = dataset._sample_and_crop_frames(test_video_path)

    assert len(frames) == dataset.num_frames
    assert all(isinstance(frame, np.ndarray) for frame in frames)
    assert all(frame.shape == (dataset.resize_shape[0], dataset.resize_shape[1], 3) for frame in frames)
