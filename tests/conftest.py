"""
测试配置和共享夹具
"""

from pathlib import Path

import pytest
import torch

# 测试资源目录（存放测试用的视频片段、图片等）
TEST_RESOURCES = Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def test_video_path():
    """返回测试用视频路径"""
    video_path = TEST_RESOURCES / "test_video.mp4"
    if not video_path.exists():
        pytest.skip(f"测试视频不存在：{video_path}")
    return str(video_path)


@pytest.fixture(scope="session")
def dummy_video_tensor():
    """生成一个用于测试的视频张量"""
    batch_size = 2
    channels = 3
    frames = 30  # 匹配默认的 num_frames
    height = width = 112  # 匹配默认的 resize_shape
    return torch.randn(batch_size, channels, frames, height, width)


@pytest.fixture(scope="session")
def device():
    """返回可用的计算设备（优先 CUDA）"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_dataset_path(tmp_path):
    """创建一个测试用的数据集目录结构"""
    data_root = tmp_path / "data"

    # 创建训练集和验证集目录
    for mode in ["train", "val"]:
        mode_dir = data_root / mode
        mode_dir.mkdir(parents=True)

        # 创建类别目录
        for class_name in ["举手", "玩手机", "正常"]:
            class_dir = mode_dir / class_name
            class_dir.mkdir()

            # 创建一个空的示例视频文件
            # 实际测试时可以复制真实的测试视频
            (class_dir / "dummy.mp4").touch()

    return str(data_root)


@pytest.fixture
def sample_embeddings():
    """生成用于损失函数测试的示例特征向量"""
    batch_size = 12  # P=3, K=4
    embedding_dim = 128
    return torch.randn(batch_size, embedding_dim, requires_grad=True)


@pytest.fixture
def sample_labels():
    """生成对应的标签"""
    num_classes = 3
    samples_per_class = 4
    return torch.repeat_interleave(torch.arange(num_classes), samples_per_class)
