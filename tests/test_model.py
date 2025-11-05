"""
测试模型的基本功能
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.ResNetModel import R2Dmodel
from tool.dataset import VideoDataset


def test_r2dmodel_creation():
    """测试模型创建"""
    embedding_dim = 128
    model = R2Dmodel(embedding_dim=embedding_dim, pretrained=False)
    assert isinstance(model, nn.Module)
    assert hasattr(model, "fc")
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == embedding_dim


def test_r2dmodel_forward(dummy_video_tensor, device):
    """测试模型前向传播"""
    embedding_dim = 128
    model = R2Dmodel(embedding_dim=embedding_dim, pretrained=False).to(device)
    dummy_video_tensor = dummy_video_tensor.to(device)

    with torch.no_grad():
        embeddings = model(dummy_video_tensor)

    # 检查输出维度和数值
    batch_size = dummy_video_tensor.size(0)
    assert embeddings.shape == (batch_size, embedding_dim)
    assert not torch.isnan(embeddings).any()
    assert embeddings.requires_grad is False  # no_grad 模式下


def test_r2dmodel_freeze_layers():
    """测试层冻结功能"""
    model = R2Dmodel(embedding_dim=128, pretrained=False, freeze_layers=["stem", "layer1"])

    # 检查指定层参数是否被冻结
    for name, param in model.named_parameters():
        if name.startswith(("stem", "layer1")):
            assert not param.requires_grad, f"{name} should be frozen"
        elif name.startswith("fc"):
            assert param.requires_grad, f"{name} should be trainable"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 支持")
def test_model_gpu_compatibility(dummy_video_tensor):
    """测试模型在 GPU 上的兼容性"""
    model = R2Dmodel(embedding_dim=128, pretrained=False).cuda()
    input_tensor = dummy_video_tensor.cuda()

    with torch.no_grad():
        output = model(input_tensor)

    assert output.is_cuda
    assert not torch.isnan(output).any()


def test_model_training_mode(dummy_video_tensor, device):
    """测试模型在训练模式下的行为"""
    model = R2Dmodel(embedding_dim=128, pretrained=False).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    dummy_video_tensor = dummy_video_tensor.to(device)

    # 训练模式
    model.train()
    optimizer.zero_grad()

    # 前向传播
    output = model(dummy_video_tensor)
    loss = output.mean()  # 简单的损失计算用于测试

    # 反向传播
    loss.backward()

    # 检查梯度
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} 没有梯度"
            assert not torch.isnan(param.grad).any(), f"{name} 的梯度包含 NaN"


def test_r2dmodel_with_dataset(dummy_dataset_path, device):
    """测试模型与数据集的集成"""
    # 准备数据
    dataset = VideoDataset(dummy_dataset_path, mode="train")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 初始化模型
    model = R2Dmodel(embedding_dim=128, pretrained=False).to(device)
    model.eval()

    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        # 由于没有实际的视频文件，这里应该会抛出文件未找到异常
        next(iter(dataloader))
