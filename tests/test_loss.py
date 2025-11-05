"""
测试 TripletLossWrapper 损失函数
"""

import torch

from tool.loss import TripletLossWrapper


def test_triplet_loss_initialization():
    """测试损失函数的初始化"""
    loss_fn = TripletLossWrapper(margin=0.2, miner_type="semihard")
    assert hasattr(loss_fn, "loss_func")
    assert hasattr(loss_fn, "miner")
    assert loss_fn.margin == 0.2


def test_triplet_loss_forward():
    """测试损失函数的前向传播"""
    loss_fn = TripletLossWrapper(margin=0.2, miner_type="semihard")
    batch_size = 12
    embedding_dim = 128
    num_classes = 3
    samples_per_class = 4

    # 创建模拟数据
    embeddings = torch.randn(batch_size, embedding_dim, requires_grad=True)
    labels = torch.repeat_interleave(torch.arange(num_classes), samples_per_class)

    # 计算损失
    loss = loss_fn(embeddings, labels)

    # 基本检查
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert loss.item() >= 0  # 损失值应该非负
    assert loss.numel() == 1  # 应该是标量


def test_triplet_loss_backward():
    """测试损失函数的反向传播"""
    loss_fn = TripletLossWrapper(margin=0.2, miner_type="semihard")
    batch_size = 12
    embedding_dim = 128
    num_classes = 3
    samples_per_class = 4

    # 创建模拟数据
    embeddings = torch.randn(batch_size, embedding_dim, requires_grad=True)
    labels = torch.repeat_interleave(torch.arange(num_classes), samples_per_class)

    # 前向和反向传播
    loss = loss_fn(embeddings, labels)
    loss.backward()

    # 检查梯度
    assert embeddings.grad is not None
    assert not torch.isnan(embeddings.grad).any()  # 梯度不应该有 NaN
    assert not torch.isinf(embeddings.grad).any()  # 梯度不应该有 Inf


def test_triplet_loss_edge_cases():
    """测试损失函数的边界情况"""
    loss_fn = TripletLossWrapper(margin=0.2, miner_type="semihard")

    # 测试单类别情况
    embeddings = torch.randn(4, 128, requires_grad=True)
    labels = torch.zeros(4, dtype=torch.long)
    loss = loss_fn(embeddings, labels)
    assert not torch.isnan(loss)

    # 测试每类只有一个样本的情况
    embeddings = torch.randn(3, 128, requires_grad=True)
    labels = torch.arange(3)
    loss = loss_fn(embeddings, labels)
    assert not torch.isnan(loss)
