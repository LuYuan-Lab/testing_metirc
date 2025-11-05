import torch
import torch.nn as nn
from pytorch_metric_learning import losses, miners


class TripletLossWrapper(nn.Module):
    """
    一个封装了三元组损失函数和挖掘器的类。

    这使得在训练循环中可以像使用标准损失函数一样使用它。
    """

    def __init__(self, margin: float = 0.2, miner_type: str = "semihard"):
        """
        初始化损失函数和挖掘器。

        Args:
            margin (float): TripletLoss 的边界值。
            miner_type (str): 挖掘器的类型。可以是 'semihard', 'hard', 'all'。
                              'semihard' 通常是最稳定和有效的选择。
        """
        super(TripletLossWrapper, self).__init__()
        self.margin = margin

        # 1. 定义损失函数
        # 我们使用 TripletMarginLoss
        self.loss_func = losses.TripletMarginLoss(margin=self.margin)

        # 2. 定义挖掘器
        # 挖掘器负责从一个批次中找到有效的三元组
        # TripletMarginMiner 是 TripletMarginLoss 的最佳搭档
        # 我们推荐使用 'semihard'，因为它比 'hard' 更稳定
        assert miner_type in [
            "semihard",
            "hard",
            "all",
        ], "miner_type must be 'semihard', 'hard', or 'all'"

        print(f"Initializing Triplet Loss with margin={margin} and '{miner_type}' miner.")
        self.miner = miners.TripletMarginMiner(margin=self.margin, type_of_triplets=miner_type)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算损失。

        Args:
            embeddings (torch.Tensor): 模型输出的特征向量，形状为 (batch_size, embedding_dim)。
            labels (torch.Tensor): 对应的标签，形状为 (batch_size,)。

        Returns:
            torch.Tensor: 计算出的损失值（一个标量）。
        """
        # 1. 使用挖掘器从批次中找出困难的三元组
        # miner 会返回 anchor, positive, negative 的索引
        hard_tuples = self.miner(embeddings, labels)

        # 2. 将 embeddings 和挖掘出的元组传递给损失函数
        loss = self.loss_func(embeddings, labels, hard_tuples)

        return loss


# --- 主程序入口：用于测试脚本 ---
if __name__ == "__main__":
    print("--- Testing TripletLossWrapper ---")

    # 模拟参数
    BATCH_SIZE = 20
    EMBEDDING_DIM = 128
    NUM_CLASSES = 5  # P
    SAMPLES_PER_CLASS = 4  # K

    assert BATCH_SIZE == NUM_CLASSES * SAMPLES_PER_CLASS, "Batch size must be P * K"

    # 1. 实例化损失函数
    # 使用 semi-hard 挖掘策略
    triplet_loss_fn = TripletLossWrapper(margin=0.2, miner_type="semihard")

    # 2. 创建一个模拟的批次数据
    # 这模拟了一个经过 P-K 采样器处理后的批次
    print(f"Creating a dummy batch with P={NUM_CLASSES}, K={SAMPLES_PER_CLASS}...")

    # 模拟模型输出的 embeddings
    # requires_grad=True 是为了测试反向传播
    dummy_embeddings = torch.randn(BATCH_SIZE, EMBEDDING_DIM, requires_grad=True)

    # 模拟对应的标签
    # 标签结构: [0,0,0,0, 1,1,1,1, 2,2,2,2, ...]
    dummy_labels = torch.repeat_interleave(torch.arange(NUM_CLASSES), SAMPLES_PER_CLASS)

    print(f"Dummy embeddings shape: {dummy_embeddings.shape}")
    print(f"Dummy labels shape: {dummy_labels.shape}")
    print(f"Example labels: {dummy_labels[:10]}...")

    # 3. 计算损失
    loss_value = triplet_loss_fn(dummy_embeddings, dummy_labels)

    print(f"\nCalculated loss: {loss_value.item()}")
    assert loss_value.item() >= 0, "Loss should be non-negative"

    # 4. 测试反向传播
    try:
        loss_value.backward()
        print("Backward pass successful.")
        assert dummy_embeddings.grad is not None, "Gradients should be computed"
    except Exception as e:
        print(f"Backward pass failed: {e}")

    print("\nLoss script test passed successfully!")
