import torch
import torch.nn as nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18


def R2Dmodel(embedding_dim: int = 128, pretrained: bool = True, freeze_layers: list = None):
    # 1. 加载模型
    if pretrained:
        print("Loading pretrained weights from Kinetics-400.")
        weights = R2Plus1D_18_Weights.KINETICS400_V1
    else:
        weights = None

    model = r2plus1d_18(weights=weights)

    # 2.1 冻结指定层
    if freeze_layers is not None:
        print(f"Freezing specific layers: {freeze_layers}")
        for name, module in model.named_children():
            if name in freeze_layers:
                for param in module.parameters():
                    param.requires_grad = False
        # 兼容嵌套层（如layer1, layer2等包含多个block）
        # 支持传递如layer1.0, layer2.1等
        for layer_name in freeze_layers:
            if "." in layer_name:
                names = layer_name.split(".")
                sub_module = model
                try:
                    for n in names:
                        # 支持数字索引
                        if n.isdigit():
                            sub_module = sub_module[int(n)]
                        else:
                            sub_module = getattr(sub_module, n)
                    for param in sub_module.parameters():
                        param.requires_grad = False
                except Exception as e:
                    print(f"Warning: Could not freeze {layer_name}: {e}")

    # 3. 替换最后一层（投影头/嵌入层）
    # 获取原始分类头的输入特征维度
    num_ftrs = model.fc.in_features

    # 创建一个新的线性层，将特征映射到你指定的 embedding_dim 维度
    # 这个层现在是你的 "Projection Head"
    model.fc = nn.Linear(num_ftrs, embedding_dim)

    if freeze_layers is not None and "fc" in freeze_layers:
        for param in model.fc.parameters():
            param.requires_grad = True

    return model


# 主程序入口：用于测试脚本是否能正常工作
if __name__ == "__main__":
    # --- 测试参数 ---
    EMBEDDING_DIM = 128  # 目标输出维度

    # --- 创建模型 ---
    print("Creating a model for metric learning (embedding generation)...")
    # 示例：冻结前两层（stem和layer1）
    embedding_model = R2Dmodel(
        embedding_dim=EMBEDDING_DIM,
        pretrained=True,
        freeze_layers=[
            "stem",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "fc",
        ],  # 你可以自定义要冻结的层
    )

    # --- 模拟输入并进行一次前向传播 ---
    # 视频模型的标准输入形状: [Batch, Channels, Time, Height, Width]
    batch_size = 4
    num_frames = 16
    channels = 3
    height = 112
    width = 112

    # 创建一个假的视频张量
    dummy_video_tensor = torch.randn(batch_size, channels, num_frames, height, width)

    print(f"\nTesting forward pass with a dummy tensor of shape: {dummy_video_tensor.shape}")

    # 将输入传递给模型
    embeddings = embedding_model(dummy_video_tensor)

    # 检查输出形状是否正确
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Expected output shape: [{batch_size}, {EMBEDDING_DIM}]")

    # 断言检查
    assert embeddings.shape == (batch_size, EMBEDDING_DIM), "Output shape is incorrect!"

    print("\nModel script test passed successfully!")
    print(embedding_model)
