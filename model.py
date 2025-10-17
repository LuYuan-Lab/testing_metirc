import torch
import torch.nn as nn
import torch.nn.functional as F
# --- 修改 Start ---
# 我们不仅导入模型本身，还导入对应的权重枚举
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
# --- 修改 End ---


class R2Plus1DNet(nn.Module):
    def __init__(self, num_classes=5, embedding_dim=128, pretrained=True):
        """
        初始化模型。
        Args:
            num_classes (int): 项目中的类别总数。虽然主要用 embedding，但某些损失函数可能需要。
            embedding_dim (int): 最终输出的嵌入向量维度。
            pretrained (bool): 是否加载 Kinetics-400 上的预训练权重。
        """
        super(R2Plus1DNet, self).__init__()
        
        # --- 修改 Start ---
        # 根据 pretrained 参数决定使用哪种权重
        if pretrained:
            # 使用官方推荐的枚举对象指定预训练权重
            weights = R2Plus1D_18_Weights.KINETICS400_V1
        else:
            weights = None
        
        # 加载模型时传入权重对象
        self.backbone = r2plus1d_18(weights=weights)
        # --- 修改 End ---
        
        # 2. 改造模型的分类头 (Embedding Head)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.embedding_head = nn.Linear(in_features, embedding_dim)
        
    def forward(self, x):
        """
        定义模型的前向传播。
        """
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


# --- 模型测试代码 (保持不变) ---
if __name__ == '__main__':
    # 模拟输入数据
    dummy_video_batch = torch.randn(4, 3, 16, 224, 224) 
    
    print("正在实例化模型...")
    model = R2Plus1DNet(num_classes=5, embedding_dim=64, pretrained=True)
    print("模型实例化成功！")
    
    model.eval()
    
    print("\n正在进行一次前向传播测试...")
    with torch.no_grad():
        output_embedding = model(dummy_video_batch)
    print("前向传播成功！")
    
    print(f"\n输入张量的形状: {dummy_video_batch.shape}")
    print(f"输出嵌入向量的形状: {output_embedding.shape}")
    print(f"预期输出形状是: (4, 64)")
    
    norms = torch.norm(output_embedding, p=2, dim=1)
    print(f"\n输出向量的 L2 范数（长度）: \n{norms}")
    print("如果归一化成功，上述值应该都非常接近 1.0")
