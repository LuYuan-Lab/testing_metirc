import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

# ============================================
# 🔧 高级 3D 卷积模块实现
# ============================================
class DynamicConv3D(nn.Module):
    """动态卷积：多卷积核加权求和"""
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, n_kernels=2):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
                                    for _ in range(n_kernels)])
        self.softmax = nn.Softmax(dim=0)
        self.n_kernels = n_kernels

    def forward(self, x):
        weights = self.softmax(torch.ones(self.n_kernels, device=x.device))
        out = sum(w * conv(x) for w, conv in zip(weights, self.convs))
        return out


class Conv3DWithTransformer(nn.Module):
    """3D卷积 + Transformer捕获全局时空特征"""
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, num_heads=4):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.norm = nn.LayerNorm(out_ch)
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=out_ch, nhead=num_heads),
            num_layers=1
        )

    def forward(self, x):
        # x: B,C,T,H,W -> B,T*H*W,C
        B,C,T,H,W = x.shape
        x = self.conv(x)
        x = x.permute(0,2,3,4,1)  # B,T,H,W,C
        x = x.reshape(B,T*H*W,C)
        x = self.trans(x)
        x = x.reshape(B,T,H,W,C).permute(0,4,1,2,3)
        return x

# ============================================
# 🔧 自定义卷积构造接口
# ============================================
def build_conv3d(in_channels, out_channels, kernel_size, stride, padding, conv_type="3d"):
    if conv_type == "3d":
        return nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    
    elif conv_type == "depthwise":
        return nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )
    
    elif conv_type == "2p1d":
        k_t, k_h, k_w = kernel_size
        s_t, s_h, s_w = stride
        p_t, p_h, p_w = padding
        mid_channels = out_channels
        return nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, (1,k_h,k_w),(1,s_h,s_w),(0,p_h,p_w),bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, (k_t,1,1),(s_t,1,1),(p_t,0,0),bias=False)
        )

    elif conv_type == "dynamic":
        return DynamicConv3D(in_channels, out_channels, kernel_size, stride, padding)

    elif conv_type == "deform":
        from mmcv.ops import DeformConv3d
        return DeformConv3d(in_channels, out_channels, kernel_size, stride, padding)

    elif conv_type == "transformer":
        return Conv3DWithTransformer(in_channels, out_channels, kernel_size, stride, padding)

    else:
        raise ValueError(f"Unsupported conv_type: {conv_type}")


# ============================================
# 🔄 替换 Conv2Plus1D 模块
# ============================================
def R2Dmodel_custom_conv(embedding_dim=128, pretrained=True, conv_type="3d", freeze_layers=None):
    """
    构造支持多种卷积的 R2Plus1D_18 模型
    """
    # 1. 加载模型
    weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
    model = r2plus1d_18(weights=weights)

    # 2. 获取 Conv2Plus1D 类型
    temp_model = r2plus1d_18()
    Conv2Plus1D = type(next(m for m in temp_model.modules() if 'Conv2Plus1D' in str(type(m))))
    del temp_model

    # 3. 递归替换
    def replace_conv2plus1d_modules(module):
        for name, child in list(module.named_children()):
            if isinstance(child, Conv2Plus1D):
                convs = [m for m in child.modules() if isinstance(m, nn.Conv3d)]
                if len(convs)==0:
                    print(f"Warning: {name} contains no Conv3d children, skipped")
                    continue
                first_conv = convs[0]
                last_conv = convs[-1]

                in_planes = getattr(first_conv, 'in_channels', None)
                out_planes = getattr(last_conv, 'out_channels', None)

                def to_tuple(x):
                    return x if isinstance(x, tuple) else (x,x,x)

                f_stride = to_tuple(getattr(first_conv,'stride',(1,1,1)))
                l_stride = to_tuple(getattr(last_conv,'stride',(1,1,1)))
                t_stride = l_stride[0]
                h_stride = f_stride[1]
                w_stride = f_stride[2]

                f_pad = to_tuple(getattr(first_conv,'padding',(0,0,0)))
                l_pad = to_tuple(getattr(last_conv,'padding',(0,0,0)))
                t_pad = l_pad[0]
                h_pad = f_pad[1]
                w_pad = f_pad[2]

                kernel_size = (3,3,3)
                stride = (t_stride,h_stride,w_stride)
                padding = (t_pad,h_pad,w_pad)

                new_conv = build_conv3d(in_planes, out_planes, kernel_size, stride, padding, conv_type)
                setattr(module, name, new_conv)
            else:
                replace_conv2plus1d_modules(child)

    replace_conv2plus1d_modules(model)

    # 4. 替换全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, embedding_dim)

    # 5. 冻结指定层
    if freeze_layers:
        for name, mod in model.named_children():
            if name in freeze_layers:
                print(f"Freezing {name}")
                for param in mod.parameters():
                    param.requires_grad = False

    return model


# ============================================
# 🧪 测试示例
# ============================================
if __name__ == "__main__":
    batch_size, C, T, H, W = 2, 3, 16, 112, 112
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = R2Dmodel_custom_conv(
        embedding_dim=128,
        pretrained=True,
        conv_type="dynamic",       # "3d"/"2p1d"/"depthwise"/"dynamic"/"deform"/"transformer"
        freeze_layers=['stem']
    ).to(device)

    dummy = torch.randn(batch_size, C, T, H, W, device=device)
    out = model(dummy)
    print(f"✅ Output shape: {out.shape}")
