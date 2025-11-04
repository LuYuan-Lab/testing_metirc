import torch
import torch.nn as nn
from torch import Tensor
from torch.hub import load_state_dict_from_url
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

# ------------------ Conv / Block 定义 ------------------
class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, midplanes: int, stride: int = 1, padding: int = 1):
        super().__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes, out_planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False
            ),
        )

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return stride, stride, stride

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        super().__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)
        self.conv1 = nn.Sequential(conv_builder(inplanes, planes, midplanes, stride), nn.BatchNorm3d(planes), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(conv_builder(planes, planes, midplanes), nn.BatchNorm3d(planes))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class R2Plus1dStem(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Conv3d(3, 45, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3), bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

# ------------------ VideoResNet 构造 ------------------
class VideoResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock]], conv_makers: Sequence[Type[Conv2Plus1D]],
                 layers: List[int], stem: Callable[..., nn.Module], num_classes: int = 400):
        super().__init__()
        self.inplanes = 64
        self.stem = stem()
        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, conv_builder, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

# ------------------ r2plus1d_18 构造函数 ------------------
def r2plus1d_18(weights: bool = True):
    model = VideoResNet(
        BasicBlock,
        [Conv2Plus1D]*4,
        [2,2,2,2],
        R2Plus1dStem,
        num_classes=400
    )
    if weights:
        url = "https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth"
        state_dict = load_state_dict_from_url(url, progress=True)
        model.load_state_dict(state_dict)
        print("Loaded pretrained weights from Kinetics-400")
    return model

# ------------------ R2Dmodel 接口 ------------------
def R2Dmodel(embedding_dim: int = 128, pretrained: bool = True, freeze_layers: list = None):
    model = r2plus1d_18(weights=pretrained)

    if freeze_layers is not None:
        for name, module in model.named_children():
            if name in freeze_layers:
                for param in module.parameters():
                    param.requires_grad = False
        # 支持嵌套层冻结 layer1.0 等
        for layer_name in freeze_layers:
            if '.' in layer_name:
                names = layer_name.split('.')
                sub_module = model
                try:
                    for n in names:
                        sub_module = getattr(sub_module, n) if not n.isdigit() else sub_module[int(n)]
                    for param in sub_module.parameters():
                        param.requires_grad = False
                except Exception as e:
                    print(f"Warning: Could not freeze {layer_name}: {e}")

    # 替换 fc 为 embedding_dim
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, embedding_dim)

    if freeze_layers is not None and 'fc' in freeze_layers:
        for param in model.fc.parameters():
            param.requires_grad = True

    return model

# ------------------ 测试 forward ------------------
if __name__ == "__main__":
    EMBEDDING_DIM = 128
    embedding_model = R2Dmodel(
        embedding_dim=EMBEDDING_DIM,
        pretrained=True,
        freeze_layers=['stem','layer1','layer2','layer3','layer4','fc']
    )

    batch_size, C, T, H, W = 4, 3, 16, 112, 112
    dummy_video = torch.randn(batch_size, C, T, H, W)
    embeddings = embedding_model(dummy_video)
    print("Output shape:", embeddings.shape)
    assert embeddings.shape == (batch_size, EMBEDDING_DIM)
    print("Forward pass successful!")
