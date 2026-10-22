import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_


class MFM(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(MFM, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


if __name__ == "__main__":
    # 设置输入张量大小
    batch_size = 1
    channels = 32  # 输入的通道数
    height, width = 256, 256  # 假设输入图像尺寸为 256*256

    # 创建输入张量列表，假设有两个特征图
    input_tensor1 = torch.randn(batch_size, channels, height, width)  # 输入张量1
    input_tensor2 = torch.randn(batch_size, channels, height, width)  # 输入张量2

    # 初始化 MFM 模块
    mfm = MFM(dim=channels, height=2, reduction=8)
    print(mfm)

    # 前向传播测试
    output = mfm([input_tensor1, input_tensor2])

    # 打印输入和输出的形状
    print(f"Input1 shape: {input_tensor1.shape}")
    print(f"Input2 shape: {input_tensor2.shape}")
    print(f"Output shape: {output.shape}")