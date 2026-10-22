import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSelectionAttention(nn.Module):
    def __init__(self, in_channels, reduction=3):
        super(SemanticSelectionAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_reduce = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.conv1x1_expand = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, x):
        y = self.global_avg_pool(x)  # (B, C, 1, 1)
        # print("y1",y.shape)
        y = self.gelu(self.conv1x1_reduce(y))  # 通道压缩
        # print("y2", y.shape)
        y = self.sigmoid(self.conv1x1_expand(y))  # 通道扩展
        # print("y3", y.shape)
        return x * y  # 加权


class ConvNormGELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvNormGELU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.norm(self.conv(x)))


class SPFUnit(nn.Module):
    def __init__(self, in_channels, out_channels, mask_channels=1):
        super(SPFUnit, self).__init__()
        self.block1 = ConvNormGELU(in_channels + mask_channels, out_channels)
        self.block2 = ConvNormGELU(out_channels, out_channels)
        self.semantic_attention = SemanticSelectionAttention(out_channels)

    def forward(self, x, m):
        if m.shape[2:] != x.shape[2:]:
            m_resized = F.interpolate(m.float(), size=x.shape[2:], mode='bilinear', align_corners=False)
        else:
            m_resized = m.float()

        x_concat = torch.cat([m_resized, x], dim=1)  # 拼接
        # print("x_concat",x_concat.shape)
        x = self.block1(x_concat)
        # print("x1",x.shape)
        x = self.block2(x)
        # print("x2", x.shape)
        x = self.semantic_attention(x)  # 注意力加权
        # print("x3", x.shape)
        return x
if __name__ == '__main__':
    x = torch.randn(6, 3, 256, 256)  # 输入特征图
    m = torch.randn(6, 1, 256, 256)  # 语义掩码
    model = SPFUnit(3, 3)
    output = model(x, m)
    print(output.shape)  # 应输出 (6, 24, 256, 256)
