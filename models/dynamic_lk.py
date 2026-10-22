# ============== Dynamic Large-Kernel (ConvAtt2) & Triple-Branch ==============
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ConvAtt2(nn.Module):
    """
    动态(小核) + 静态(大核) 的注意力卷积块：
    - 输入/输出通道 = in_channels
    - 前 att_channels 走动态+静态，其余通道保留后用 1x1 融合
    """
    def __init__(self, in_channels, att_channels=16, lk_size=13, sk_size=3, reduction=2):
        super().__init__()
        assert 0 < att_channels <= in_channels, "att_channels 必须在 (0, in_channels] 区间"
        self.in_channels = in_channels
        self.att_channels = att_channels
        self.idt_channels = in_channels - att_channels
        self.lk_size = lk_size
        self.sk_size = sk_size

        mid = max(1, att_channels // reduction)
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(att_channels, mid, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(mid, att_channels * sk_size * sk_size, 1, bias=True)
        )
        # 让动态分支训练初期几乎不扰动
        nn.init.zeros_(self.kernel_gen[-1].weight)
        nn.init.zeros_(self.kernel_gen[-1].bias)

        # 静态大核（可学习参数）
        self.lk_filter = nn.Parameter(torch.randn(att_channels, att_channels, lk_size, lk_size))
        nn.init.kaiming_normal_(self.lk_filter, mode='fan_out', nonlinearity='relu')

        # 1x1 融合，保持通道数不变
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape
        # 通道拆分
        F_att, F_idt = torch.split(x, [self.att_channels, self.idt_channels], dim=1)

        # 动态 3x3 卷积核：每个样本、每个通道一套
        kernel = self.kernel_gen(F_att).reshape(B * self.att_channels, 1, self.sk_size, self.sk_size)
        F_att_re = rearrange(F_att, 'b c h w -> 1 (b c) h w')
        out_dk = F.conv2d(F_att_re, kernel, padding=self.sk_size // 2, groups=B * self.att_channels)
        out_dk = rearrange(out_dk, '1 (b c) h w -> b c h w', b=B, c=self.att_channels)

        # 静态大核
        out_lk = F.conv2d(F_att, self.lk_filter, padding=self.lk_size // 2)

        out_att = out_lk + out_dk
        out = torch.cat([out_att, F_idt], dim=1)
        out = self.fusion(out)
        return out


class DynLKTriple(nn.Module):
    """
    三分支动态大核模块：
    - 给定三个大核尺寸（例如 [7, 11, 15]）
    - 每个分支一个 ConvAtt2（独立参数），每个分支输出 dim 通道
    - 最终按通道拼接：输出形状 [B, 3*dim, H, W]
    """
    def __init__(self, dim, lk_sizes=(7, 11, 15), att_ratio=4, sk_size=3, reduction=2):
        super().__init__()
        att_channels = max(1, dim // att_ratio)
        self.branches = nn.ModuleList([
            ConvAtt2(in_channels=dim,
                     att_channels=att_channels,
                     lk_size=ks,
                     sk_size=sk_size,
                     reduction=reduction)
            for ks in lk_sizes
        ])

    def forward(self, x):
        outs = [b(x) for b in self.branches]  # 每个都是 [B, dim, H, W]
        return torch.cat(outs, dim=1)         # [B, 3*dim, H, W]
