# -*- coding: utf-8 -*-
# Code Structure of HS-FPN (https://arxiv.org/abs/2412.10116)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as DCT  # pip install torch_dct
from einops import rearrange
from einops.layers.torch import Rearrange

# ------------------------- utilities -------------------------
def _safe_groups(c: int, base: int = 32) -> int:
    """Return a group number that divides c, preferring <= base."""
    return max(1, math.gcd(c, base))

def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

# ------------------- DCT Spatial Interaction -------------------
class DctSpatialInteraction(nn.Module):
    def __init__(self, in_channels: int, ratio=(0.25, 0.25), isdct: bool = True):
        super().__init__()
        self.ratio = ratio
        self.isdct = isdct  # True in p1&p2, False in p3&p4
        if not self.isdct:
            self.spatial1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

    def _weight(self, H: int, W: int, ref: torch.Tensor) -> torch.Tensor:
        h0, w0 = int(H * self.ratio[0]), int(W * self.ratio[1])
        w = torch.ones((1, 1, H, W), device=ref.device, dtype=ref.dtype)
        w[..., :h0, :w0] = 0
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))
        B, C, H, W = x.shape
        spec = DCT.dct_2d(x, norm='ortho')
        w = self._weight(H, W, x).expand_as(spec)  # B×C×H×W
        spec = spec * w
        mask = DCT.idct_2d(spec, norm='ortho')
        return x * mask

# ------------------- DCT Channel Interaction -------------------
class DctChannelInteraction(nn.Module):
    def __init__(self, in_channels: int, patch=(8, 8), ratio=(0.25, 0.25), isdct: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.h, self.w = patch
        self.ratio = ratio
        self.isdct = isdct

        g = _safe_groups(in_channels, 32)  # auto-fix groups
        self.channel1x1 = nn.Conv2d(in_channels, in_channels, 1, groups=g, bias=False)
        self.channel2x1 = nn.Conv2d(in_channels, in_channels, 1, groups=g, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def _weight(self, H: int, W: int, ref: torch.Tensor) -> torch.Tensor:
        h0, w0 = int(H * self.ratio[0]), int(W * self.ratio[1])
        w = torch.ones((1, 1, H, W), device=ref.device, dtype=ref.dtype)
        w[..., :h0, :w0] = 0
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, H, W = x.size()
        if not self.isdct:  # True in p3&p4
            amaxp = F.adaptive_max_pool2d(x, (1, 1))
            aavgp = F.adaptive_avg_pool2d(x, (1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))
            return x * torch.sigmoid(self.channel2x1(channel))

        # DCT path (p1&p2)
        spec = DCT.dct_2d(x, norm='ortho')
        w = self._weight(H, W, x).expand_as(spec)
        spec = spec * w
        dct_ = DCT.idct_2d(spec, norm='ortho')

        amaxp = F.adaptive_max_pool2d(dct_, (self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_, (self.h, self.w))
        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)

        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        return x * torch.sigmoid(self.channel2x1(channel))

# --------------------------- Conv ---------------------------
class Conv(nn.Module):
    """Conv-BN-Act; defaults to SiLU."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):  # noqa: D401
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        # Use only after BN folding into conv weights.
        return self.act(self.conv(x))

# --------------------------- HFP ---------------------------
class HFP(nn.Module):
    def __init__(self, in_channels: int, ratio=(0.25, 0.25), patch=(8, 8), isdct: bool = True):
        super().__init__()
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct=isdct)
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct=isdct)
        g = _safe_groups(in_channels, 32)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, in_channels)
        )

    def forward(self, x):
        spatial = self.spatial(x)
        channel = self.channel(x)
        return self.out(spatial + channel)

# ----------------------- Pixel Attention -----------------------
class PixelAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # depthwise conv over [x, pattn1] concatenated along a fake time axis
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        # x, pattn1: (B, C, H, W)
        x = x.unsqueeze(2)       # B,C,1,H,W
        pattn1 = pattn1.unsqueeze(2)  # B,C,1,H,W
        x2 = torch.cat([x, pattn1], dim=2)          # B,C,2,H,W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)  # B,2C,H,W
        pattn2 = self.pa2(x2)
        return self.sigmoid(pattn2)

# ------------------ High-Low Frequency Fusion ------------------
class HLPFM(nn.Module):
    """
    HLPFM(dim).forward(x, y) -> (B, dim, H, W)
    - HFP 提取高频增强
    - PixelAttention 生成像素门控
    - result = (x + y) + pattn2 * x + (1 - pattn2) * y
    """
    def __init__(self, dim: int):
        super().__init__()
        # 建议在浅层 isdct=True，深层 isdct=False（外部可再包一层）
        self.HFP = HFP(dim, ratio=(0.25, 0.25), patch=(8, 8), isdct=True)
        self.pa = PixelAttention(dim)
        self.conv = Conv(dim, dim, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # x, y: (B, C, H, W), C=dim
        if x.shape[-2:] != y.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)

        initial = x + y
        pattn1 = self.HFP(initial)
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1.0 - pattn2) * y
        result = self.conv(result)
        return result

# -------------------------- quick test --------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    inp1 = torch.randn(1, 64, 128, 128)
    inp2 = torch.randn(1, 64, 128, 128)
    hlpfm = HLPFM(dim=64)
    out = hlpfm(inp1, inp2)
    print("HLPFM_input_size:", inp1.size())
    print("HLPFM_output_size:", out.size())
