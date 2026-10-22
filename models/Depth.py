import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthEncoder(nn.Module):
    """1ch 深度 -> C 通道，引导特征"""
    def __init__(self, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
        )
    def forward(self, d):
        return self.net(d)

class DepthAwareCombiner(nn.Module):
    """由 (x_low, x_high, d_embed) 预测逐像素逐通道 gate + FiLM"""
    def __init__(self, channels):
        super().__init__()
        self.fuse     = nn.Conv2d(3*channels, channels, 1, bias=True)
        self.to_gate  = nn.Conv2d(channels, channels, 1, bias=True)
        self.to_gamma = nn.Conv2d(2*channels, channels, 1, bias=True)
        self.to_beta  = nn.Conv2d(2*channels, channels, 1, bias=True)

    def forward(self, xl, xh, de):
        h  = torch.cat([xl, xh, de], dim=1)
        h  = self.fuse(h)
        g  = torch.sigmoid(self.to_gate(h))          # [B,C,H,W]
        mix= g * xh + (1. - g) * xl
        z  = torch.cat([mix, de], dim=1)
        gamma = torch.tanh(self.to_gamma(z))
        beta  = self.to_beta(z)
        return (1. + gamma) * mix + beta

class CBAM_Lite(nn.Module):
    def __init__(self, c, rr=4):
        super().__init__()
        hid = max(1, c // rr)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(c, hid), nn.ReLU(inplace=True),
            nn.Linear(hid, c), nn.Sigmoid()
        )
        self.spa = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.avg(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        x = x * w
        avg = torch.mean(x, 1, keepdim=True)
        mx, _ = torch.max(x, 1, keepdim=True)
        a = torch.sigmoid(self.spa(torch.cat([avg, mx], 1)))
        return x * a

class AFEM_DepthGuidedBlock(nn.Module):
    """单尺度深度引导融合：输入 x_low(来自上一尺度上采样或skip)、x_high(当前尺度特征)、depth(1ch)"""
    def __init__(self, in_ch_low, in_ch_high, out_ch):
        super().__init__()
        assert out_ch == in_ch_high
        self.proj_low = nn.Conv2d(in_ch_low, out_ch, 1, bias=False) if in_ch_low != out_ch else nn.Identity()
        self.de   = DepthEncoder(out_ch)
        self.comb = DepthAwareCombiner(out_ch)
        self.cbam = CBAM_Lite(out_ch, rr=4)
        self.tail = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.ReLU(inplace=True)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
        )

    def forward(self, x_low, x_high, depth):
        H, W = x_high.shape[2:]
        x_low = F.interpolate(x_low, size=(H, W), mode='bilinear', align_corners=False)
        depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)

        # normalize depth to [0,1]
        d_min = depth.amin(dim=(2,3), keepdim=True)
        d_max = depth.amax(dim=(2,3), keepdim=True)
        depth = (depth - d_min) / (d_max - d_min + 1e-6)

        x_low = self.proj_low(x_low)
        de    = self.de(depth)
        y     = self.comb(x_low, x_high, de)
        y     = self.cbam(y)
        y     = self.tail(y)
        return y + self.refine(y)

class FinalFusionHead(nn.Module):
    """最终像素自适应融合：y = α*img_depth + (1-α)*img_seg"""
    def __init__(self, in_img_ch=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_img_ch*2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, img_depth, img_seg):
        a = self.gate(torch.cat([img_depth, img_seg], dim=1))
        return a * img_depth + (1. - a) * img_seg, a
