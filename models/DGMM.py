# dgam.py
# -*- coding: utf-8 -*-
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Conv -> Norm -> Act 小组件（用于上下采样处）"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1, act=True, norm=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, groups=groups, bias=not norm)
        self.norm = nn.BatchNorm2d(out_ch) if norm else nn.Identity()
        self.act  = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DepthGuidedAttentionModule(nn.Module):
    """
    DGAM: Depth-Guided Attention Module
    参考论文《Depth-Guided Dehazing Network for Long-Range Aerial Scenes》中的 DGAM 思路实现

    输入:
        feat:  [B, C, H, W]   去雾主干特征
        depth: [B, 1, H, W]   深度图 (0~1), 若未归一化, 可在 forward 中 set normalize_depth=True

    超参:
        in_channels          : 输入特征通道 C
        proj_channels        : 注意力计算的投影维度 d_k (= d_v)，默认自动设为 max(32, C // proj_ratio)
        proj_ratio           : 从 C 映射到投影维度的压缩比（仅当 proj_channels 未指定时生效）
        reduce_factor        : 空间下采样倍数（默认 4，即 1/4 分辨率；实现用两次 stride=2）
        attn_dropout         : 注意力得分的 dropout
        proj_dropout         : 投影后的特征 dropout
        use_depth_norm       : 是否对 depth 做轻量归一化（min-max 到 0~1，按 batch）
        fuse_add             : 融合方式，当前为 residual add（论文 Eq.(5)）
    """
    def __init__(
        self,
        in_channels: int,
        proj_channels: Optional[int] = None,
        proj_ratio: int = 4,
        reduce_factor: int = 4,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_depth_norm: bool = False,
    ):
        super().__init__()
        assert reduce_factor in (2, 4), "reduce_factor 仅支持 2 或 4（实现为 stride=2 下采样的次数）"
        self.in_channels   = in_channels
        self.reduce_factor = reduce_factor
        self.use_depth_norm = use_depth_norm

        d_k = proj_channels if proj_channels is not None else max(32, in_channels // proj_ratio)
        d_v = d_k  # 论文里 Q/K/V 同一维度更简单

        # -------- 下采样到 1/4 分辨率（两次 stride=2）--------
        red_ch = max(32, in_channels // 2)
        self.feat_down1  = ConvBNAct(in_channels, red_ch, k=3, s=2)
        self.feat_down2  = ConvBNAct(red_ch, red_ch, k=3, s=2) if reduce_factor == 4 else nn.Identity()
        self.depth_down1 = ConvBNAct(1,        16,    k=3, s=2)  # 深度分支轻量
        self.depth_down2 = ConvBNAct(16,       16,    k=3, s=2) if reduce_factor == 4 else nn.Identity()

        # -------- 线性投影（用 1x1 conv 等价于逐像素线性层）--------
        self.to_q = nn.Conv2d(16,     d_k, kernel_size=1, bias=True)   # Q from depth~
        self.to_k = nn.Conv2d(red_ch, d_k, kernel_size=1, bias=True)   # K from feat~
        self.to_v = nn.Conv2d(red_ch, d_v, kernel_size=1, bias=True)   # V from feat~

        self.attn_drop = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

        # -------- 上采样回原分辨率并融合 --------
        # 先把 d_v -> in_channels（1x1），再用两次 stride=2 的反卷积/插值上采样
        self.proj_out = nn.Conv2d(d_v, in_channels, kernel_size=1, bias=True)

        # 上采样：用最近邻插值 + 轻量卷积（更稳，避免棋盘格）
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBNAct(in_channels, in_channels, k=3, s=1),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBNAct(in_channels, in_channels, k=3, s=1),
        ) if reduce_factor == 4 else nn.Identity()

        # 输出融合前的轻量调制（可帮助稳定训练）
        self.out_norm = nn.BatchNorm2d(in_channels)
        self.out_act  = nn.SiLU(inplace=True)

        # 可选的门控（让模块在不需要时学会“旁路”）
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(16, in_channels // 16), 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(16, in_channels // 16), in_channels, 1, bias=True),
            nn.Sigmoid()
        )

        # 参数初始化（更稳定）
        self._init_weights()

    @staticmethod
    def _depth_min_max_norm(d: torch.Tensor, eps: float = 1e-6):
        # d: [B,1,H,W] -> 逐样本归一化到 0~1，避免 batch 间尺度差异
        B = d.size(0)
        d = d.view(B, -1)
        d_min = d.min(dim=1, keepdim=True)[0]
        d_max = d.max(dim=1, keepdim=True)[0]
        d = (d - d_min) / (d_max - d_min + eps)
        return d.view(B, 1, -1)  # 先返回展平的形状，便于与后续逻辑衔接

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Q/K/V/Proj 用较小 std
        for m in [self.to_q, self.to_k, self.to_v, self.proj_out]:
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        feat:  [B, C, H, W]
        depth: [B, 1, H, W]
        return: [B, C, H, W]
        """
        B, C, H, W = feat.shape
        assert depth.shape[:2] == (B, 1), f"depth shape should be [B,1,H,W], got {tuple(depth.shape)}"

        # -------- 下采样 --------
        f1 = self.feat_down1(feat)       # [B, C', H/2, W/2]
        f2 = self.feat_down2(f1)         # [B, C', H/4, W/4] 或 Identity
        d1 = self.depth_down1(depth)     # [B, 16,  H/2, W/2]
        d2 = self.depth_down2(d1)        # [B, 16,  H/4, W/4] 或 Identity

        # -------- 线性投影 -> Q/K/V --------
        # 深度可选归一化（轻量 min-max），保证数值稳定；先在低分辨率上操作
        if self.use_depth_norm:
            # 把 d2 展平做 min-max，再 reshape 回去
            Bd, Cd, Hd, Wd = d2.shape  # Cd=16
            d2_flat = d2.view(Bd, Cd, Hd * Wd)
            # 对通道内做 min-max，考虑到 16 通道来自卷积特征，这里更常见做逐通道归一化
            d2_min = d2_flat.min(dim=2, keepdim=True)[0]
            d2_max = d2_flat.max(dim=2, keepdim=True)[0]
            d2 = (d2_flat - d2_min) / (d2_max - d2_min + 1e-6)
            d2 = d2.view(Bd, Cd, Hd, Wd)

        Q = self.to_q(d2)  # [B, d_k, h, w]
        K = self.to_k(f2)  # [B, d_k, h, w]
        V = self.to_v(f2)  # [B, d_v, h, w]

        # -------- 展平空域为 Token 序列，做 QK^T -> softmax -> AV --------
        Bq, dq, h, w = Q.shape
        _, dk, _, _ = K.shape
        _, dv, _, _ = V.shape
        N = h * w

        # [B, d, h, w] -> [B, d, N] -> [B, N, d]
        Ql = Q.view(B, dq, N).transpose(1, 2).contiguous()  # [B, N, d_k]
        Kl = K.view(B, dk, N).transpose(1, 2).contiguous()  # [B, N, d_k]
        Vl = V.view(B, dv, N).transpose(1, 2).contiguous()  # [B, N, d_v]

        # 注意力：以 depth->Q 作为查询，feat->K 作为键
        scale = float(dq) ** -0.5
        attn = torch.bmm(Ql, Kl.transpose(1, 2)) * scale    # [B, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 聚合：Att * V
        FA = torch.bmm(attn, Vl)  # [B, N, d_v]
        FA = self.proj_drop(FA)

        # 恢复回特征图
        FA = FA.transpose(1, 2).contiguous().view(B, dv, h, w)  # [B, d_v, h, w]

        # -------- 映射回 in_channels 并上采样到原分辨率 --------
        FA = self.proj_out(FA)  # [B, C, h, w]
        FA = self.up1(FA)       # [B, C, ~2h, ~2w]
        FA = self.up2(FA)       # 若 reduce_factor==4，再上采一次 -> [B, C, H, W]

        # -------- 融合（残差）--------
        out = self.out_act(self.out_norm(FA))
        gate = self.gate(feat)  # [B, C, 1, 1]
        out = feat + gate * out
        return out


# =========================== 简单自测 ===========================
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 64, 256, 256
    x = torch.randn(B, C, H, W)
    d = torch.rand(B, 1, H, W)  # 深度图 0~1

    dgam = DepthGuidedAttentionModule(
        in_channels=C,
        proj_channels=64,    # 或者不填，自动 C//4
        reduce_factor=4,     # 对应论文 1/4 尺度做注意力
        attn_dropout=0.0,
        proj_dropout=0.0,
        use_depth_norm=False
    )

    y = dgam(x, d)
    print("input :", x.shape)
    print("depth :", d.shape)
    print("output:", y.shape)
    # 反向传播检查
    y.mean().backward()
    total_params = sum(p.numel() for p in dgam.parameters() if p.requires_grad)
    print(f"DGAM trainable params: {total_params/1e6:.3f} M")
