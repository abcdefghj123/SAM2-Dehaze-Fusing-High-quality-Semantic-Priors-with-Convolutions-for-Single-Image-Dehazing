import torch
import torch.nn as nn
import torch.nn.functional as F

class SPTUnit(nn.Module):
    def __init__(self, in_channels: int, prior_channels: int, sam_channels: int):
        """
        Args:
            in_channels: F_i 的通道数
            prior_channels: 先验 P 的通道数
            sam_channels: 语义图 M_SAM 的通道数
        """
        super().__init__()

        # 生成 F'_i：注意输入通道是 C + C_sam
        self.concat_fusion = nn.Sequential(
            nn.Conv2d(in_channels + sam_channels, in_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        )

        # 特征分支 f(·)
        self.feature_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        )

        # 先验更新 g(P) + P
        self.prior_branch = nn.Sequential(
            nn.Conv2d(prior_channels, prior_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(prior_channels, prior_channels, 3, 1, 1)
        )

        # H_i 通道对齐（若 H_i 通道与 C 不同则做 1×1 投影；相同也可保留用于轻微再标定）
        self.h_proj = nn.Conv2d(in_channels, in_channels, 1)

        # 用 H_i 生成像素门控（可选，更稳）
        self.h_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.Sigmoid()
        )

    def forward(self, F_i, M_SAM, P_i, H_i):
        """
        Args:
            F_i: (B, C, H, W)
            M_SAM: (B, C_sam, H?, W?)
            P_i: (B, C_prior, H, W)
            H_i: (B, C or other, H?, W?)  — 来自 HLPFM 的特征
        Returns:
            F_spt_i: (B, C, H, W)
            P_i_plus_1: (B, C_prior, H, W)
        """
        # 尺寸对齐
        if F_i.shape[-2:] != M_SAM.shape[-2:]:
            M_SAM = F.interpolate(M_SAM, size=F_i.shape[-2:], mode='bilinear', align_corners=False)
        if F_i.shape[-2:] != H_i.shape[-2:]:
            H_i = F.interpolate(H_i, size=F_i.shape[-2:], mode='bilinear', align_corners=False)

        # F'（图中的 Feature map F'）
        F_prime_i = self.concat_fusion(torch.cat([F_i, M_SAM], dim=1))   # (B,C,H,W)

        # 特征分支
        F_feature = self.feature_branch(F_prime_i)                       # (B,C,H,W)

        # 先验更新
        P_i_plus_1 = self.prior_branch(P_i) + P_i                        # (B,Cp,H,W)

        # H_i 融合：先通道对齐，再作为像素门控
        H_i = self.h_proj(H_i)                                           # (B,C,H,W)
        h_gate = self.h_gate(H_i)                                        # (B,C,H,W) ∈ [0,1]
        F_hybrid = F_feature * (1 + h_gate)                              # 轻微放大由 H_i 指示的区域

        # 先验调制（采用 sigmoid 门控更稳）
        P_gate = torch.sigmoid(P_i_plus_1.to(F_hybrid.dtype))
        F_spt_i = F_hybrid * P_gate + F_prime_i                          # ⊗ + ⊕

        return F_spt_i


class SAMPrior(nn.Module):
    def __init__(self, in_channels: int, prior_channels: int):
        """
        Args:
            in_channels: I_LQ 与 M_SAM 拼接后的通道数 (C_img + C_sam)
            prior_channels: 输出先验通道数
        """
        super().__init__()
        self.prior_fusion = nn.Sequential(
            nn.Conv2d(in_channels, prior_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(prior_channels, prior_channels, 3, 1, 1)
        )

    def forward(self, I_LQ, M_SAM):
        if I_LQ.shape[-2:] != M_SAM.shape[-2:]:
            M_SAM = F.interpolate(M_SAM, size=I_LQ.shape[-2:], mode='bilinear', align_corners=False)
        P = self.prior_fusion(torch.cat([I_LQ, M_SAM], dim=1))
        return P
