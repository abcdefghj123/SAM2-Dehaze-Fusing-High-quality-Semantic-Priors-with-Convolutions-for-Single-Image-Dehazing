import torch
import torch.nn as nn
import torch.nn.functional as F
from HLPFM import HLPFM
class SPTUnit(nn.Module):
    def __init__(self, in_channels, prior_channels):
        """
        初始化SPT单元
        Args:
            in_channels: 输入特征图 F_i 的通道数
            prior_channels: SAM先验表示 P 的通道数
        """
        super(SPTUnit, self).__init__()

        # f 函数：卷积 + ReLU + 卷积
        self.feature_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

        self.prior_branch = nn.Sequential(
            nn.Conv2d(prior_channels, prior_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(prior_channels, prior_channels, kernel_size=3, padding=1)
        )

        # 用于生成 F'_i
        self.concat_fusion = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, F_i, M_SAM, P_i):
        """
        前向传播
        Args:
            F_i: 当前输入特征图 (B, C, H, W)
            M_SAM: SAM语义分割图 (B, C_sam, H, W)
            P_i: SAM先验表示 (B, C_prior, H, W)
        Returns:
            F_spt_i: 输出增强特征图 (B, C, H, W)
            P_i_plus_1: 更新后的SAM先验表示 (B, C_prior, H, W)
        """
        # 生成 F'_i
        if F_i.size()[-2:] != M_SAM.size()[-2:]:
            M_SAM = F.interpolate(M_SAM, size=F_i.size()[-2:], mode='bilinear', align_corners=False)

        F_prime_i = self.concat_fusion(torch.cat([F_i, M_SAM], dim=1))

        # 特征分支
        F_feature = self.feature_branch(F_prime_i)

        # 先验分支
        P_i_plus_1 = self.prior_branch(P_i) + P_i

        # 融合操作
        F_spt_i = F_feature * P_i_plus_1 + F_prime_i

        return F_spt_i


class SAMPrior(nn.Module):
    def __init__(self, in_channels, prior_channels):
        """
        初始化 SAM 先验模块
        Args:
            in_channels: 输入图像与 SAM 语义图的通道数总和
            prior_channels: 生成的 SAM 先验表示通道数
        """
        super(SAMPrior, self).__init__()

        self.prior_fusion = nn.Sequential(
            nn.Conv2d(in_channels, prior_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(prior_channels, prior_channels, kernel_size=3, padding=1)
        )

    def forward(self, I_LQ, M_SAM):
        """
        前向传播++6
        Args:
            I_LQ: 低质量输入图像 (B, C_img, H, W)
            M_SAM: SAM 语义分割图 (B, C_sam, H, W)
        Returns:
            P: SAM先验表示 (B, C_prior, H, W)
        """
        # 拼接低质量图像与语义图
        if I_LQ.size()[-2:] != M_SAM.size()[-2:]:
            M_SAM = F.interpolate(M_SAM, size=I_LQ.size()[-2:], mode='bilinear', align_corners=False)
        # print("M",M_SAM.shape)
        # print("I",I_LQ.shape)
        concat_input = torch.cat([I_LQ, M_SAM], dim=1)

        # 生成先验表示 P
        #（消融实验2）
        P = self.prior_fusion(concat_input)
        #P = self.prior_fusion(I_LQ)
        return P
