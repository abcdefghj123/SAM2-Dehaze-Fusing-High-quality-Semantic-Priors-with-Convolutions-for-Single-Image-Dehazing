import torch
import torch.nn as nn
import torch.nn.functional as F


class DGAM(nn.Module):
    """
    Depth-Guided Attention Module
    输入:
        D: 深度图 (B,1,H,W)
        F: 特征图 (B,C,H,W)
    输出:
        F_out: 融合深度引导的特征 (B,out_channels,H,W)
    """
    def __init__(self, in_channels, out_channels, reduction=4):
        super(DGAM, self).__init__()

        # 下采样
        self.down_depth = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.down_feat  = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

        # Q/K/V 映射
        self.W_Q = nn.Conv2d(1, in_channels // reduction, kernel_size=1)       # query from depth
        self.W_K = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)  # key from feat
        self.W_V = nn.Conv2d(in_channels, in_channels, kernel_size=1)         # value from feat

        # 输出映射
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, D, F_in):
        B, C, H, W = F_in.shape

        # 下采样
        D_ = self.down_depth(D)  # (B,1,H/2,W/2)
        F_ = self.down_feat(F_in)  # (B,C,H/2,W/2)

        # Q/K/V
        Q_D = self.W_Q(D_)  # (B,Cq,h,w)
        K_F = self.W_K(F_)  # (B,Ck,h,w)
        V_F = self.W_V(F_)  # (B,Cv,h,w)

        Bq, Cq, Hq, Wq = Q_D.shape
        Bk, Ck, Hk, Wk = K_F.shape
        Bv, Cv, Hv, Wv = V_F.shape
        assert (Hq, Wq) == (Hk, Wk) == (Hv, Wv), "Q/K/V spatial mismatch!"

        N = Hq * Wq  # 动态获取 token 数

        Q = Q_D.view(B, Cq, N).permute(0, 2, 1).contiguous()  # (B,N,Cq)
        K = K_F.view(B, Ck, N).contiguous()  # (B,Ck,N)
        V = V_F.view(B, Cv, N).permute(0, 2, 1).contiguous()  # (B,N,Cv)

        # 注意力
        Att = torch.matmul(Q, K) / (Ck ** 0.5)   # (B,N,N)
        Att = F.softmax(Att, dim=-1)

        # 加权 V
        F_A = torch.matmul(Att, V)               # (B,N,Cv)
        F_A = F_A.permute(0, 2, 1).contiguous().view(B, Cv, H // 2, W // 2)

        # 输出
        F_A = self.conv_out(F_A)
        F_A_up = F.interpolate(F_A, size=(H, W), mode="bilinear", align_corners=False)

        F_out = F_A_up + F_in
        return F_out


if __name__ == "__main__":
    # 测试
    model = DGAM(in_channels=64, out_channels=64)
    B, C, H, W = 2, 64, 128, 128
    D = torch.randn(B, 1, H, W)
    F_in = torch.randn(B, C, H, W)
    out = model(D, F_in)
    print("输出形状:", out.shape)  # 期望 (2,64,128,128)
