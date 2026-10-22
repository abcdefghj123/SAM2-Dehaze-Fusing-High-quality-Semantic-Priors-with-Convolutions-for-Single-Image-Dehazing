import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as Fnn   # 改个别名，避免和输入变量冲突

import torch
import torch.nn as nn
import torch.nn.functional as Fnn

class DGAM(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(DGAM, self).__init__()

        self.down_depth = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.down_feat  = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

        self.W_Q = nn.Conv2d(1, in_channels // reduction, kernel_size=1)
        self.W_K = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.W_V = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, D, F, return_att=False):
        B, C, H, W = F.shape

        # 下采样
        D_ = self.down_depth(D)
        F_ = self.down_feat(F)

        # Q K V
        Q_D = self.W_Q(D_)
        K_F = self.W_K(F_)
        V_F = self.W_V(F_)

        N = (H // 2) * (W // 2)
        Q = Q_D.view(B, -1, N).permute(0, 2, 1)   # (B,N,Cq)
        K = K_F.view(B, -1, N)                   # (B,Cq,N)
        V = V_F.view(B, -1, N).permute(0, 2, 1)  # (B,N,C)

        Att = torch.matmul(Q, K) / (K.size(1) ** 0.5)
        Att = Fnn.softmax(Att, dim=-1)  # (B,N,N)

        # 特征加权
        F_A = torch.matmul(Att, V)  # (B,N,C)
        F_A = F_A.permute(0, 2, 1).view(B, -1, H // 2, W // 2)

        F_A = self.conv_out(F_A)
        F_A_up = nn.functional.interpolate(F_A, size=(H, W), mode="bilinear", align_corners=False)

        F_out = F_A_up + F

        if return_att:
            return F_out, Att
        return F_out


if __name__ == '__main__':
    B, C, H, W = 1, 64, 128, 128
    D = torch.randn(B, 1, H, W)  # 深度图
    F = torch.randn(B, C, H, W)  # 特征图

    model = DGAM(in_channels=64, out_channels=64)
    out, att = model(D, F, return_att=True)

    print("输出特征 shape:", out.shape)
    print("注意力矩阵 shape:", att.shape)  # (B, N, N) 其中 N=(H/2)*(W/2)

    # ---------- 可视化 ----------
    # 随便取一行作为注意力分布，reshape 成 2D
    N = (H // 2) * (W // 2)
    att_map = att[0, 100, :].reshape(H // 2, W // 2).detach().cpu().numpy()

    plt.imshow(att_map, cmap="jet")
    plt.colorbar()
    plt.title("Depth-Guided Attention (query=100)")
    plt.show()