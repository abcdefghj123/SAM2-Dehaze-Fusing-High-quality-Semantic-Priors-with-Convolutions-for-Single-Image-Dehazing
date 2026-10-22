import torch
import torch.nn as nn
import torch.nn.functional as F


class SGDFusion(nn.Module):
    """
    Semantic-Guided Depth Fusion (轻量版, Window Attention)
    输入: feat (特征图), depth (深度图), seg (语义图)
    - 避免全局 N×N 注意力爆显存
    - 在小窗口内做注意力, O(B * N * window^2)
    """
    def __init__(self, dim, window_size=8, reduction=4):
        super(SGDFusion, self).__init__()
        self.dim = dim
        self.window_size = window_size

        # 映射到 Query/Key/Value
        self.query = nn.Conv2d(dim, dim // reduction, 1)
        self.key   = nn.Conv2d(dim, dim // reduction, 1)
        self.value = nn.Conv2d(dim, dim, 1)

        # 深度与语义嵌入
        self.d_emb = nn.Conv2d(1, dim // reduction, 1)
        self.s_emb = nn.Conv2d(1, dim // reduction, 1)

        # 输出投影
        self.proj = nn.Conv2d(dim, dim, 1)

    def window_partition(self, x, ws):
        """把特征切成小窗口"""
        B, N, C = x.shape
        H = W = int(N ** 0.5)  # 假设输入是方形
        x = x.view(B, H, W, C)
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(-1, ws * ws, C)  # (B*nW, Nw, C)

    def window_reverse(self, x, ws, H, W):
        """把小窗口合并回原图"""
        Bn, Nw, C = x.shape
        B = Bn // ((H // ws) * (W // ws))
        x = x.view(B, H // ws, W // ws, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B, H, W, C)

    def forward(self, feat, depth, seg):
        B, C, H, W = feat.shape
        ws = self.window_size

        # ✅ pad 到能被 window_size 整除
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        feat = F.pad(feat, (0, pad_w, 0, pad_h), mode="reflect")
        depth = F.pad(depth, (0, pad_w, 0, pad_h), mode="reflect")
        seg = F.pad(seg, (0, pad_w, 0, pad_h), mode="reflect")
        Hp, Wp = feat.shape[2], feat.shape[3]

        # 对齐深度 & 语义
        depth_resized = F.interpolate(depth, size=(Hp, Wp), mode='bilinear', align_corners=False)
        seg_resized = F.interpolate(seg, size=(Hp, Wp), mode='bilinear', align_corners=False)
        D_emb = self.d_emb(depth_resized)
        S_emb = self.s_emb(seg_resized)

        # Q, K, V
        Q = self.query(feat) + D_emb + S_emb
        K = self.key(feat) + D_emb + S_emb
        V = self.value(feat)

        # 转成 (B, N, C)
        Q = Q.permute(0, 2, 3, 1).contiguous().view(B, -1, Q.shape[1])
        K = K.permute(0, 2, 3, 1).contiguous().view(B, -1, K.shape[1])
        V = V.permute(0, 2, 3, 1).contiguous().view(B, -1, V.shape[1])

        # 窗口划分
        Qw = self.window_partition(Q, ws, Hp, Wp)
        Kw = self.window_partition(K, ws, Hp, Wp)
        Vw = self.window_partition(V, ws, Hp, Wp)

        # 注意力
        attn = torch.bmm(Qw, Kw.transpose(1, 2)) / (Kw.size(-1) ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, Vw)

        # 恢复窗口
        out = self.window_reverse(out, ws, Hp, Wp)
        out = out.permute(0, 3, 1, 2).contiguous()

        # ✅ 去掉 padding，恢复原始大小
        out = out[:, :, :H, :W]

        return self.proj(out) + feat[:, :, :H, :W]
