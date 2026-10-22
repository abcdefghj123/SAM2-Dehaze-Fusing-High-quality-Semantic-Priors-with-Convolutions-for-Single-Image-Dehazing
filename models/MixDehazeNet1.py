# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============== 依赖于你工程内已有模块（路径保持与原工程一致） ===============
from .deconv import DEConv
from .scconv import ScConv
from .RDB import RDB
from .DepthwiseSeparableConv import Depthwise_separable_conv
from .SPT import SAMPrior, SPTUnit
from .DyConverse import DyConverse

# =========================== 基础网络组件（与你现有一致/兼容，并加入条件调制） ===========================

class MixStructureBlock(nn.Module):
    """
    扩展：use_cond=True 时，支持以 cond_global (B,C,1,1) 生成 γ,β，对两个子路径分别做 FiLM 调制。
    说明：为稳健，只用 cond_global；如需逐像素 SPADE，可把 cond_local 也转 γ/β 注入到 BN 前。
    """
    def __init__(self, dim, use_cond=False):
        super().__init__()
        self.use_cond = use_cond

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        # 原 1x1 conv 改为差分卷积 + ScConv
        self.conv1   = DEConv(dim)
        self.conv1_1 = ScConv(dim, 24)

        self.conv2   = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv3_19= nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_13= nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')

        # Simple Pixel Attention
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.mlp  = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )

        if self.use_cond:
            # 用全局先验 (B,C,1,1) 生成 per-channel γ,β
            self.cond_gamma1 = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=True), nn.Tanh())
            self.cond_beta1  = nn.Conv2d(dim, dim, 1, bias=True)
            self.cond_gamma2 = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=True), nn.Tanh())
            self.cond_beta2  = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x, cond_local=None, cond_global=None):
        # ----- path 1 -----
        identity = x
        x = self.norm1(x)
        if self.use_cond and (cond_global is not None):
            gamma = self.cond_gamma1(cond_global)  # B×C×1×1
            beta  = self.cond_beta1(cond_global)
            x = (1. + gamma) * x + beta

        x1 = self.conv1(x)
        x2 = self.conv1_1(x)
        x  = x1 + x2
        x  = self.conv2(x)
        x  = torch.cat([self.conv3_19(x), self.conv3_13(x), self.conv3_7(x)], dim=1)
        x  = self.mlp(x)
        x  = identity + x

        # ----- path 2 -----
        identity = x
        x = self.norm2(x)
        if self.use_cond and (cond_global is not None):
            gamma = self.cond_gamma2(cond_global)
            beta  = self.cond_beta2(cond_global)
            x = (1. + gamma) * x + beta

        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp2(x)
        x = identity + x
        return x


class BasicLayer(nn.Module):
    """支持条件传递的层级堆叠"""
    def __init__(self, dim, depth, use_cond=False):
        super().__init__()
        self.blocks = nn.ModuleList([MixStructureBlock(dim=dim, use_cond=use_cond) for _ in range(depth)])

    def forward(self, x, cond_local=None, cond_global=None):
        for blk in self.blocks:
            x = blk(x, cond_local=cond_local, cond_global=cond_global)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        if kernel_size is None:
            kernel_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
            padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect'
        )
    def forward(self, x): return self.proj(x)


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        if kernel_size is None:
            kernel_size = 1
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )
    def forward(self, x): return self.proj(x)


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super().__init__()
        self.height = height
        d = max(int(dim / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        x = torch.cat(in_feats, dim=1).view(B, self.height, C, H, W)
        feats_sum = torch.sum(x, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))
        out = torch.sum(x * attn, dim=1)
        return out

# =========================== 新增：先验编码器（供编码阶段各尺度使用） ===========================

class PriorEncoder(nn.Module):
    """
    将 (mask, depth) 编为与阶段通道一致的先验特征：
    - local: B×C×H×W（逐像素）
    - global: B×C×1×1（全局）
    深度做 min-max 归一化到 [0,1]，增强稳健性。
    """
    def __init__(self, c_out):
        super().__init__()
        self.mask_enc  = nn.Sequential(nn.Conv2d(1, c_out, 3, padding=1, bias=True), nn.ReLU(inplace=True))
        self.depth_enc = nn.Sequential(nn.Conv2d(1, c_out, 3, padding=1, bias=True), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.merge = nn.Conv2d(c_out, c_out, 1, bias=True)  # 用于 me+de 后再细调

    @torch.no_grad()
    def _norm_depth(self, d):
        dmin = d.amin(dim=(2,3), keepdim=True)
        dmax = d.amax(dim=(2,3), keepdim=True)
        return (d - dmin) / (dmax - dmin + 1e-6)

    def forward(self, mask, depth, size_hw):
        m = F.interpolate(mask,  size=size_hw, mode='bilinear', align_corners=False)
        d = F.interpolate(depth, size=size_hw, mode='bilinear', align_corners=False)
        d = self._norm_depth(d)
        me = self.mask_enc(m)          # B×C×H×W
        de = self.depth_enc(d)         # B×C×H×W
        feat = self.merge(me + de)     # B×C×H×W
        g = self.pool(feat)            # B×C×1×1
        return feat, g

# =========================== 新增：深度先验融合组件（AFEM-DepthGuided） ===========================

class DepthEncoder(nn.Module):
    """1ch 深度 -> C 通道，引导特征（解码分支用）"""
    def __init__(self, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
        )
    def forward(self, d): return self.net(d)


class DepthAwareCombiner(nn.Module):
    """由 (x_low, x_high, d_embed) 预测 gate + FiLM（解码分支用）"""
    def __init__(self, channels):
        super().__init__()
        self.fuse     = nn.Conv2d(3 * channels, channels, 1, bias=True)
        self.to_gate  = nn.Conv2d(channels, channels, 1, bias=True)
        self.to_gamma = nn.Conv2d(2 * channels, channels, 1, bias=True)
        self.to_beta  = nn.Conv2d(2 * channels, channels, 1, bias=True)

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
    """单尺度深度引导融合（解码分支）"""
    def __init__(self, in_ch_low, in_ch_high, out_ch):
        super().__init__()
        assert out_ch == in_ch_high
        self.proj_low = nn.Conv2d(in_ch_low, out_ch, 1, bias=False) if in_ch_low != out_ch else nn.Identity()
        self.de   = DepthEncoder(out_ch)
        self.comb = DepthAwareCombiner(out_ch)
        self.cbam = CBAM_Lite(out_ch, rr=4)
        self.tail = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
        )

    def forward(self, x_low, x_high, depth):
        H, W = x_high.shape[2:]
        x_low = F.interpolate(x_low, size=(H, W), mode='bilinear', align_corners=False)
        depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)

        # normalize depth to [0,1]
        d_min = depth.amin(dim=(2, 3), keepdim=True)
        d_max = depth.amax(dim=(2, 3), keepdim=True)
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
            nn.Conv2d(in_img_ch * 2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, img_depth, img_seg):
        a = self.gate(torch.cat([img_depth, img_seg], dim=1))
        return a * img_depth + (1. - a) * img_seg, a

# =========================== 主体：共享编码器 + 双并行解码器（编码阶段加入 FiLM 先验） ===========================

class MixDehazeNet(nn.Module):
    """
    共享编码器；两个并行解码器：
      - Seg 解码器：分割先验（沿用 SAMPrior/SPT、SKFusion 等）
      - Depth 解码器：深度先验（AFEM-DepthGuided）
    同时在 编码阶段 引入 FiLM 条件调制（方案二）。
    最终用 FinalFusionHead 做自适应融合。
    """
    def __init__(self, in_chans=3, out_chans=4,  # seg 解码器中间输出 K(1)+B(3)
                 num_classes=1, seg_dim=32, output_nc=3,
                 embed_dims=[24, 48, 96, 48, 24],
                 depths=[1, 1, 2, 1, 1]):
        super().__init__()
        self.patch_size = 4

        # ===== 编码器：打开 use_cond=True =====
        self.patch_embed = PatchEmbed(patch_size=1, in_chans=in_chans,
                                      embed_dim=embed_dims[0], kernel_size=3)
        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0], use_cond=True)
        self.patch_merge1 = PatchEmbed(patch_size=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1], kernel_size=3)
        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1], use_cond=True)
        self.patch_merge2 = PatchEmbed(patch_size=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2], kernel_size=3)
        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2], use_cond=True)

        # 先验编码器（与编码阶段通道一致）
        self.prior_enc_s1 = PriorEncoder(c_out=embed_dims[0])  # 24
        self.prior_enc_s2 = PriorEncoder(c_out=embed_dims[1])  # 48
        self.prior_enc_s3 = PriorEncoder(c_out=embed_dims[2])  # 96

        # ===== Seg 解码器 =====
        self.seg_up1 = nn.Sequential(                     # 96 -> 48
            DyConverse(in_channels=embed_dims[2], scale=2, style='lp'),
            nn.Conv2d(embed_dims[2], embed_dims[3], 1)
        )
        self.seg_fuse1 = SKFusion(embed_dims[3])
        self.seg_layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3])

        self.seg_up2 = nn.Sequential(                     # 48 -> 24
            DyConverse(in_channels=embed_dims[3], scale=2, style='lp'),
            nn.Conv2d(embed_dims[3], embed_dims[4], 1)
        )
        self.seg_fuse2 = SKFusion(embed_dims[4])
        self.seg_layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4])

        # 分割先验模块（与你原来一致）
        self.spt_SAMPrior  = SAMPrior(25, 24)
        self.spt_SAMPrior1 = SAMPrior(49, 48)
        self.spt_SPTUnit1  = SPTUnit(24, 24)
        self.spt_SPTUnit2  = SPTUnit(48, 48)

        # seg 重建头（K/B 分解 + coarse/seg/combine）
        self.seg_unembed = PatchUnEmbed(patch_size=1, out_chans=out_chans,
                                        embed_dim=embed_dims[4], kernel_size=3)
        self.seg_convs = nn.Sequential(RDB(num_classes), nn.Conv2d(num_classes, seg_dim, 1, bias=True))
        self.coarse_convs = nn.Sequential(Depthwise_separable_conv(output_nc, seg_dim))
        self.combine_convs = nn.Sequential(
            nn.Conv2d(seg_dim, seg_dim, 3, 1, 1, bias=True),
            nn.Conv2d(seg_dim, seg_dim, 3, 1, 1, bias=True),
            nn.Conv2d(seg_dim, output_nc, 3, 1, 1, bias=True),
            nn.Tanh()
        )

        # ===== Depth 解码器 =====
        self.dep_up1 = nn.Sequential(                     # 96 -> 48
            DyConverse(in_channels=embed_dims[2], scale=2, style='lp'),
            nn.Conv2d(embed_dims[2], embed_dims[3], 1)
        )
        self.dep_fuse1 = AFEM_DepthGuidedBlock(in_ch_low=embed_dims[1],  # skip2 (48)
                                               in_ch_high=embed_dims[3],
                                               out_ch=embed_dims[3])
        self.dep_layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3])

        self.dep_up2 = nn.Sequential(                     # 48 -> 24
            DyConverse(in_channels=embed_dims[3], scale=2, style='lp'),
            nn.Conv2d(embed_dims[3], embed_dims[4], 1)
        )
        self.dep_fuse2 = AFEM_DepthGuidedBlock(in_ch_low=embed_dims[0],  # skip1 (24)
                                               in_ch_high=embed_dims[4],
                                               out_ch=embed_dims[4])
        self.dep_layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4])

        self.dep_head = nn.Sequential(                    # 24 -> 3
            nn.Conv2d(embed_dims[4], embed_dims[4], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims[4], output_nc, 3, padding=1)
        )

        # ===== 最终自适应融合 =====
        self.final_fuse = FinalFusionHead(in_img_ch=output_nc)

    # ---------- utils ----------
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    # ---------- 编码（方案二：加入先验 FiLM） ----------
    def encode(self, x, mask=None, depth=None):
        # stage1 @ H,W, C=24
        x = self.patch_embed(x)            # B,24,H,W
        if (mask is not None) and (depth is not None):
            c1_local, c1_global = self.prior_enc_s1(mask, depth, size_hw=x.shape[-2:])
        else:
            c1_local = c1_global = None
        x = self.layer1(x, cond_local=c1_local, cond_global=c1_global)
        skip1 = x                          # B,24,H,W

        # stage2 @ H/2,W/2, C=48
        x = self.patch_merge1(x)           # B,48,H/2,W/2
        if (mask is not None) and (depth is not None):
            c2_local, c2_global = self.prior_enc_s2(mask, depth, size_hw=x.shape[-2:])
        else:
            c2_local = c2_global = None
        x = self.layer2(x, cond_local=c2_local, cond_global=c2_global)
        skip2 = x                          # B,48,H/2,W/2

        # stage3 @ H/4,W/4, C=96
        x = self.patch_merge2(x)           # B,96,H/4,W/4
        if (mask is not None) and (depth is not None):
            c3_local, c3_global = self.prior_enc_s3(mask, depth, size_hw=x.shape[-2:])
        else:
            c3_local = c3_global = None
        x = self.layer3(x, cond_local=c3_local, cond_global=c3_global)  # B,96,H/4,W/4
        return x, skip1, skip2

    # ---------- Seg 解码 ----------
    def decode_seg(self, x96, skip1, skip2, mask, x_in):
        # H/2
        x = self.seg_up1(x96)                              # 96->48
        x = self.seg_fuse1([x, self.skip2(skip2)]) + x
        m2 = F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=False)
        P1 = self.spt_SAMPrior1(x, m2)
        x  = x + self.spt_SPTUnit2(x, m2, P1)
        x  = self.seg_layer4(x)

        # H
        x = self.seg_up2(x)                                # 48->24
        x = self.seg_fuse2([x, self.skip1(skip1)]) + x
        m1 = F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=False)
        P  = self.spt_SAMPrior(x, m1)
        x  = x + self.spt_SPTUnit1(x, m1, P)
        x  = self.seg_layer5(x)

        feat = self.seg_unembed(x)                         # out_chans=4 => K(1)+B(3)
        K, B = torch.split(feat, (1, 3), dim=1)
        img = K * x_in - B + x_in                          # 粗恢复图
        return img

    # ---------- Depth 解码 ----------
    def decode_depth(self, x96, skip1, skip2, depth):
        # H/2
        x = self.dep_up1(x96)                              # 96->48
        d1= F.interpolate(depth, size=x.size()[-2:], mode='bilinear', align_corners=False)
        x = self.dep_fuse1(x_low=self.skip2(skip2), x_high=x, depth=d1)
        x = self.dep_layer4(x)

        # H
        x = self.dep_up2(x)                                # 48->24
        d0= F.interpolate(depth, size=x.size()[-2:], mode='bilinear', align_corners=False)
        x = self.dep_fuse2(x_low=self.skip1(skip1), x_high=x, depth=d0)
        x = self.dep_layer5(x)

        img = self.dep_head(x)                             # 24->3
        return img

    # ---------- 前向 ----------
    def forward(self, x, mask, depth):
        H, W = x.shape[2:]
        xin = self.check_image_size(x)
        msk = F.interpolate(mask,  size=xin.size()[-2:], mode='bilinear', align_corners=False)
        dpt = F.interpolate(depth, size=xin.size()[-2:], mode='bilinear', align_corners=False)

        # 共享编码（方案二：编码阶段就注入先验）
        x96, skip1, skip2 = self.encode(xin, msk, dpt)
        # Seg 分支（分割先验）
        img_seg = self.decode_seg(x96, skip1, skip2, msk, xin)
        img_seg = img_seg[:, :, :H, :W]                   # 裁回原图大小
        img_seg = self.coarse_convs(img_seg) + self.seg_convs(mask)
        img_seg = self.combine_convs(img_seg)             # (B,3,H,W)

        # Depth 分支（深度先验）
        img_dep = self.decode_depth(x96, skip1, skip2, dpt)
        img_dep = img_dep[:, :, :H, :W]

        # 最终融合
        img, alpha = self.final_fuse(img_dep, img_seg)    # (B,3,H,W), (B,1,H,W)
        return img, img_seg, img_dep, alpha


# =========================== 便捷构造函数（不同规模） ===========================

def MixDehazeNet_t():
    return MixDehazeNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[1, 1, 2, 1, 1])

def MixDehazeNet_s():
    return MixDehazeNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[2, 2, 4, 2, 2])

def MixDehazeNet_b():
    return MixDehazeNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[4, 4, 8, 4, 4])

def MixDehazeNet_l():
    return MixDehazeNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[8, 8, 16, 8, 8])
