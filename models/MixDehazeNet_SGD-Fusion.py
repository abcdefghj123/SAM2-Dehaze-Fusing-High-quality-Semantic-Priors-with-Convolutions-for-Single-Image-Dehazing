import torch
import torch.nn as nn
import torch.nn.functional as F
from .deconv import DEConv
from .scconv import ScConv
from .RDB import RDB
from .DepthwiseSeparableConv import Depthwise_separable_conv
from .SPF import SPFUnit
from .SPT import SAMPrior, SPTUnit
from .DyConverse import DyConverse
from .SGDFusion import SGDFusion


# ==========================================================
# 基本模块
# ==========================================================

class MixStructureBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        # 多分支卷积
        self.conv1 = DEConv(dim)
        self.conv1_1 = ScConv(dim, 24)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')

        # Pixel Attention
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
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # MLP 融合
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)

        x1 = self.conv1(x)
        x2 = self.conv1_1(x)
        x = x1 + x2

        x = self.conv2(x)
        x = torch.cat([self.conv3_19(x), self.conv3_13(x), self.conv3_7(x)], dim=1)
        x = self.mlp(x)
        x = identity + x

        identity = x
        x = self.norm2(x)
        x = torch.cat([
            self.Wv(x) * self.Wg(x),
            self.ca(x) * x,
            self.pa(x) * x
        ], dim=1)
        x = self.mlp2(x)
        x = identity + x
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.blocks = nn.ModuleList([MixStructureBlock(dim=dim) for _ in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        if kernel_size is None:
            kernel_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                              stride=patch_size, padding=(kernel_size - patch_size + 1) // 2,
                              padding_mode='reflect')

    def forward(self, x):
        return self.proj(x)


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        if kernel_size is None:
            kernel_size = 1
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2,
                      kernel_size=kernel_size, padding=kernel_size // 2,
                      padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        return self.proj(x)


# ==========================================================
# 主网络
# ==========================================================

class MixDehazeNet_SGDFusion(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, num_classes=1,
                 seg_dim=32, output_nc=3,
                 embed_dims=[24, 48, 96, 48, 24],
                 depths=[1, 1, 2, 1, 1]):
        super().__init__()
        self.patch_size = 4

        # Patch Embedding
        self.patch_embed = PatchEmbed(1, in_chans, embed_dims[0], kernel_size=3)

        # Encoder
        self.layer1 = BasicLayer(embed_dims[0], depths[0])
        self.sgdfusion1 = SGDFusion(embed_dims[0], window_size=8)
        self.patch_merge1 = PatchEmbed(2, embed_dims[0], embed_dims[1], kernel_size=3)
        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(embed_dims[1], depths[1])
        self.sgdfusion2 = SGDFusion(embed_dims[1], window_size=8)
        self.patch_merge2 = PatchEmbed(2, embed_dims[1], embed_dims[2], kernel_size=3)
        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(embed_dims[2], depths[2])
        self.sgdfusion3 = SGDFusion(embed_dims[2], window_size=8)

        # Decoder
        self.patch_split1 = nn.Sequential(
            DyConverse(embed_dims[2], scale=2, style='lp'),
            nn.Conv2d(embed_dims[2], embed_dims[3], 1)
        )
        self.sgdfusion4 = SGDFusion(embed_dims[3], window_size=8)
        self.layer4 = BasicLayer(embed_dims[3], depths[3])

        self.patch_split2 = nn.Sequential(
            DyConverse(embed_dims[3], scale=2, style='lp'),
            nn.Conv2d(embed_dims[3], embed_dims[4], 1)
        )
        self.sgdfusion5 = SGDFusion(embed_dims[4], window_size=8)
        self.layer5 = BasicLayer(embed_dims[4], depths[4])

        # 输出
        self.patch_unembed = PatchUnEmbed(1, out_chans, embed_dims[4], kernel_size=3)

        # 语义模块
        self.spt_SAMPrior = SAMPrior(25, 24)
        self.spt_SAMPrior1 = SAMPrior(49, 48)
        self.spt_SPTUnit1 = SPTUnit(24, 24)
        self.spt_SPTUnit2 = SPTUnit(48, 48)

        self.seg_convs = nn.Sequential(RDB(num_classes), nn.Conv2d(num_classes, seg_dim, 1, bias=True))
        self.coarse_convs = nn.Sequential(Depthwise_separable_conv(output_nc, seg_dim))
        self.combine_convs = nn.Sequential(
            nn.Conv2d(seg_dim, seg_dim, 3, 1, 1, bias=True),
            nn.Conv2d(seg_dim, seg_dim, 3, 1, 1, bias=True),
            nn.Conv2d(seg_dim, output_nc, 3, 1, 1, bias=True),
            nn.Tanh()
        )

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    def forward_features(self, x, mask, depth):
        # Stage1
        x = self.layer1(self.patch_embed(x))
        depth_resized = F.interpolate(depth, size=x.shape[-2:], mode='bilinear', align_corners=False)
        mask_resized = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.sgdfusion1(x, depth_resized, mask_resized)
        skip1 = x
        P = self.spt_SAMPrior(x, mask_resized)
        x = x + self.spt_SPTUnit1(x, mask_resized, P)
        x = self.patch_merge1(x)

        # Stage2
        x = self.layer2(x)
        depth_resized = F.interpolate(depth, size=x.shape[-2:], mode='bilinear', align_corners=False)
        mask_resized = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.sgdfusion2(x, depth_resized, mask_resized)
        skip2 = x
        P1 = self.spt_SAMPrior1(x, mask_resized)
        x = x + self.spt_SPTUnit2(x, mask_resized, P1)
        x = self.patch_merge2(x)

        # Stage3
        x = self.layer3(x)
        depth_resized = F.interpolate(depth, size=x.shape[-2:], mode='bilinear', align_corners=False)
        mask_resized = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.sgdfusion3(x, depth_resized, mask_resized)

        # Stage4
        x = self.patch_split1(x)
        x = self.layer4(x)
        depth_resized = F.interpolate(depth, size=x.shape[-2:], mode='bilinear', align_corners=False)
        mask_resized = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.sgdfusion4(x, depth_resized, mask_resized)

        # Stage5
        x = self.patch_split2(x)
        x = self.layer5(x)
        depth_resized = F.interpolate(depth, size=x.shape[-2:], mode='bilinear', align_corners=False)
        mask_resized = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.sgdfusion5(x, depth_resized, mask_resized)

        return self.patch_unembed(x)

    def forward(self, x, mask, depth):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        feat = self.forward_features(x, mask, depth)

        K, B = torch.split(feat, (1, 3), dim=1)
        x = K * x - B + x
        x = x[:, :, :H, :W]

        x = self.coarse_convs(x) + self.seg_convs(mask)
        return self.combine_convs(x)


# ==========================================================
# 四种配置
# ==========================================================

def MixDehazeNet_t():
    return MixDehazeNet_SGDFusion(embed_dims=[24, 48, 96, 48, 24], depths=[1, 1, 2, 1, 1])

def MixDehazeNet_s():
    return MixDehazeNet_SGDFusion(embed_dims=[24, 48, 96, 48, 24], depths=[2, 2, 4, 2, 2])

def MixDehazeNet_b():
    return MixDehazeNet_SGDFusion(embed_dims=[24, 48, 96, 48, 24], depths=[4, 4, 8, 4, 4])

def MixDehazeNet_l():
    return MixDehazeNet_SGDFusion(embed_dims=[24, 48, 96, 48, 24], depths=[8, 8, 16, 8, 8])
