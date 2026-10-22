import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis
from torchinfo import summary
import torch.utils.benchmark as benchmark
from .deconv import DEConv
from .scconv import ScConv
from .RDB import RDB
from .DepthwiseSeparableConv import Depthwise_separable_conv
from .SPF import SPFUnit
from .SPT import SAMPrior
from .SPT import SPTUnit
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from .deconv import DEConv
from .scconv import ScConv

class MixStructureBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        #原始
        # self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        #加入差分卷积
        self.conv1 = DEConv(dim)
        self.conv1_1 = ScConv(dim,24)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')
        # Simple Pixel Attention
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
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
            # nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x1 = self.conv1(x)
        x2 = self.conv1_1(x)
        x = x1 + x2

        # x = torch.cat([self.conv1(x),self.conv1_1(x)],dim=1)
        x = self.conv2(x)
        x = torch.cat([self.conv3_19(x), self.conv3_13(x), self.conv3_7(x)], dim=1)
        x = self.mlp(x)
        x = identity + x

        identity = x
        x = self.norm2(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp2(x)
        x = identity + x
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.dim = dim
        self.depth = depth
        # build blocks
        self.blocks = nn.ModuleList(
            [MixStructureBlock(dim=dim) for i in range(depth)])
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

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

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class MixDehazeNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, num_classes=1, seg_dim=32,output_nc=3,
                 embed_dims=[24, 48, 96, 48, 24],
                 depths=[1, 1, 2, 1, 1]):
        super(MixDehazeNet, self).__init__()
        # setting
        self.patch_size = 4

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], kernel_size=3)

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1])

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], kernel_size=3)

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2])

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4])

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

        self.spt_SAMPrior = SAMPrior(25,24)
        self.spt_SAMPrior1 = SAMPrior(49, 48)

        self.spt_SPTUnit1 = SPTUnit(24, 24)
        self.spt_SPTUnit2 = SPTUnit(48, 48)

        # semantic fused
        self.seg_convs = nn.Sequential(
            RDB(num_classes),
            nn.Conv2d(num_classes, seg_dim, 1, bias=True)
        )

        self.coarse_convs = nn.Sequential(
            Depthwise_separable_conv(output_nc, seg_dim)
        )

        self.combine_convs = nn.Sequential(
            nn.Conv2d(seg_dim, seg_dim, 3, 1, 1, bias=True),
            nn.Conv2d(seg_dim, seg_dim, 3, 1, 1, bias=True),
            nn.Conv2d(seg_dim, output_nc, 3, 1, 1, bias=True),
            nn.Tanh()
        )

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    def forward_features(self, x,mask):
        x = self.patch_embed(x)
        x = self.layer1(x)
        # print("x1",x.shape)  #(6,24,256,256)
        skip1 = x
        P = self.spt_SAMPrior(x, mask)
        x1 = self.spt_SPTUnit1(x, mask, P)
        x = x + x1
        x = self.patch_merge1(x)
        x = self.layer2(x)
        # print("x2", x.shape) #(6,48,128,128)
        skip2 = x

        mask = F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=False)
        P1 = self.spt_SAMPrior1(x, mask)
        x2 = self.spt_SPTUnit2(x, mask, P1)
        x = x + x2
        # print("x",x.shape)
        x = self.patch_merge2(x)
        x = self.layer3(x)#(6,96,64,64)
        x = self.patch_split1(x)#(6,48,128,128)
        # print("x333", x.shape)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        P1 = self.spt_SAMPrior1(x, mask)
        x2 = self.spt_SPTUnit2(x, mask, P1)
        x = x + x2
        x = self.layer4(x)
        # print("x4", x.shape)   #(6,48,128,128)
        x = self.patch_split2(x)#(6,24,256,256)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        mask = F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=False)
        P = self.spt_SAMPrior(x, mask)
        x1 = self.spt_SPTUnit1(x, mask, P)
        x = x + x1
        x = self.layer5(x)
        # print("x5", x.shape)   #(6,24,256,256)
        x = self.patch_unembed(x)
        # print("x",x.shape)
        return x
        # 消G1+G2+G3+G4
    # def forward_features(self, x, mask):
    #     # print("x",x.shape)
    #     # print("mask",mask.shape)
    #     x = self.patch_embed(x)
    #     x = self.layer1(x)
    #     # print("x1",x.shape)  #(6,24,256,256)
    #     skip1 = x
    #     P = self.spt_SAMPrior(x, mask)
    #     x1 = self.spt_SPTUnit1(x, mask, P)
    #     x = x + x1
    #     x = self.patch_merge1(x)
    #     x = self.layer2(x)
    #     # print("x2", x.shape) #(6,48,128,128)
    #     skip2 = x
    #
    #     mask = F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=False)
    #     print("x",x.shape)
    #     print("mask",mask.shape)
    #     P1 = self.spt_SAMPrior1(x, mask)
    #     x2 = self.spt_SPTUnit2(x, mask, P1)
    #     x = x + x2
    #     # print("x",x.shape)
    #     x = self.patch_merge2(x)
    #     x = self.layer3(x)  # (6,96,64,64)
    #     x = self.patch_split1(x)  # (6,48,128,128)
    #     # print("x333", x.shape)
    #
    #     x = self.fusion1([x, self.skip2(skip2)]) + x
    #     P1 = self.spt_SAMPrior1(x, mask)
    #     x2 = self.spt_SPTUnit2(x, mask, P1)
    #     x = x + x2
    #     x = self.layer4(x)
    #     # print("x4", x.shape)   #(6,48,128,128)
    #     x = self.patch_split2(x)  # (6,24,256,256)
    #
    #     x = self.fusion2([x, self.skip1(skip1)]) + x
    #     mask = F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=False)
    #     P = self.spt_SAMPrior(x, mask)
    #     x1 = self.spt_SPTUnit1(x, mask, P)
    #     x = x + x1
    #     x = self.layer5(x)
    #     # print("x5", x.shape)   #(6,24,256,256)
    #     x = self.patch_unembed(x)
    #     # print("x",x.shape)
    #     return x
    #
    #     # 消G1+G2+G3
    # def forward_features(self, x, mask):
    #     x = self.patch_embed(x)
    #     x = self.layer1(x)
    #     # print("x1",x.shape)  #(6,24,256,256)
    #     skip1 = x
    #     P = self.spt_SAMPrior(x, mask)
    #     x1 = self.spt_SPTUnit1(x, mask, P)
    #     x = x + x1
    #     x = self.patch_merge1(x)
    #     x = self.layer2(x)
    #     # print("x2", x.shape) #(6,48,128,128)
    #     skip2 = x
    #
    #     mask = F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=False)
    #     P1 = self.spt_SAMPrior1(x, mask)
    #     x2 = self.spt_SPTUnit2(x, mask, P1)
    #     x = x + x2
    #     # print("x",x.shape)
    #     x = self.patch_merge2(x)
    #     x = self.layer3(x)  # (6,96,64,64)
    #     x = self.patch_split1(x)  # (6,48,128,128)
    #     # print("x333", x.shape)
    #
    #     x = self.fusion1([x, self.skip2(skip2)]) + x
    #     P1 = self.spt_SAMPrior1(x, mask)
    #     x2 = self.spt_SPTUnit2(x, mask, P1)
    #     x = x + x2
    #     x = self.layer4(x)
    #     # print("x4", x.shape)   #(6,48,128,128)
    #     x = self.patch_split2(x)  # (6,24,256,256)
    #
    #     x = self.fusion2([x, self.skip1(skip1)]) + x
    #     # mask = F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=False)
    #     # P = self.spt_SAMPrior(x, mask)
    #     # x1 = self.spt_SPTUnit1(x, mask, P)
    #     # x = x + x1
    #     x = self.layer5(x)
    #     # print("x5", x.shape)   #(6,24,256,256)
    #     x = self.patch_unembed(x)
    #     # print("x",x.shape)
    #     return x
    # # 消G1+G2
    # def forward_features(self, x, mask):
    #     x = self.patch_embed(x)
    #     x = self.layer1(x)
    #     # print("x1",x.shape)  #(6,24,256,256)
    #     skip1 = x
    #     P = self.spt_SAMPrior(x, mask)
    #     x1 = self.spt_SPTUnit1(x, mask, P)
    #     x = x + x1
    #     x = self.patch_merge1(x)
    #     x = self.layer2(x)
    #     # print("x2", x.shape) #(6,48,128,128)
    #     skip2 = x
    #
    #     mask = F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=False)
    #     P1 = self.spt_SAMPrior1(x, mask)
    #     x2 = self.spt_SPTUnit2(x, mask, P1)
    #     x = x + x2
    #     # print("x",x.shape)
    #     x = self.patch_merge2(x)
    #     x = self.layer3(x)  # (6,96,64,64)
    #     x = self.patch_split1(x)  # (6,48,128,128)
    #     # print("x333", x.shape)
    #
    #     x = self.fusion1([x, self.skip2(skip2)]) + x
    #     # P1 = self.spt_SAMPrior1(x, mask)
    #     # x2 = self.spt_SPTUnit2(x, mask, P1)
    #     # x = x + x2
    #     x = self.layer4(x)
    #     # print("x4", x.shape)   #(6,48,128,128)
    #     x = self.patch_split2(x)  # (6,24,256,256)
    #
    #     x = self.fusion2([x, self.skip1(skip1)]) + x
    #     # mask = F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=False)
    #     # P = self.spt_SAMPrior(x, mask)
    #     # x1 = self.spt_SPTUnit1(x, mask, P)
    #     # x = x + x1
    #     x = self.layer5(x)
    #     # print("x5", x.shape)   #(6,24,256,256)
    #     x = self.patch_unembed(x)
    #     # print("x",x.shape)
    #     return x
    # #消G1
    # def forward_features(self, x,mask):
    #     x = self.patch_embed(x)
    #     x = self.layer1(x)
    #     # print("x1",x.shape)  #(6,24,256,256)
    #     skip1 = x
    #     P = self.spt_SAMPrior(x, mask)
    #     x1 = self.spt_SPTUnit1(x, mask, P)
    #     x = x + x1
    #     x = self.patch_merge1(x)
    #     x = self.layer2(x)
    #     # print("x2", x.shape) #(6,48,128,128)
    #     skip2 = x
    #
    #     # mask = F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=False)
    #     # P1 = self.spt_SAMPrior1(x, mask)
    #     # x2 = self.spt_SPTUnit2(x, mask, P1)
    #     # x = x + x2
    #     # print("x",x.shape)
    #     x = self.patch_merge2(x)
    #     x = self.layer3(x)#(6,96,64,64)
    #     x = self.patch_split1(x)#(6,48,128,128)
    #     # print("x333", x.shape)
    #
    #     x = self.fusion1([x, self.skip2(skip2)]) + x
    #     # P1 = self.spt_SAMPrior1(x, mask)
    #     # x2 = self.spt_SPTUnit2(x, mask, P1)
    #     # x = x + x2
    #     x = self.layer4(x)
    #     # print("x4", x.shape)   #(6,48,128,128)
    #     x = self.patch_split2(x)#(6,24,256,256)
    #
    #     x = self.fusion2([x, self.skip1(skip1)]) + x
    #     # mask = F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=False)
    #     # P = self.spt_SAMPrior(x, mask)
    #     # x1 = self.spt_SPTUnit1(x, mask, P)
    #     # x = x + x1
    #     x = self.layer5(x)
    #     # print("x5", x.shape)   #(6,24,256,256)
    #     x = self.patch_unembed(x)
    #     # print("x",x.shape)
    #     return x
    #（消融实验3）
    # def forward_features(self, x,mask):
    #     x = self.patch_embed(x)
    #     x = self.layer1(x)
    #     # print("x1",x.shape)  #(6,24,256,256)
    #     skip1 = x
    #     P = self.spt_SAMPrior(x, mask)
    #     # x1 = self.spt_SPTUnit1(x, mask, P)
    #     x = x + P
    #     x = self.patch_merge1(x)
    #     x = self.layer2(x)
    #     # print("x2", x.shape) #(6,48,128,128)
    #     skip2 = x
    #
    #     mask = F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=False)
    #     P1 = self.spt_SAMPrior1(x, mask)
    #     # x2 = self.spt_SPTUnit2(x, mask, P1)
    #     x = x + P1
    #     # print("x",x.shape)
    #     x = self.patch_merge2(x)
    #     x = self.layer3(x)#(6,96,64,64)
    #     x = self.patch_split1(x)#(6,48,128,128)
    #     # print("x333", x.shape)
    #
    #     x = self.fusion1([x, self.skip2(skip2)]) + x
    #     P1 = self.spt_SAMPrior1(x, mask)
    #     # x2 = self.spt_SPTUnit2(x, mask, P1)
    #     x = x + P1
    #     x = self.layer4(x)
    #     # print("x4", x.shape)   #(6,48,128,128)
    #     x = self.patch_split2(x)#(6,24,256,256)
    #
    #     x = self.fusion2([x, self.skip1(skip1)]) + x
    #     mask = F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=False)
    #     P = self.spt_SAMPrior(x, mask)
    #     # x1 = self.spt_SPTUnit1(x, mask, P)
    #     x = x + P
    #     x = self.layer5(x)
    #     # print("x5", x.shape)   #(6,24,256,256)
    #     x = self.patch_unembed(x)
    #     # print("x",x.shape)
    #     return x

    # #（消融实验2）
    # def forward_features(self, x, mask):
    #     x = self.patch_embed(x)
    #     x = self.layer1(x)
    #     # print("x1",x.shape)  #(6,24,256,256)
    #     skip1 = x
    #     P = self.spt_SAMPrior(x, mask)
    #     x1 = self.spt_SPTUnit1(x, mask, P)
    #     x = x + x1
    #     x = self.patch_merge1(x)
    #     x = self.layer2(x)
    #     # print("x2", x.shape) #(6,48,128,128)
    #     skip2 = x
    #
    #     mask = F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=False)
    #     P1 = self.spt_SAMPrior1(x, mask)
    #     x2 = self.spt_SPTUnit2(x, mask, P1)
    #     x = x + x2
    #     # print("x",x.shape)
    #     x = self.patch_merge2(x)
    #     x = self.layer3(x)  # (6,96,64,64)
    #     x = self.patch_split1(x)  # (6,48,128,128)
    #     # print("x333", x.shape)
    #
    #     x = self.fusion1([x, self.skip2(skip2)]) + x
    #     P1 = self.spt_SAMPrior1(x, mask)
    #     x2 = self.spt_SPTUnit2(x, mask, P1)
    #     x = x + x2
    #     x = self.layer4(x)
    #     # print("x4", x.shape)   #(6,48,128,128)
    #     x = self.patch_split2(x)  # (6,24,256,256)
    #
    #     x = self.fusion2([x, self.skip1(skip1)]) + x
    #     mask = F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=False)
    #     P = self.spt_SAMPrior(x, mask)
    #     x1 = self.spt_SPTUnit1(x, mask, P)
    #     x = x + x1
    #     x = self.layer5(x)
    #     # print("x5", x.shape)   #(6,24,256,256)
    #     x = self.patch_unembed(x)
    #     # print("x",x.shape)
    #     return x

    #（消融实验1）
    # def forward_features(self, x, mask):
    #     x = self.patch_embed(x)
    #     x = self.layer1(x)
    #     skip1 = x
    #
    #     # 融合 mask（替代 SPT）
    #     mask1 = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
    #     mask1 = nn.Conv2d(mask.shape[1], x.shape[1], 1).to(x.device)(mask1)
    #     x = x + mask1
    #
    #     x = self.patch_merge1(x)
    #     x = self.layer2(x)
    #     skip2 = x
    #
    #     mask2 = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
    #     mask2 = nn.Conv2d(mask.shape[1], x.shape[1], 1).to(x.device)(mask2)
    #     x = x + mask2
    #
    #     x = self.patch_merge2(x)
    #     x = self.layer3(x)
    #     x = self.patch_split1(x)
    #
    #     x = self.fusion1([x, self.skip2(skip2)]) + x
    #
    #     mask3 = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
    #     mask3 = nn.Conv2d(mask.shape[1], x.shape[1], 1).to(x.device)(mask3)
    #     x = x + mask3
    #
    #     x = self.layer4(x)
    #     x = self.patch_split2(x)
    #
    #     x = self.fusion2([x, self.skip1(skip1)]) + x
    #
    #     mask4 = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
    #     mask4 = nn.Conv2d(mask.shape[1], x.shape[1], 1).to(x.device)(mask4)
    #     x = x + mask4
    #
    #     x = self.layer5(x)
    #     x = self.patch_unembed(x)
    #     return x

    def forward(self, x, mask):
        # print("mask",mask.shape)
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        # print("mask", mask.shape)
        # print("x", x.shape)
        #fused_feature = torch.cat([x, mask], dim=1)
        # print("fused_feature", fused_feature.shape)
        feat = self.forward_features(x,mask)
        # print("feat", feat.shape)
        # 2022/11/26
        K, B = torch.split(feat, (1, 3), dim=1)
        x = K * x - B + x
        x = x[:, :, :H, :W]
        #print("X",x.shape)
        x = self.coarse_convs(x) + self.seg_convs(mask)
        x = self.combine_convs(x)
        return x

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