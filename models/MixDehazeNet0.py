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
from .DyConverse import DyConverse
from .DGAM import DGAM

class MDPAN(nn.Module):
    """
    Meta Depth Prompt Attention Network
    输入：C_image = cat(D_real, D_dcp)  shape [B, 2, H, W]
    输出：3 张 meta attention maps（AMDP1~3），用于在 layer1~3 注入
    """
    def __init__(self, in_channels=2, num_prompts=3, mid=32):
        super().__init__()
        self.num_prompts = num_prompts
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, num_prompts, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, c_image):  # [B,2,H,W]
        attn = self.net(c_image)  # [B,3,H,W]
        # 返回 [AMDP1, AMDP2, AMDP3]，每个都是 [B,1,H,W]
        return torch.chunk(attn, self.num_prompts, dim=1)

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
    def __init__(self, in_chans=8, out_chans=4, num_classes=1, seg_dim=32, output_nc=3,
                 embed_dims=[24, 48, 96, 48, 24],
                 depths=[1, 1, 2, 1, 1]):
        super(MixDehazeNet, self).__init__()
        self.patch_size = 4

        # 注意：in_chans 改成 8（I_real(3) + I_dcp(3) + D_real(1) + D_dcp(1)）
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
        # 原来的上采样
        # self.patch_split1 = PatchUnEmbed(
        #     patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])
        #
        # assert embed_dims[1] == embed_dims[3]
        # self.fusion1 = SKFusion(embed_dims[3])
        #
        # self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3])
        #
        # self.patch_split2 = PatchUnEmbed(
        #     patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])
        #
        # assert embed_dims[0] == embed_dims[4]
        # self.fusion2 = SKFusion(embed_dims[4])
        #
        # self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4])

        # 动态反卷积上采样
        # 原有的动态上采样+融合保持
        self.patch_split1 = nn.Sequential(
            DyConverse(in_channels=embed_dims[2], scale=2, style='lp'),
            nn.Conv2d(embed_dims[2], embed_dims[3], 1)
        )
        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])
        self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3])

        self.patch_split2 = nn.Sequential(
            DyConverse(in_channels=embed_dims[3], scale=2, style='lp'),
            nn.Conv2d(embed_dims[3], embed_dims[4], 1)
        )
        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])
        self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4])

        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

        # MDPAN（只用图像级深度提示 C_image = cat(D_real, D_dcp)）
        self.mdpan = MDPAN(in_channels=2, num_prompts=3)

        # ——以下保持你的分割/融合头不变——
        self.spt_SAMPrior = SAMPrior(25,24)
        self.spt_SAMPrior1 = SAMPrior(49, 48)
        self.spt_SPTUnit1 = SPTUnit(24, 24)
        self.spt_SPTUnit2 = SPTUnit(48, 48)

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

    def forward_features(self, I_real, I_dcp, D_real, D_dcp, mask):
        # 图像级深度提示 C_image
        C_image = torch.cat([D_real, D_dcp], dim=1)  # [B,2,H,W]
        F_input = torch.cat([I_real, I_dcp, C_image], dim=1)  # [B,8,H,W]

        # 生成 3 张 AMDP（输入分辨率）
        AMDPs = self.mdpan(C_image)  # tuple of 3 tensors: [B,1,H,W] * 3

        # 下采样到各层分辨率的函数
        def resize_amdp(a, x_like):
            return F.interpolate(a, size=x_like.shape[-2:], mode='bilinear', align_corners=False)

        # Stem
        x = self.patch_embed(F_input)
        # ----- Layer1 -----
        x = self.layer1(x)
        x = x * resize_amdp(AMDPs[0], x)  # 注入 AMDP1
        skip1 = x

        # SPT（保持你原有的 mask/先验）
        P = self.spt_SAMPrior(x, F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False))
        x1 = self.spt_SPTUnit1(x, F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False), P)
        x = x + x1

        # ----- Down1 -----
        x = self.patch_merge1(x)
        x = self.layer2(x)
        x = x * resize_amdp(AMDPs[1], x)  # 注入 AMDP2
        skip2 = x

        mask_2 = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        P1 = self.spt_SAMPrior1(x, mask_2)
        x2 = self.spt_SPTUnit2(x, mask_2, P1)
        x = x + x2

        # ----- Down2 -----
        x = self.patch_merge2(x)
        x = self.layer3(x)
        x = x * resize_amdp(AMDPs[2], x)  # 注入 AMDP3

        # ----- Up1 -----
        x = self.patch_split1(x)
        x = self.fusion1([x, self.skip2(skip2)]) + x

        mask_up1 = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        P1 = self.spt_SAMPrior1(x, mask_up1)
        x2 = self.spt_SPTUnit2(x, mask_up1, P1)
        x = x + x2

        x = self.layer4(x)

        # ----- Up2 -----
        x = self.patch_split2(x)
        x = self.fusion2([x, self.skip1(skip1)]) + x

        mask_up2 = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        P = self.spt_SAMPrior(x, mask_up2)
        x1 = self.spt_SPTUnit1(x, mask_up2, P)
        x = x + x1

        x = self.layer5(x)
        x = self.patch_unembed(x)
        return x

    def forward(self, I_real, I_dcp, D_real, D_dcp, mask):
        # 统一 pad 到 patch_size 的整数倍
        def pad_like(x):
            H, W = x.shape[2:]
            mod_pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
            mod_pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
            return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect'), H, W, mod_pad_h, mod_pad_w

        I_real, H, W, ph, pw = pad_like(I_real)
        I_dcp = F.pad(I_dcp, (0, pw, 0, ph), 'reflect')
        D_real = F.pad(D_real, (0, pw, 0, ph), 'reflect')
        D_dcp = F.pad(D_dcp, (0, pw, 0, ph), 'reflect')
        mask = F.pad(mask, (0, pw, 0, ph), 'reflect')

        feat = self.forward_features(I_real, I_dcp, D_real, D_dcp, mask)

        # 你原先的 K/B 方式保持不变
        K, B = torch.split(feat, (1, 3), dim=1)
        x = K * I_real - B + I_real
        x = x[:, :, :H, :W]

        x = self.coarse_convs(x) + self.seg_convs(mask[:, :, :H, :W])
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