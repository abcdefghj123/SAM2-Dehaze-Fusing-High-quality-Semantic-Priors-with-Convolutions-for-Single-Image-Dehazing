import torch
import torch.nn as nn
import torch.nn.functional as F
from .deconv import DEConv
from .scconv import ScConv
from .RDB import RDB
from .DepthwiseSeparableConv import Depthwise_separable_conv
from .SPF import SPFUnit
from .SPT import SAMPrior
from .SPT import SPTUnit
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
        #self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
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
        #x = self.conv1(x)
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
    def __init__(self, in_chans=3, out_chans=4,
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

    def forward(self, x, mask):
        # print("mask",mask.shape)
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        feat = self.forward_features(x,mask)
        # print("feat", feat.shape)
        # 2022/11/26
        K, B = torch.split(feat, (1, 3), dim=1)

        x = K * x - B + x
        x = x[:, :, :H, :W]
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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MixDehazeNet_s()
    model.to(device)

    input_tensor = torch.randn(1, 3, 256, 256).to(device)  # 输入图像数据
    mask_tensor = torch.randn(1, 1, 256, 256).to(device)  # 创建一个与输入图像形状相同的 mask 数据

    # 调用 summary 查看模型概况
    print("Model Summary:")
    summary(model, input_data=(input_tensor, mask_tensor))  # 传递两个输入

    # 计算 MACs
    flops = FlopCountAnalysis(model, (input_tensor, mask_tensor))  # 传递两个输入参数
    print(f"MACs (in GFLOPS): {flops.total() / 1e9:.4f} GFLOPS")

    # 计算延迟
    timer = benchmark.Timer(
        stmt="model(input_tensor, mask_tensor)",  # 被测试的代码
        setup="import torch",  # 不需要重新定义 model，globals 中已传递
        globals={"model": model, "input_tensor": input_tensor, "mask_tensor": mask_tensor},
        # 将 model 和 input_tensor 正确传递
        num_threads=1  # 可以根据硬件环境调整线程数
    )

    # 测量延迟：执行 100 次前向传播，计算平均时间
    latency = timer.timeit(100)  # 测量 100 次前向传播的时间
    print(f"Latency (ms): {latency.mean * 1e3:.4f} ms")  # 将秒转为毫秒


