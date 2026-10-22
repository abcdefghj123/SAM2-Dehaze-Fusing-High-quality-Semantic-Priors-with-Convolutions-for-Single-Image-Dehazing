import torch
import torch.nn as nn
import torch.nn.functional as F


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    """
    动态上采样模块
    style='lp': 直接局部感知
    style='pl': PixelShuffle 后再局部感知
    """
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super(DySample, self).__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing="ij")).transpose(1, 2).repeat(1, self.groups, 1).reshape(
            1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H, dtype=x.dtype, device=x.device) + 0.5
        coords_w = torch.arange(W, dtype=x.dtype, device=x.device) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij")).transpose(1, 2).unsqueeze(1).unsqueeze(0)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(
            x.reshape(B * self.groups, -1, H, W), coords,
            mode='bilinear', align_corners=False, padding_mode="border"
        ).view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.01 + self.init_pos
        else:
            offset = self.offset(x) * 0.01 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.01 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.01 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


class DyConverse(nn.Module):
    """
    DyConverse 动态反卷积模块 (ICCV2025 Converse2D 改进版)
    用 DySample 替代固定上采样，提升细节恢复能力
    """
    def __init__(self, in_channels, scale=2, style='lp',
                 kernel_size=5, padding=4, padding_mode='circular', eps=1e-6):
        super(DyConverse, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.padding = padding
        self.padding_mode = padding_mode
        self.eps = eps

        assert self.out_channels == self.in_channels

        self.weight = nn.Parameter(torch.randn(1, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.ones(1, in_channels, 1, 1) * -1.0)  # 初始化为负，避免 biaseps 过小
        self.weight.data = F.softmax(
            self.weight.data.view(1, in_channels, -1), dim=-1
        ).view(1, in_channels, kernel_size, kernel_size)

        self.dy_upsample = DySample(in_channels, scale=scale, style=style, groups=4, dyscope=True)

    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, pad=[self.padding] * 4, mode=self.padding_mode, value=0)

        biaseps = torch.sigmoid(self.bias) + self.eps
        _, _, h, w = x.shape

        STy = self.dy_upsample(x)
        if self.scale != 1:
            x_up = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        else:
            x_up = x

        FB = self.p2o(self.weight, (h * self.scale, w * self.scale))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)
        FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))

        FR = FBFy + torch.fft.fftn(biaseps * x_up, dim=(-2, -1))
        x1 = FB * FR
        FBR = torch.mean(self.splits(x1, self.scale), dim=-1)
        invW = torch.mean(self.splits(F2B, self.scale), dim=-1)

        denom = torch.clamp(invW + biaseps, min=self.eps)
        invWBR = FBR / denom

        FCBinvWBR = FBC * invWBR.repeat(1, 1, self.scale, self.scale)
        FX = (FR - FCBinvWBR) / (biaseps + self.eps)
        out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

        if self.padding > 0:
            out = out[..., self.padding * self.scale:-self.padding * self.scale,
                      self.padding * self.scale:-self.padding * self.scale]

        return out

    def splits(self, a, scale):
        *leading_dims, W, H = a.size()
        W_s, H_s = W // scale, H // scale
        b = a.view(*leading_dims, scale, W_s, scale, H_s)
        permute_order = list(range(len(leading_dims))) + [
            len(leading_dims)+1, len(leading_dims)+3,
            len(leading_dims), len(leading_dims)+2
        ]
        b = b.permute(*permute_order).contiguous()
        b = b.view(*leading_dims, W_s, H_s, scale * scale)
        return b

    def p2o(self, psf, shape):
        otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
        otf[..., :psf.shape[-2], :psf.shape[-1]].copy_(psf)
        otf = torch.roll(
            otf, (-int(psf.shape[-2] / 2), -int(psf.shape[-1] / 2)), dims=(-2, -1)
        )
        otf = torch.fft.fftn(otf, dim=(-2, -1))
        return otf


if __name__ == "__main__":
    model = DyConverse(in_channels=32, scale=2, style='lp')
    x = torch.randn(1, 32, 64, 64)
    y = model(x)
    print("Input:", x.shape)
    print("Output:", y.shape)
