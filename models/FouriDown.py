import torch
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
import time
def ComplexResize(x, size, mode='bilinear', align_corners=False):
    # Split the tensor into its real and imaginary parts
    real_part = torch.real(x)
    imag_part = torch.imag(x)
    real_part_resized = F.interpolate(real_part, size=size, mode='bilinear', align_corners=False)
    imag_part_resized = F.interpolate(imag_part, size=size, mode='bilinear', align_corners=False)
    tensor_resized = torch.complex(real_part_resized, imag_part_resized)

    return tensor_resized

class FouriDown(nn.Module):

    def __init__(self, in_channel, base_channel):
        super(FouriDown, self).__init__()
        self.real_fuse = nn.Sequential(nn.Conv2d(in_channel * 4, in_channel * 4, 1, 1, 0, groups=in_channel),
                                       nn.LeakyReLU(0.1, inplace=False),
                                       nn.Conv2d(in_channel * 4, in_channel * 4, 1, 1, 0, groups=in_channel))
        self.imag_fuse = nn.Sequential(nn.Conv2d(in_channel * 4, in_channel * 4, 1, 1, 0, groups=in_channel),
                                       nn.LeakyReLU(0.1, inplace=False),
                                       nn.Conv2d(in_channel * 4, in_channel * 4, 1, 1, 0, groups=in_channel))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.Downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.channel2x = nn.Conv2d(in_channel, base_channel, 1, 1)

    def forward(self, x):

        B, C, H, W = x.shape
        img_fft = torch.fft.fft2(x)
        real = img_fft.real
        imag = img_fft.imag
        mid_row, mid_col = img_fft.shape[2] // 4, img_fft.shape[3] // 4

        # Split into 16 patches
        img_fft_A = img_fft[:, :, :mid_row, :mid_col]
        img_fft_2 = img_fft[:, :, :mid_row, mid_col:mid_col * 2]
        img_fft_1 = img_fft[:, :, :mid_row, mid_col * 2:mid_col * 3]
        img_fft_B = img_fft[:, :, :mid_row, mid_col * 3:]

        img_fft_5 = img_fft[:, :, mid_row:mid_row * 2, :mid_col]
        img_fft_6 = img_fft[:, :, mid_row:mid_row * 2, mid_col:mid_col * 2]
        img_fft_7 = img_fft[:, :, mid_row:mid_row * 2, mid_col * 2:mid_col * 3]
        img_fft_8 = img_fft[:, :, mid_row:mid_row * 2, mid_col * 3:]

        img_fft_9 = img_fft[:, :, mid_row * 2:mid_row * 3, :mid_col]
        img_fft_10 = img_fft[:, :, mid_row * 2:mid_row * 3, mid_col:mid_col * 2]
        img_fft_11 = img_fft[:, :, mid_row * 2:mid_row * 3, mid_col * 2:mid_col * 3]
        img_fft_12 = img_fft[:, :, mid_row * 2:mid_row * 3, mid_col * 3:]

        img_fft_C = img_fft[:, :, mid_row * 3:, :mid_col]
        img_fft_3 = img_fft[:, :, mid_row * 3:, mid_col:mid_col * 2]
        img_fft_4 = img_fft[:, :, mid_row * 3:, mid_col * 2:mid_col * 3]
        img_fft_D = img_fft[:, :, mid_row * 3:, mid_col * 3:]

        # Cluster superposing groups and spectral shuffle
        fuse_A = torch.cat((torch.cat((img_fft_A, img_fft_B), dim=-1), torch.cat((img_fft_C, img_fft_D), dim=-1)),
                           dim=-2)
        fuse_B = torch.cat((torch.cat((img_fft_1, img_fft_2), dim=-1), torch.cat((img_fft_4, img_fft_3), dim=-1)),
                           dim=-2)
        fuse_C = torch.cat((torch.cat((img_fft_9, img_fft_12), dim=-1), torch.cat((img_fft_5, img_fft_8), dim=-1)),
                           dim=-2)
        fuse_D = torch.cat((torch.cat((img_fft_11, img_fft_10), dim=-1), torch.cat((img_fft_7, img_fft_6), dim=-1)),
                           dim=-2)

        tensors = [fuse_A, fuse_B, fuse_C, fuse_D]
        heights = [tensor.shape[2] for tensor in tensors]
        widths = [tensor.shape[3] for tensor in tensors]
        if len(set(heights)) == 1 and len(set(widths)) == 1:
            fuse = torch.stack(tensors, dim=2)
        else:
            for t in tensors:
                resized_tensors = [ComplexResize(t, size=(H // 2, W // 2), mode='bilinear', align_corners=False) for
                                   tensor in tensors]
                fuse = torch.stack(resized_tensors, dim=2)

        # Adaptive attention
        fuse = fuse.view(B, 4 * C, H // 2, W // 2)
        real = fuse.real
        imag = fuse.imag
        real_weight = self.real_fuse(real)
        imag_weight = self.imag_fuse(imag)
        fuse_weight = torch.complex(real_weight, imag_weight)
        fuse_weight = fuse_weight.view(B, C, 4, H // 2, W // 2)
        real_sigmoid = F.softmax(fuse_weight.real + 0.25, dim=2)
        imag_sigmoid = F.softmax(fuse_weight.imag + 0.25, dim=2)
        fuse_weight = torch.complex(real_sigmoid, imag_sigmoid)
        fuse = torch.complex(real, imag)
        fuse = fuse.view(B, C, 4, H // 2, W // 2)
        fuse = fuse * fuse_weight

        # Superposing
        fuse = fuse.sum(dim=2)
        img = torch.abs(torch.fft.ifft2(fuse))
        img = img + self.Downsample(x)
        img = self.lrelu(self.channel2x(img))

        return img

if __name__ == '__main__':
    batch_size = 8  # 批量大小
    in_channel = 3  # 输入通道数（例如 RGB 图像）
    base_channel = 96  # 基础通道数（卷积层输出通道数）
    height, width = 256, 256  # 输入图像的尺寸

    # 随机生成一个测试输入图像（假设图像为 RGB 图像）
    x = torch.randn(batch_size, in_channel, height, width).cuda()

    # 实例化 FouriDown 模块并将其移到 GPU
    model = FouriDown(in_channel=in_channel, base_channel=base_channel).cuda()

    # 向前传播（前向传播）
    with torch.no_grad():
        output = model(x)

    # 打印输出的尺寸以验证其正确性
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {output.shape}")
