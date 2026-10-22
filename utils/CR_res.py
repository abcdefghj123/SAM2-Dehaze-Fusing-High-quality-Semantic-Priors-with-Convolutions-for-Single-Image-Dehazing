import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models


class Resnet152(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Resnet152, self).__init__()
        res_pretrained_features = models.resnet152(pretrained=True)
        self.slice1 = torch.nn.Sequential(*list(res_pretrained_features.children())[:-5])
        self.slice2 = torch.nn.Sequential(*list(res_pretrained_features.children())[-5:-4])
        self.slice3 = torch.nn.Sequential(*list(res_pretrained_features.children())[-4:-3])
        self.slice4 = torch.nn.Sequential(*list(res_pretrained_features.children())[-3:-2])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        return [h_relu1, h_relu2, h_relu3, h_relu4]


class ContrastLoss_res(nn.Module):
    def __init__(self, ablation=False):
        super(ContrastLoss_res, self).__init__()
        self.vgg = Resnet152().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0.0
        eps = 1e-6  # 安全项，防止除0
        for i in range(len(a_vgg)):
            a_feat, p_feat, n_feat = a_vgg[i], p_vgg[i], n_vgg[i]

            d_ap = self.l1(a_feat, p_feat.detach())
            if not self.ab:
                d_an = self.l1(a_feat, n_feat.detach())
                # 防止极小分母放大
                contrastive = d_ap / (d_an + eps)
            else:
                contrastive = d_ap

            # 防止单层异常放大
            contrastive = torch.clamp(contrastive, 0, 10.0)
            loss += self.weights[i] * contrastive

        # 最后整体数值保护
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e3, neginf=0.0)
        return loss
