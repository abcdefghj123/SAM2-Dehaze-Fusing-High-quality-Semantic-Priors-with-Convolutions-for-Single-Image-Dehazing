import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from thop import profile
import torch
import time
#被改进的模块 看Ai缝合怪b站视频：2025.8.2 更新的视频
class ChannelAttention(nn.Module):
    def __init__(self, in_planes=32, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)
class ConvAtt2(nn.Module):
    def __init__(self, in_channels, att_channels=16, lk_size=13, sk_size=3, reduction=2):
        """
        :param in_channels: AIFHG输入特征图通道数
        :param att_channels: AIFHG用于注意力通道数，默认为16
        :param lk_size: AIFHG 静态大核卷积核尺寸（如图中13）
        :param sk_size: AIFHG动态卷积核尺寸（如图中3）
        :param reduction: AIFHG动态卷积中间层压缩因子
        """
        super().__init__()
        self.in_channels = in_channels
        self.att_channels = att_channels
        self.idt_channels = in_channels - att_channels
        self.lk_size = lk_size
        self.sk_size = sk_size

        # 动态卷积核生成器
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(att_channels, att_channels // reduction, 1),
            nn.GELU(),#AIFHG
            nn.Conv2d(att_channels // reduction, att_channels * sk_size * sk_size, 1)
        )
        nn.init.zeros_(self.kernel_gen[-1].weight)
        nn.init.zeros_(self.kernel_gen[-1].bias)

        # 共享静态大核卷积核：定义为参数，非卷积层
        self.lk_filter = nn.Parameter(torch.randn(att_channels, att_channels, lk_size, lk_size))
        nn.init.kaiming_normal_(self.lk_filter, mode='fan_out', nonlinearity='relu')

        # 融合层
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.att_channels + self.idt_channels, f"Input channel {C} must match att + idt ({self.att_channels} + {self.idt_channels})"

        # 通道拆分
        F_att, F_idt = torch.split(x, [self.att_channels, self.idt_channels], dim=1)

        # 生成动态卷积核 [B * att, 1, 3, 3]
        kernel = self.kernel_gen(F_att).reshape(B * self.att_channels, 1, self.sk_size, self.sk_size)

        # 动态卷积操作
        F_att_re = rearrange(F_att, 'b c h w -> 1 (b c) h w')
        out_dk = F.conv2d(F_att_re, kernel, padding=self.sk_size // 2, groups=B * self.att_channels)
        out_dk = rearrange(out_dk, '1 (b c) h w -> b c h w', b=B, c=self.att_channels)

        # 静态大核卷积
        out_lk = F.conv2d(F_att, self.lk_filter, padding=self.lk_size // 2)

        # 融合（两个卷积结果加和）
        out_att = out_lk + out_dk

        # 拼接 F_idt（保留通道）
        out = torch.cat([out_att, F_idt], dim=1)

        # 1x1 融合
        out = self.fusion(out)
        return out
class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x



# Dynamic Perception Large-kernel Attention Module (DPLKA)
class DPLKA(nn.Module): #lk_size=[5,7,9]、[7,9,11]、[9,11,13]、[11,13,15]
    def __init__(self,n_feat,lk_size=[9,11,13]):  # 1.这个lk_size这个列表，可以灵活设置不同核的列表，如果列表长度为4，就是4分支，比如：lk_size=[7,9,11,13]，跑实验看效果，比如不同核列表组 可以做消融实验，分支数也可以做消融实验

        super(DPLKA, self).__init__()
        modules_body = []
        att_channels = 16   # 2.这个划分通道数也可以灵活去设置，不一定就是16，可设置一个动态的，比如原始输入通道数的: n_feat// 2, n_feat// 3 , n_feat// 4 , n_feat// 5  ,跑实验看效果，可以做消融实验哪一种效果是最佳的
        self.ConvAtt2_body =nn.ModuleList()
        self.lk_length = len(lk_size)
        bn = True
        act = nn.ReLU(True)
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3,padding=1, bias=False))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        self.CA = ChannelAttention(in_planes=n_feat)
        self.body = nn.Sequential(*modules_body)
        for i in range(self.lk_length):  # i是从0开始取
            self.ConvAtt2_body.append(nn.Sequential(ConvAtt2(in_channels=n_feat, att_channels=att_channels, lk_size=lk_size[i])))
        self.cbr = CBR(n_feat*self.lk_length,n_feat,kernel_size=3)   #3. 这个CBR模块中的卷积核采用的是k=3,到时候自己也可以设置k=1，跑实验看效果
    def forward(self, x):
        res1 = self.body(x)  # 4.这部分主要采用两层普通卷积去提取增强局部特征
        res = self.CA(res1)  # 5.利用通道注意力
        ConvAtt_List = []    # 6.多分支采用的是ConvAtt卷积注意力模块 ，因为它的大核默认是13，所有我们采用动态的核列表组，相比较静态核来说，感知局部和全局效果会更好！
        for i in range(self.lk_length):
            ConvAtt_List.append(self.ConvAtt2_body[i](res))
        if self.lk_length > 1:  # 7. 我在这里做了一个判断，如果我们采用的不是单分支，在此处需要对ConvAtt_List 列表中的特征图 在通道维度上进行拼接
            res = torch.cat(ConvAtt_List, dim=1)
        else:
            res = ConvAtt_List[0] #8.如果是单分支，取列表中的值
            res = channel_shuffle(res, groups=4)   #9. 进行通道洗牌操作

        res = self.cbr(res) # 10.注意力加权操作
        res = res * res1
        res = res + x # 11.残差连接，防止信息的丢失
        return res

class FeedForward(nn.Module):
    """
    MLP block with pre-layernorm, GELU activation, and dropout.
    """

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x=x.permute(0,2,3,1)
        return self.net(x).permute(0,3,1,2)
class DPLKABlock(nn.Module):
    """
    Transformer block stacking multiple Multipole_Attention2D + FeedForward layers.
    """

    def __init__(
        self,
        n_feat,
        lk_size=[9, 11, 13],
        att_channels=16,
        depth=2 # 堆叠层数
    ):
        super().__init__()
        self.norm = nn.BatchNorm2d(n_feat)
        self.layers = nn.ModuleList([])
        mlp_dim = int(4*n_feat)  # FeedForward中间层维度
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        DPLKA(
                            n_feat,
                            lk_size,
                            att_channels

                        ),
                        FeedForward(n_feat, mlp_dim),
                    ]
                )
            )

    def forward(self, x):
        """
        Expected input shape: [B, C,H, W]
        """
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# 创建一个DPLKABlock实例
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inp = torch.randn(1, 32, 64, 64).to(device)

    model = DPLKA(n_feat=32, lk_size=[9,11,13], att_channels=16).to(device).eval()
    if device.type == "cuda": torch.cuda.synchronize()

    t0 = time.time()
    _ = model(inp)
    if device.type == "cuda": torch.cuda.synchronize()
    print("DPLKA forward ok. in:", inp.shape, "out:", _.shape, "time:", time.time() - t0)

    if profile is not None:
        macs, params = profile(model, inputs=(inp,))
        print(f"Params: {params/1e6:.3f} M")
        print(f"FLOPs:  {macs/1e9:.3f} G")
    else:
        print("thop 未安装，跳过 FLOPs 统计")

