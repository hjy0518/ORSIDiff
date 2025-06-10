
import numpy as np
""""
backbone is SwinTransformer
"""
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
import torch.nn.functional as F
import os
from timm.models.layers import to_2tuple, trunc_normal_
# import onnx
import math
from einops import repeat
from timm.models.layers import DropPath

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )



def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding





class FM(nn.Module):
    def __init__(self, dim):
        super(FM, self).__init__()
        self.dim = dim
        self.ca = ChannelAttention(dim)
        self.att = Attention(dim,64)
        self.br1 = DWConv(32,3)
        self.conv = nn.Conv2d(128,64,kernel_size=1)

        self.time_embed = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim),
            nn.SiLU(),
            nn.Linear(4 * self.dim, self.dim),
        )

    def forward(self, r,timesteps):
        time_token = self.time_embed(timestep_embedding(timesteps, self.dim))
        time_token = time_token.unsqueeze(dim=1)
        time_token = time_token.unsqueeze(dim=1)

        time_token = time_token.transpose(3, 1)
        r = time_token + r
        ca = self.ca(r)
        out_r = r * ca +r

        out = self.att(out_r)
        x1,x2 = torch.chunk(out,2,1)
        x1 = self.br1(x1)
        out = self.conv(torch.cat((out,x1,x2),1))

        return out


class Attention(nn.Module):
    def __init__(self, dim,oup):
        super(Attention, self).__init__()
        self.dim = dim//4
        self.ca = ChannelAttention(dim)
        self.sa_conv1 = nn.Conv2d(self.dim,1,kernel_size=7,padding=3)
        self.sa_conv2 = nn.Conv2d(self.dim, 1, kernel_size=5,padding=2)
        self.sa_conv3 = nn.Conv2d(self.dim, 1, kernel_size=3,padding=1)
        self.sa_conv4 = nn.Conv2d(self.dim, 1, kernel_size=1,padding=0)

        self.sa_fusion = nn.Conv2d(1, 1, kernel_size=3,padding=1)
        self.conv_end = nn.Conv2d(dim*2, oup,kernel_size=1)
    def forward(self, x):
        ca = self.ca(x)
        x = x * ca

        x1,x2,x3,x4 = torch.chunk(x,4,dim=1)
        sa1 = self.sa_conv1(x1)
        sa2 = self.sa_conv2(x2)
        sa3 = self.sa_conv3(x3)
        sa4 = self.sa_conv4(x4)

        x1 = sa1 * x1
        x2 = sa2 * x2
        x3 = sa3 * x3
        x4 = sa4 * x4

        sa_fusion = self.sa_fusion(sa1 + sa2 + sa3 + sa4)
        out = self.conv_end(torch.cat((x,x1, x2, x3, x4),1))
        out = out * sa_fusion

        return out

class Decode(nn.Module):
    def __init__(self, in1,in2,in3,in4):
        super(Decode, self).__init__()
        self.dowm = nn.AvgPool2d(2)
        self.xconv_down1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            self.dowm,
            nn.Conv2d(32,64,kernel_size=1),
            self.dowm,
            nn.Conv2d(64, 32, kernel_size=1),
        )

        self.xconv_down2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            self.dowm,
            nn.Conv2d(64,32,kernel_size=1),
        )

        self.xconv_down3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            self.dowm,
            nn.Conv2d(64,32,kernel_size=1),
        )

        self.xconv_down4 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            self.dowm,
            nn.Conv2d(64,32,kernel_size=1),
        )





        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_dw1 = nn.Sequential(
            conv1x1(in1*3,in1),

            self.upsample2
        )
        self.conv_dw2 = nn.Sequential(
            conv1x1(in2*3, in1),
            self.upsample2
        )
        self.conv_dw3 = nn.Sequential(
            conv1x1(in3*3,in2),
            self.upsample2
        )

        self.conv_dw4 = nn.Sequential(
            conv1x1(in4*2, in3),
            self.upsample2
        )


        self.conv_up4 = nn.Sequential(
            nn.Conv2d(in_channels=in4*2, out_channels=in3, kernel_size=1),
            nn.BatchNorm2d(in3),
            nn.GELU(),
            nn.Conv2d(in_channels=in3, out_channels=in3, kernel_size=1),
            self.upsample2
        )
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(in_channels=in3*3, out_channels=in2, kernel_size=1),
            nn.BatchNorm2d(in2),
            nn.GELU(),
            nn.Conv2d(in_channels=in2, out_channels=in2, kernel_size=1),
            self.upsample2
        )
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(in_channels=in2*3, out_channels=in1, kernel_size=1),
            nn.BatchNorm2d(in1),
            nn.GELU(),
            nn.Conv2d(in_channels=in1, out_channels=in1, kernel_size=1),
            self.upsample2
        )
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(in_channels=in1*3, out_channels=in1, kernel_size=1),
            nn.BatchNorm2d(in1),
            nn.GELU(),
            nn.Conv2d(in_channels=in1, out_channels=in1, kernel_size=1),
            self.upsample2
        )


        self.p_1 = nn.Sequential(
            nn.Conv2d(in_channels=in1, out_channels=in1//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in1//2),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=in1//2, out_channels=1, kernel_size=3, padding=1, bias=True),
        )

        self.p2 = nn.Conv2d(in1, 1, kernel_size=3, padding=1)
        self.p3 = nn.Conv2d(in2, 1, kernel_size=3, padding=1)
        self.p4 = nn.Conv2d(in3, 1, kernel_size=3, padding=1)

        self.crs4 = ConvAttention(in3)
        self.crs3 = ConvAttention(in2)
        self.crs2 = ConvAttention(in1)
        self.crs1 = ConvAttention(in1)


    def forward(self,x1,x2,x3,x4,xt,s):
        xt1 = self.xconv_down1(xt)
        xt2 = self.xconv_down2(xt1)
        xt3 = self.xconv_down3(xt2)
        xt4 = self.xconv_down4(xt3)


        x1_1, x1_2 = torch.chunk(x1, 2, 1)
        x2_1, x2_2 = torch.chunk(x2, 2, 1)
        x3_1, x3_2 = torch.chunk(x3, 2, 1)
        x4_1, x4_2 = torch.chunk(x4, 2, 1)

        up4 = self.conv_up4(torch.cat((x4_2,xt4),1))
        dw4 = self.conv_dw4(torch.cat((x4_1,xt4),1))
        up4,dw4,pre4 = self.crs4(up4,dw4)

        up3 = self.conv_up3(torch.cat((up4, x3_2,xt3),1))
        dw3 = self.conv_dw3(torch.cat((x3_1, dw4,xt3),1))
        up3, dw3, pre3 = self.crs3(up3, dw3)

        up2 = self.conv_up2(torch.cat((up3, x2_2,xt2), 1))
        dw2 = self.conv_dw2(torch.cat((x2_1, dw3,xt2),1))
        up2, dw2, pre2 = self.crs2(up2, dw2)

        up1 = self.conv_up1(torch.cat((up2, x1_2,xt1), 1))
        dw1 = self.conv_dw1(torch.cat((x1_1, dw2,xt1), 1))
        up1, dw1, pre1 = self.crs1(up1, dw1)

        pred1 = self.p_1(pre1)
        pred2 = F.interpolate(self.p2(pre2), size=s, mode='bilinear')
        pred3 = F.interpolate(self.p3(pre3), size=s, mode='bilinear')
        pred4 = F.interpolate(self.p4(pre4), size=s, mode='bilinear')

        return pred1,pred2,pred3,pred4

class ConvAttention(nn.Module):
    def __init__(self,channel):
        super(ConvAttention, self).__init__()

        self.qsa = SpatialAttention()
        self.ksa = SpatialAttention()
        self.query_conv = nn.Conv2d(1, 1, kernel_size=1)
        self.key_conv = nn.Conv2d(1, 1, kernel_size=1)
        self.qk_conv = nn.Conv2d(2, 1, kernel_size=1)
        self.value_conv_1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.value_conv_2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.qkv = nn.Conv2d(1, 1, kernel_size=1)
        self.conv_1 = conv1x1(channel, channel)
        self.conv_2 = conv1x1(channel, channel, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1),
            DA(channel),
        )


    def forward(self, x1, x2):

        proj_query = self.query_conv(self.qsa(x1))
        proj_key = self.key_conv(self.ksa(x2))
        energy = torch.cat((proj_query, proj_key),1)
        energy = self.qk_conv(energy)

        proj_value_1 = self.value_conv_1(x1)
        proj_value_2 = self.value_conv_2(x2)

        out_1 = proj_value_1*energy
        out_1 = self.conv_1(out_1 + x1)

        out_2 = proj_value_2*energy
        out_2 = self.conv_2(out_2 + x2)
        out12 = torch.cat((out_1,out_2),1)

        x_out = self.conv(out12) * self.qkv(energy)

        return out_1,out_2,x_out

class DWConv(nn.ModuleList):
    def __init__(self,dim,krenel_dim):
        super(DWConv,self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=krenel_dim, padding=krenel_dim//2, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        mip = min(8, in_planes // ratio)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, mip, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(mip, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = self.sigmoid(max_out)

        return out

class Channel_Avg_Attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Channel_Avg_Attention, self).__init__()
        mip = min(8, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, mip, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(mip, in_planes//2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = self.sigmoid(out)

        return out


class Channel_Max_Attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Channel_Max_Attention, self).__init__()
        mip = min(8, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, mip, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(mip, in_planes//2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = self.sigmoid(out)

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class DA(nn.ModuleList):
    def __init__(self,dim):
        super(DA,self).__init__()
        self.res = nn.Conv2d(dim,dim,kernel_size=1)
        self.dwconv5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)  # depthwise conv
        self.conv_spatial = nn.Conv2d(dim,dim,kernel_size=7,stride=1,padding=9,groups=dim,dilation=3)

        self.avg_ca = Channel_Avg_Attention(dim*2)
        self.max_ca = Channel_Max_Attention(dim*2)
        self.conv3 = nn.Conv2d(dim*2,dim,kernel_size=1)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.res(x)
        x1 = self.dwconv5(x)
        x2 = self.conv_spatial(x1)
        x12 = torch.cat((x1,x2),1)
        avg_ca = self.avg_ca(x12)
        max_ca = self.max_ca(x12)
        x1 = x1 * avg_ca
        x2 = x2 * max_ca
        out = self.conv3(torch.cat((x1,x2),1))

        out = out.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)

        out = out.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        out = out + res

        return out

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class DWConvFFN(nn.ModuleList):
    def __init__(self,dim):
        super(DWConvFFN,self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        x = res + x
        return x


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    unloader = torchvision.transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

if __name__ == '__main__':
    import torch
    import torchvision
    from thop import profile

    model = MyNet().cuda()

    a = torch.randn(1, 3, 384, 384).cuda()
    b = torch.randn(1, 3, 384, 384).cuda()
    flops, params = profile(model, (a,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

