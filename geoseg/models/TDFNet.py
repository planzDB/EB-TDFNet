import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

import math

import numpy as np
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm

from .vmamba import VSSM


class ConvBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, sp=False, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvBlock(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
            
        self.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            #nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        stages_out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            stages_out.append(x)
        return stages_out

    def forward(self, x):
        x = self.forward_features(x)
        """
        torch.Size([1, 96, 128, 128])
        torch.Size([1, 192, 64, 64])
        torch.Size([1, 384, 32, 32])
        torch.Size([1, 768, 16, 16])
        """
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {

    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
}


@register_model
def convnext_encoder(pretrained=True,in_22k=True, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],**kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"],strict=False)
        # print('Convnext_pretrained')
        
    return model


class LRDU(nn.Module):
    def __init__(self,in_c,factor):
        super(LRDU,self).__init__()

        self.up_factor = factor
        self.factor1 = factor*factor//2
        self.factor2 = factor*factor
        self.up = nn.Sequential(
            nn.Conv2d(in_c, self.factor1*in_c, (1,7), padding=(0, 3), groups=in_c),
            nn.Conv2d(self.factor1*in_c, self.factor2*in_c, (7,1), padding=(3, 0), groups=in_c),
            nn.PixelShuffle(factor)
        )

    def forward(self,x):
        x = self.up(x)
        return x

    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    
class cSE(nn.Module):  # noqa: N801
    """
    The channel-wise SE (Squeeze and Excitation) block from the
    `Squeeze-and-Excitation Networks`__ paper.
    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65939
    and
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    Shape:
    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    __ https://arxiv.org/abs/1709.01507
    """

    def __init__(self, in_channels: int, r: int = 16):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
            r: The reduction ratio of the intermediate channels.
                Default: 16.
        """
        super().__init__()
        self.linear1 = nn.Linear(in_channels, in_channels // r)
        self.linear2 = nn.Linear(in_channels // r, in_channels)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        input_x = x

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear1(x), inplace=True)
        x = self.linear2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)
        return x

class sSE(nn.Module):  # noqa: N801
    """
    The sSE (Channel Squeeze and Spatial Excitation) block from the
    `Concurrent Spatial and Channel ‘Squeeze & Excitation’
    in Fully Convolutional Networks`__ paper.
    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    Shape:
    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    __ https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels: int):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        input_x = x

        x = self.conv(x)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)
        return x


class scSE(nn.Module):  # noqa: N801
    """
    The scSE (Concurrent Spatial and Channel Squeeze and Channel Excitation)
    block from the `Concurrent Spatial and Channel ‘Squeeze & Excitation’
    in Fully Convolutional Networks`__ paper.
    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    Shape:
    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    __ https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels: int, r: int = 16):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
            r: The reduction ratio of the intermediate channels.
                Default: 16.
        """
        super().__init__()
        self.cse_block = cSE(in_channels, r)
        self.sse_block = sSE(in_channels)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        cse = self.cse_block(x)
        sse = self.sse_block(x)
        x = torch.add(cse, sse)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)


class DWconv(nn.Module):
    def __init__(self,in_ch,out_ch,stride=1,padding=1,dilation=1):
        super(DWconv, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class BFM(nn.Module):
    def __init__(self,ch_in, r_2, ch_out, drop_rate=0.1):
        super(BFM, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate
        self.sigmoid = nn.Sigmoid()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.dw1 = DWconv(ch_in,ch_in // r_2,padding=1)
        self.dw2 = DWconv(ch_in, ch_in // r_2,padding=1)
        self.dw3 = DWconv(ch_in // r_2, ch_in, padding=1)
        
        self.scse = scSE(ch_in)
        self.dw4 = DWconv(ch_in, ch_in, padding=1)
        
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in*2, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        
        self.conv_skip = nn.Sequential(
            nn.Conv2d(ch_in*2, ch_out, kernel_size=3, stride=1, padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
        )
        self.se = cSE(ch_out)

    def forward(self, g, x):
        ##Mamba_branch
        y1 = self.avg_pool(x)
        y1 = self.dw1(y1)
        y2 = self.max_pool(x)
        y2 = self.dw2(y2)
        y = self.relu(y1+y2)
        y = self.dw3(y)
        y = self.sigmoid(y)*x

        ##CNN_branch
        c1 = self.scse(g)
        c1 = self.dw4(c1)
        c2 = self.sigmoid(c1)*g
        
        fuse = torch.cat([y, c2], 1)
        fuse= self.se(self.conv(fuse) + self.conv_skip(fuse))
        
        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse  

class EDFM(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(EDFM,self).__init__()
        self.W_g = nn.Sequential(
            DWconv(ch_out, ch_out, stride=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            )
        self.W_x = nn.Sequential(
            nn.Upsample(scale_factor=2),
            DWconv(ch_in, ch_out, stride=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.psi = nn.Sequential(
            DWconv(ch_out, ch_out, stride=1),
            nn.BatchNorm2d(ch_out),
            nn.Sigmoid()
        )

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = g1 + x1
        psi1 = self.psi(psi)*g1
        return psi1    
    
class TDFNet(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=2,
                 mid_channel = 96,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.1,
                 load_ckpt_path='/root/EB-TDFNet/pre_trained_weights/vmamba_tiny_e292.pth',
                 pretrained = True
                ):
        super().__init__()
        
        self.num_classes = num_classes
        self.load_ckpt_path = load_ckpt_path
        # ca_sa
        self.ca_1 = ChannelAttention(1*mid_channel)
        self.sa_1 = SpatialAttention() 

        self.ca_2 = ChannelAttention(2*mid_channel)
        self.sa_2 = SpatialAttention()

        self.ca_3 = ChannelAttention(4*mid_channel)
        self.sa_3 = SpatialAttention()

        self.ca_4 = ChannelAttention(8*mid_channel)
        self.sa_4 = SpatialAttention()
        
        self.Translayer_1 = BasicConv2d(1*mid_channel, 1*mid_channel, 1)
        self.Translayer_2 = BasicConv2d(2*mid_channel, 2*mid_channel, 1)
        self.Translayer_3 = BasicConv2d(4*mid_channel, 4*mid_channel, 1)
        self.Translayer_4 = BasicConv2d(8*mid_channel, 8*mid_channel, 1)  
        
        self.fuse_mamba_cnn1 = BFM(ch_in=96, r_2=1, ch_out=96, drop_rate=0.)
        self.fuse_mamba_cnn2 = BFM(ch_in=192, r_2=1, ch_out=192, drop_rate=0.)
        self.fuse_mamba_cnn3 = BFM(ch_in=384, r_2=2, ch_out=384, drop_rate=0.)
        self.fuse_mamba_cnn4 = BFM(ch_in=768, r_2=2, ch_out=768, drop_rate=0.)
        
        self.Translayer_11 = BasicConv2d(mid_channel, mid_channel, 1)
        self.Translayer_21 = BasicConv2d(2*mid_channel, mid_channel, 1)
        self.Translayer_31 = BasicConv2d(4*mid_channel, mid_channel, 1)
        self.Translayer_41 = BasicConv2d(8*mid_channel, mid_channel, 1) 
        
        self.edfm3 = EDFM(ch_in=192, ch_out=96)
        self.edfm2 = EDFM(ch_in=384, ch_out=192)
        self.edfm1 = EDFM(ch_in=768, ch_out=384)

        self.deconv3 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)

        self.seg_outs = nn.Conv2d(mid_channel, mid_channel, 1, 1)
        
        self.Up4x = LRDU(96,4)      
        self.convout = nn.Conv2d(96, num_classes, kernel_size=1, stride=1, padding=0)
            
        self.backbone = convnext_encoder(pretrained,True)
        
        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate
                        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数 
            # print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            # not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            # print('Not loaded keys:', not_loaded_keys)
            # print("vmunet loaded finished!")

    
    def forward(self, x):
        seg_outs = []
        if x.size()[1] == 1: # 如果是灰度图，就将1个channel 转为3个channel
            x = x.repeat(1,3,1,1)
        f1, f2, f3, f4 = self.vmunet(x) #  f1 [2, 128, 128, 96]  f3  [2, 16, 16, 768]  [b h w c]
        # b h w c --> b c h w
        f1 = f1.permute(0, 3, 1, 2) # f1 [2, 96, 128, 128]
        f2 = f2.permute(0, 3, 1, 2)
        f3 = f3.permute(0, 3, 1, 2)
        f4 = f4.permute(0, 3, 1, 2)
        
        # use sdi  
        f1 = self.ca_1(f1) * f1
        f1 = self.sa_1(f1) * f1
        f1 = self.Translayer_1(f1) # f1 [2, 96, 128, 128]
        
        f2 = self.ca_2(f2) * f2
        f2 = self.sa_2(f2) * f2
        f2 = self.Translayer_2(f2) # f2 [2, 96, 64, 64]

        f3 = self.ca_3(f3) * f3
        f3 = self.sa_3(f3) * f3
        f3 = self.Translayer_3(f3) # f3 [2, 96, 32, 32]

        f4 = self.ca_4(f4) * f4
        f4 = self.sa_4(f4) * f4
        f4 = self.Translayer_4(f4) # f4 [2, 96, 16, 16]

        g1,g2,g3,g4 = self.backbone(x)

        y1 = self.fuse_mamba_cnn1(g1,f1)
        y2 = self.fuse_mamba_cnn2(g2,f2)
        y3 = self.fuse_mamba_cnn3(g3,f3)
        y4 = self.fuse_mamba_cnn4(g4,f4)
        
        y31 = self.edfm1(y3,y4)
        y21 = self.edfm2(y2,y31)
        y11 = self.edfm3(y1,y21)
        
        y1 = self.Translayer_11(y11)
        y2 = self.Translayer_21(y21)
        y3 = self.Translayer_31(y31)
        
        y = self.deconv3(y3) + y2
        y = self.deconv4(y) + y1
        y = self.seg_outs(y)
        
        # d1 = F.interpolate(y, scale_factor=4, mode='bilinear')
        
        d2 = self.Up4x(y)
        d1 = self.convout(d2)
        
        return d1
                   
          
        


# if __name__ == "__main__":
#     pretrained_path ='/root/BuildingExtraction/pre_trained_weights/vmamba_tiny_e292.pth'
#     model = VMUNetV2(load_ckpt_path=pretrained_path).cuda()
#     model.load_from()
    
#     img = torch.randn(2, 3, 512, 512).cuda()
#     output = model(img)
    
#     if 1:
#         from fvcore.nn import FlopCountAnalysis, parameter_count_table
#         flops = FlopCountAnalysis(model, img)
#         print("FLOPs: %.4f G" % (flops.total()/1e9))

#         total_paramters = 0
#         for parameter in model.parameters():
#             i = len(parameter.size())
#             p = 1
#             for j in range(i):
#                 p *= parameter.size(j)
#             total_paramters += p
#         print("Params: %.4f M" % (total_paramters / 1e6)) 

