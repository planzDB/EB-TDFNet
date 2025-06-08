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
import os.path
import warnings

from .vmamba import VSSM
from torchvision import models

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
        
        self.cbr = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.cbr(x)
        return x
    
class CombinedAtt_BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()
        self.cbr = BasicConv2d(in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1)
    def forward(self, x):
        x_ca = self.ca(x) * x  # Channel Attention applied to x
        x_sa = self.sa(x_ca) * x_ca  # Spatial Attention applied to the output of Channel Attention
        x = self.cbr(x_sa)
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
        self.dw3 = DWconv(ch_in // r_2, ch_out, padding=1)
        
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
        # c2 = self.sigmoid(c1)*c1
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
                 drop_path_rate=0.1,
                 load_ckpt_path='/root/EB-TDFNet/pre_trained_weights/vmamba_tiny_e292.pth',
                 pretrained = True
                ):
        super().__init__()
        
        self.num_classes = num_classes
        self.load_ckpt_path = load_ckpt_path

#         self.Translayer_1 = BasicConv2d(1*mid_channel, 1*mid_channel, 1)
#         self.Translayer_2 = BasicConv2d(2*mid_channel, 2*mid_channel, 1)
#         self.Translayer_3 = BasicConv2d(4*mid_channel, 4*mid_channel, 1)
#         self.Translayer_4 = BasicConv2d(8*mid_channel, 8*mid_channel, 1)
        self.ca_sa_trans_1 = CombinedAtt_BasicConv2d(1*mid_channel, 1*mid_channel, 1)
        self.ca_sa_trans_2 = CombinedAtt_BasicConv2d(2*mid_channel, 2*mid_channel, 1)
        self.ca_sa_trans_3 = CombinedAtt_BasicConv2d(4*mid_channel, 4*mid_channel, 1)
        self.ca_sa_trans_4 = CombinedAtt_BasicConv2d(8*mid_channel, 8*mid_channel, 1)
        
        self.Translayer_5 = BasicConv2d(128, 1*mid_channel, 1,stride=2)
        self.Translayer_6 = BasicConv2d(256, 2*mid_channel, 1,stride=2)
        self.Translayer_7 = BasicConv2d(512, 4*mid_channel, 1,stride=2)
        self.Translayer_8 = BasicConv2d(512, 8*mid_channel, 1,stride=2)  
        
        self.fuse_mamba_cnn1 = BFM(ch_in=96, r_2=1, ch_out=96, drop_rate=0.)
        self.fuse_mamba_cnn2 = BFM(ch_in=192, r_2=1, ch_out=192, drop_rate=0.)
        self.fuse_mamba_cnn3 = BFM(ch_in=384, r_2=2, ch_out=384, drop_rate=0.)
        self.fuse_mamba_cnn4 = BFM(ch_in=768, r_2=2, ch_out=768, drop_rate=0.)
        
        self.edfm3 = EDFM(ch_in=192, ch_out=96)
        self.edfm2 = EDFM(ch_in=384, ch_out=192)
        self.edfm1 = EDFM(ch_in=768, ch_out=384)
        
        self.Translayer_11 = BasicConv2d(mid_channel, mid_channel, 1)
        self.Translayer_21 = BasicConv2d(2*mid_channel, mid_channel, 1)
        self.Translayer_31 = BasicConv2d(4*mid_channel, mid_channel, 1)
        self.Translayer_41 = BasicConv2d(8*mid_channel, mid_channel, 1) 

        self.deconv3 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)

        self.seg_outs = nn.Conv2d(mid_channel, mid_channel, 1, 1)
        
        self.Up4x = LRDU(96,4)      
        self.convout = nn.Conv2d(96, num_classes, kernel_size=1, stride=1, padding=0)

        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 64
        self.down2 = vgg16_bn.features[12:22]  # 64
        self.down3 = vgg16_bn.features[22:32]  # 64
        self.down4 = vgg16_bn.features[32:42]  # 64
        
        self.vmencoder = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
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
            model_dict = self.vmencoder.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数 
            # print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmencoder.load_state_dict(model_dict)

            # not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            # print('Not loaded keys:', not_loaded_keys)
            # print("vmunet loaded finished!")

    
    def forward(self, x):
        if x.size()[1] == 1: # 如果是灰度图，就将1个channel 转为3个channel
            x = x.repeat(1,3,1,1)
        f1, f2, f3, f4 = self.vmencoder(x) #  f1 [2, 128, 128, 96]  f3  [2, 16, 16, 768]  [b h w c]
        # b h w c --> b c h w
        f1 = f1.permute(0, 3, 1, 2) # f1 [2, 96, 128, 128]
        f2 = f2.permute(0, 3, 1, 2)
        f3 = f3.permute(0, 3, 1, 2)
        f4 = f4.permute(0, 3, 1, 2)
  
        f1 = self.ca_sa_trans_1(f1)
        f2 = self.ca_sa_trans_2(f2)
        f3 = self.ca_sa_trans_3(f3)
        f4 = self.ca_sa_trans_4(f4)

        #vgg16/vgg19
        size = x.size()[2:]
        layer1 = self.inc(x)
        layer2 = self.down1(layer1)
        layer3 = self.down2(layer2)
        layer4 = self.down3(layer3)
        layer5 = self.down4(layer4)
        
        g1 = self.Translayer_5(layer2)
        g2 = self.Translayer_6(layer3)
        g3 = self.Translayer_7(layer4)
        g4 = self.Translayer_8(layer5)

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
        
        d2 = self.Up4x(y)
        d1 = self.convout(d2)
        
        return d1
                   
          
    
if __name__ == "__main__":
    pretrained_path ='/root/BuildingExtraction/pre_trained_weights/vmamba_tiny_e292.pth'
    model = TDFNet(load_ckpt_path=pretrained_path).cuda()
    model.load_from()
    
    img = torch.randn(2, 3, 512, 512).cuda()
    output = model(img)
    
    if 1:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        flops = FlopCountAnalysis(model, img)
        print("FLOPs: %.4f G" % (flops.total()/1e9))

        total_paramters = 0
        for parameter in model.parameters():
            i = len(parameter.size())
            p = 1
            for j in range(i):
                p *= parameter.size(j)
            total_paramters += p
        print("Params: %.4f M" % (total_paramters / 1e6)) 