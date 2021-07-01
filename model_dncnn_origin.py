# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26},
#    number={7},
#    pages={3142-3155},
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# run this to test the model

import argparse
import os, time, datetime
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from math import sqrt
from tensorboardX import SummaryWriter

class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output
def BasicConv2d(inch, outch, k_size, pad = 0):
    return nn.Conv2d(in_channels=inch, out_channels=outch, kernel_size = k_size, stride = 1, padding
            = pad ,bias = False)

class conv_bn_relu(nn.Module):
    def __init__(self,in_ch,out_ch,k_size=3,pad = 1):
        super(conv_bn_relu,self).__init__()
        #self.conv = BasicConv2d(in_ch,out_ch,k_size,pad)
        n = int(out_ch/2)
        self.conv_reduce = BasicConv2d(in_ch,n,1)
        self.bn_reduce = nn.BatchNorm2d(n)
        self.act1 = nn.ReLU()
        self.conv_conv = CDilated(n,n,k_size,d=2)
        self.bn_conv = nn.BatchNorm2d(n)
        self.act2 = nn.ReLU()
        self.conv_expend = BasicConv2d(n,out_ch,1)
        self.bn_expend = nn.BatchNorm2d(out_ch)
        self.act3 = nn.ReLU()

    def forward(self,x):
        out1 = self.act1(self.bn_reduce(self.conv_reduce(x)))
        out2 = self.act2(self.bn_conv(self.conv_conv(out1)))
        out3 = self.act3(self.bn_expend(self.conv_expend(out2))+x)
        return out3
class inception(nn.Module):
    def __init__(self, image_channels=1, kernel_size=3,depth=5):
        super(inception, self).__init__()
        layers = []
        n_channels=64
        out_channels=1
        padding=1
        self.conv1 =  nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1,padding=0,bias=False)

        self.conv2_1 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,bias=False)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,padding=2,bias=False,)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.conv2_3 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=7,padding=3,bias=False)
        self.relu2_3 = nn.ReLU(inplace=True)
        self.final = nn.Conv2d(in_channels=96,out_channels=64,kernel_size=1,padding=0,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data,a=0,mode = 'fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(mean=0,std=sqrt(2./9./64.)).clamp_(-0.025,0.025)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        out = self.conv1(x)
        
        x1 = self.relu2_1(self.conv2_1(out))
        x2 = self.relu2_2(self.conv2_2(out))
        x3 = self.relu2_3(self.conv2_3(out))
        
        x_cat = torch.cat((x1,x2,x3),1)
        return nn.functional.relu(self.final(x_cat))

class finetune(nn.Module):
    def __init__(self, image_channels=2, kernel_size=3,depth=5):
        super(finetune, self).__init__()
        layers = []
        n_channels=64
        out_channels=1
        padding=1
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(inception())
        layers.append(nn.Conv2d(in_channels=64,out_channels=1,kernel_size=3,padding=1,bias=False))
        self.conv1 = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data,a=0,mode = 'fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(mean=0,std=sqrt(2./9./64.)).clamp_(-0.025,0.025)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        out = self.conv1(x)       
        return out


class Model(nn.Module):

    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(Model, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))
        self.mask = nn.Sequential(*layers)


    def forward(self,origin_x):
        x = origin_x.clone()
        out = self.mask(x)
        if origin_x.size()[1]==3:
            _,_,in_x = torch.chunk(origin_x, 3, dim=1)
        elif origin_x.size()[1]==2:
            _,in_x = torch.chunk(origin_x, 2, dim=1)
        else:
            in_x = origin_x
          
        return in_x+out,in_x

