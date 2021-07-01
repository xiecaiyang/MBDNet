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
import torch.nn.functional as F
class ConvRelu(nn.Module):
    '''
    This class defines the dilated convolution.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=0, bias=False, dilation=d)
        self.act = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(nOut)
    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.act(self.bn(self.conv(input)))
        return output

def BasicConv2d(inch, outch, k_size, pad = 0):
    return nn.Conv2d(in_channels=inch, out_channels=outch, kernel_size = k_size, stride = 1, padding
            = pad ,bias = False)

class Model(nn.Module):

    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(Model, self).__init__()
        kSize = kernel_size
        self.l1 = ConvRelu(1, 32, kSize, stride=1, d=1)
        self.l2 = ConvRelu(32, 64, kSize, stride=1, d=1)
        self.l3 = ConvRelu(64, 64, kSize, stride=1, d=1)
        self.l4 = ConvRelu(64, 128, kSize, stride=1, d=1)
        self.l5 = ConvRelu(128, 128, kSize, stride=2, d=1)
        self.l6 = ConvRelu(128, 256, kSize, stride=2, d=1)
        self.l7 = ConvRelu(256, 512, kSize, stride=2, d=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512,4)
        self.fc2 = nn.Linear(512,4)
        self.act = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data,a=0,mode = 'fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(mean=0,std=sqrt(2./9./64.)).clamp_(-0.025,0.025)
                nn.init.constant_(m.bias, 0)
    def forward(self,origin_x):
        output = self.l1(origin_x)
        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        output = self.l5(output)
        output = self.l6(output)
        output = self.l7(output)
        output = self.avgpool(output)
        output = output.view(-1,512)
        n_type = self.fc1(output)
        n_level = self.act(self.fc2(output))
        return n_type,n_level

