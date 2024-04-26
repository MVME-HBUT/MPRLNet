# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Softmax

from collections import OrderedDict
from layers import *

class CSE(nn.Module):

    def __init__(self, in_channels):
        super(CSE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        squeezed = self.avgpool(U)
        excited = self.Conv_Excitation(self.Conv_Squeeze(squeezed))
        gate = self.norm(excited)
        return U * gate

class CONTEXT(nn.Module):

    def __init__(self, in_channels):
        super(CONTEXT, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // 8
        self.ConvQuery = nn.Conv2d(in_channels, self.channels, kernel_size=1)
        self.ConvKey = nn.Conv2d(in_channels, self.channels, kernel_size=1)
        self.ConvValue = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.SoftMax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.cse = CSE(in_channels)

    @staticmethod
    def _row_mask(B, H, W, device):
        m = torch.zeros(H, H, device=device).fill_diagonal_(float("-inf"))
        return m.unsqueeze(0).expand(B * W, -1, -1)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.ConvQuery(x)
        k = self.ConvKey(x)
        v = self.ConvValue(x)
        qh = q.permute(0, 3, 1, 2).reshape(b * w, self.channels, h).transpose(1, 2) 
        kh = k.permute(0, 3, 1, 2).reshape(b * w, self.channels, h)               
        vh = v.permute(0, 3, 1, 2).reshape(b * w, c, h)                     
        qw = q.permute(0, 2, 1, 3).reshape(b * h, self.channels, w).transpose(1, 2)  
        kw = k.permute(0, 2, 1, 3).reshape(b * h, self.channels, w)             
        vw = v.permute(0, 2, 1, 3).reshape(b * h, c, w)                 
        mask = self._row_mask(b, h, w, x.device)
        e_h = torch.bmm(qh, kh) + mask
        e_h = e_h.view(b, w, h, h).permute(0, 2, 1, 3)                       
        e_w = torch.bmm(qw, kw).view(b, h, w, w)             
        attn = self.SoftMax(torch.cat([e_h, e_w], dim=3)) 
        ah = attn[..., :h].permute(0, 2, 1, 3).reshape(b * w, h, h)
        aw = attn[..., h:h + w].reshape(b * h, w, w)
        oh = torch.bmm(vh, ah.transpose(1, 2)).view(b, w, c, h).permute(0, 2, 3, 1) 
        ow = torch.bmm(vw, aw.transpose(1, 2)).view(b, h, c, w).permute(0, 2, 1, 3) 
        se = self.cse(x)
        return x + self.gamma * (oh + ow + se)

class SEM(nn.Module):

    def __init__(self, channels=None, e_lambda=1e-4):
        super(SEM, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def extra_repr(self):
        return "lambda={:g}".format(self.e_lambda)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w - 1
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mu).pow(2)
        weight = var / (4.0 * (var.sum(dim=(2, 3), keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.act(weight)

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        bns = nn.BatchNorm2d

        self.convs = OrderedDict()
        for i in range(4, -1, -1):

            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

            self.chb = np.array([96, 96, 160, 304, 472])
            cha = self.chb[i]
            self.convs[("r", i, 1)] = SEM()
            self.convs[("r", i, 0)] = bns(cha)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.cc = CONTEXT(216)

    def forward(self, input_features):
        self.outputs = {}
        x = self.cc(input_features[-1])
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = upsample(x)
            if self.use_skips and i > 0:
                x = torch.cat((x, input_features[i - 1]), 1)
                if i < 4:
                    x = self.convs[("r", i, 1)](x)
                    x = self.convs[("r", i, 0)](x)
            x = self.convs[("upconv", i, 1)](x)
            self.outputs[('d_feature', i)] = x
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs