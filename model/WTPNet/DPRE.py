#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

import torch
from matplotlib import pyplot as plt
from torch import nn

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "sigmoid":
        module = nn.Sigmoid()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

#         super().__init__()
#         self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)
#
#     def forward(self, x):
#         patch_top_left  = x[...,  ::2,  ::2]
#         patch_bot_left  = x[..., 1::2,  ::2]
#         patch_top_right = x[...,  ::2, 1::2]
#         patch_bot_right = x[..., 1::2, 1::2]
#         x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,)
#         return self.conv(x)

class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu", conv_cls=None):
        super().__init__()
        if conv_cls is None:
            conv_cls = BaseConv
        self.conv = conv_cls(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left  = x[...,  ::2,  ::2]
        patch_bot_left  = x[..., 1::2,  ::2]
        patch_top_right = x[...,  ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return self.conv(x)

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, padding=0,
                 groups=1, bias=False, act="silu"):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=ksize, stride=stride, padding=padding,
            groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DirectionalPhotometricConv(nn.Module):
    """
    Directional photometric convolution used by DPRE.

    The four asymmetric branches emphasize local horizontal and vertical
    photometric residuals around small infrared targets. The interface matches
    BaseConv so it can be inserted into YOLO-style shallow stages.
    """
    def __init__(self, in_channels, out_channels, ksize, stride,
                 groups=1, bias=False, act="silu"):
        super().__init__()

        assert groups == 1, "DirectionalPhotometricConv first version only supports groups=1."
        assert isinstance(ksize, int), "DirectionalPhotometricConv expects integer kernel size."
        assert ksize == 3, "For first version, only ksize=3 is supported."
        assert out_channels % 4 == 0, "out_channels must be divisible by 4 for DPRE."

        k = ksize

        # asymmetric padding from official APConv.py
        pads = [
            (k, 0, 1, 0),  # left, right, top, bottom
            (0, k, 0, 1),
            (0, 1, k, 0),
            (1, 0, 0, k),
        ]
        self.pad = nn.ModuleList([nn.ZeroPad2d(p) for p in pads])

        # horizontal / vertical branches
        self.cw = ConvBNAct(in_channels, out_channels // 4, (1, k), stride, padding=0, act=act)
        self.ch = ConvBNAct(in_channels, out_channels // 4, (k, 1), stride, padding=0, act=act)

        # final fusion conv
        self.cat = ConvBNAct(out_channels, out_channels, 2, 1, padding=0, act=act)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        y = torch.cat([yw0, yw1, yh0, yh1], dim=1)
        return self.cat(y)

    def fuseforward(self, x):
        return self.forward(x)


class DPREBlock(nn.Module):
    """Residual DPRE block: x + alpha * directional photometric response."""
    def __init__(self, channels, alpha=0.1, act="silu"):
        super().__init__()
        self.alpha = alpha
        self.pconv = DirectionalPhotometricConv(channels, channels, ksize=3, stride=1, act=act)

    def forward(self, x):
        return x + self.alpha * self.pconv(x)

# class DPREBlock(nn.Module):
#     def __init__(self, channels, init_alpha=0.1, act="silu"):
#         super().__init__()
#         self.pconv = DirectionalPhotometricConv(channels, channels, ksize=3, stride=1, act=act)
#
#         init_alpha = float(init_alpha)
#         init_alpha = min(max(init_alpha, 1e-4), 1 - 1e-4)
#         init_logit = torch.log(torch.tensor(init_alpha / (1.0 - init_alpha), dtype=torch.float32))
#         self.alpha_logit = nn.Parameter(init_logit)
#
#     def forward(self, x):
#         alpha = torch.sigmoid(self.alpha_logit)
#         return x + alpha * self.pconv(x)

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act,)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1      = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m          = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels  = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2      = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x

#--------------------------------------------------#
#--------------------------------------------------#
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        #--------------------------------------------------#
        #--------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #--------------------------------------------------#
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  
        #--------------------------------------------------#
        self.conv1  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #--------------------------------------------------#
        #--------------------------------------------------#
        self.conv2  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #-----------------------------------------------#
        self.conv3  = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)

        #--------------------------------------------------#
        #--------------------------------------------------#
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        self.m      = nn.Sequential(*module_list)

    def forward(self, x):
        #-------------------------------#
        x_1 = self.conv1(x)
        #-------------------------------#
        #-------------------------------#
        x_2 = self.conv2(x)

        #-----------------------------------------------#
        #-----------------------------------------------#
        x_1 = self.m(x_1)
        #-----------------------------------------------#
        #-----------------------------------------------#
        x = torch.cat((x_1, x_2), dim=1)
        #-----------------------------------------------#
        return self.conv3(x)

class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"), depthwise=False, act="silu",):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        #-----------------------------------------------#
        #-----------------------------------------------#
        base_channels   = int(wid_mul * 64)  # 64
        base_depth      = max(round(dep_mul * 3), 1)  # 3
        
        #-----------------------------------------------#
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        #-----------------------------------------------#
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        #-----------------------------------------------#
        #-----------------------------------------------#
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
        )

        #-----------------------------------------------#
        #-----------------------------------------------#
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        #-----------------------------------------------#
        #-----------------------------------------------#
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        #-----------------------------------------------#
        #-----------------------------------------------#
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x


        x = self.dark2(x)
        outputs["dark2"] = x

        #-----------------------------------------------#
        #-----------------------------------------------#
        x = self.dark3(x)
        outputs["dark3"] = x
        #-----------------------------------------------#
        #-----------------------------------------------#
        x = self.dark4(x)
        outputs["dark4"] = x
        #-----------------------------------------------#
        #-----------------------------------------------#
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

#
# class CSPDarknet_PConv(nn.Module):
#     def __init__(
#         self,
#         dep_mul,
#         wid_mul,
#         out_features=("dark3", "dark4", "dark5"),
#         depthwise=False,
#         act="silu",
#     ):
#         super().__init__()
#         assert out_features, "please provide output features of Darknet"
#         self.out_features = out_features
#         pconv_stages = set(pconv_stages)
#
#         Conv = DWConv if depthwise else BaseConv
#
#         base_channels = int(wid_mul * 64)
#         base_depth = max(round(dep_mul * 3), 1)
#
#         # stem
#         stem_conv_cls = DirectionalPhotometricConv if "stem" in pconv_stages else BaseConv
#         self.stem = Focus(3, base_channels, ksize=3, act=act, conv_cls=stem_conv_cls)
#
#         # dark2
#         dark2_conv = DirectionalPhotometricConv if "dark2" in pconv_stages else Conv
#         self.dark2 = nn.Sequential(
#             dark2_conv(base_channels, base_channels * 2, 3, 2, act=act),
#             CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
#         )
#
#         # dark3
#         dark3_conv = DirectionalPhotometricConv if "dark3" in pconv_stages else Conv
#         self.dark3 = nn.Sequential(
#             dark3_conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
#             CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
#         )
#
#         # dark4
#         dark4_conv = DirectionalPhotometricConv if "dark4" in pconv_stages else Conv
#         self.dark4 = nn.Sequential(
#             dark4_conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
#             CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
#         )
#
#         # dark5
#         dark5_conv = DirectionalPhotometricConv if "dark5" in pconv_stages else Conv
#         self.dark5 = nn.Sequential(
#             dark5_conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
#             SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
#             CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act),
#         )
#     def forward(self, x):
#         outputs = {}
#         x = self.stem(x)
#         outputs["stem"] = x
#
#
#         x = self.dark2(x)
#         outputs["dark2"] = x
#
#         #-----------------------------------------------#
#         #-----------------------------------------------#
#         x = self.dark3(x)
#         outputs["dark3"] = x
#         #-----------------------------------------------#
#         #-----------------------------------------------#
#         x = self.dark4(x)
#         outputs["dark4"] = x
#         #-----------------------------------------------#
#         #-----------------------------------------------#
#         x = self.dark5(x)
#         outputs["dark5"] = x
#         return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknetDPRE(nn.Module):
    """CSPDarknet backbone with optional DPRE blocks after selected stages."""
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
        enhance_stages=(),
        enhance_alpha=0.1,
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        enhance_stages = set(enhance_stages)

        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        self.stem = Focus(3, base_channels, ksize=3, act=act)

        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
        )

        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act),
        )

        # Apply DPRE after selected shallow backbone stages.
        self.enhance_stem  = DPREBlock(base_channels, enhance_alpha, act) if "stem"  in enhance_stages else nn.Identity()
        self.enhance_dark2 = DPREBlock(base_channels * 2, enhance_alpha, act) if "dark2" in enhance_stages else nn.Identity()
        self.enhance_dark3 = DPREBlock(base_channels * 4, enhance_alpha, act) if "dark3" in enhance_stages else nn.Identity()
        self.enhance_dark4 = DPREBlock(base_channels * 8, enhance_alpha, act) if "dark4" in enhance_stages else nn.Identity()
        self.enhance_dark5 = DPREBlock(base_channels * 16, enhance_alpha, act) if "dark5" in enhance_stages else nn.Identity()

    def forward(self, x):
        outputs = {}

        x = self.stem(x)
        x = self.enhance_stem(x)
        outputs["stem"] = x

        x = self.dark2(x)
        x = self.enhance_dark2(x)
        outputs["dark2"] = x

        x = self.dark3(x)
        x = self.enhance_dark3(x)
        outputs["dark3"] = x

        x = self.dark4(x)
        x = self.enhance_dark4(x)
        outputs["dark4"] = x

        x = self.dark5(x)
        x = self.enhance_dark5(x)
        outputs["dark5"] = x

        return {k: v for k, v in outputs.items() if k in self.out_features}

