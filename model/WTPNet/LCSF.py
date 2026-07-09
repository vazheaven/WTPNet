import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.WTPNet.DPRE import BaseConv, DWConv


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5,
                 depthwise=False, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class FusionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=0.5,
                 depthwise=False, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(hidden_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(hidden_channels, out_channels, 1, stride=1, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


class ConvBN(nn.Sequential):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        layers = [
            nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, groups=g, bias=False),
            nn.BatchNorm2d(c2),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class DynamicSKA(nn.Module):
    """
    Small-kernel aggregation with dynamic weights.
    x: [B, C, H, W]
    w: [B, G, K*K, H, W], where G = number of dynamic groups and each group shares one KxK kernel.
    """
    def __init__(self, weight_act="softmax"):
        super().__init__()
        assert weight_act in ["softmax", "sigmoid", "none"]
        self.weight_act = weight_act

    def forward(self, x, w):
        b, c, h, width = x.shape
        _, g, kk, _, _ = w.shape
        k = int(math.sqrt(kk))
        assert k * k == kk, f"Invalid kernel size^2: {kk}"
        assert c % g == 0, f"channels ({c}) must be divisible by dynamic groups ({g})"
        group_size = c // g

        # [B, C*K*K, H*W] -> [B, C, K*K, H, W]
        x_unfold = F.unfold(x, kernel_size=k, padding=k // 2)
        x_unfold = x_unfold.view(b, c, kk, h, width)
        x_unfold = x_unfold.view(b, g, group_size, kk, h, width)

        if self.weight_act == "softmax":
            w = w.softmax(dim=2)
        elif self.weight_act == "sigmoid":
            w = w.sigmoid()

        out = (x_unfold * w.unsqueeze(2)).sum(dim=3)
        out = out.reshape(b, c, h, width)
        return out


class LKP(nn.Module):
    """
    Large-kernel perception block from LSNet idea.
    group_size: number of channels sharing one dynamic KxK kernel.
    """
    def __init__(self, dim, lks=7, sks=3, group_size=8):
        super().__init__()
        assert dim % group_size == 0, f"dim={dim} must be divisible by group_size={group_size}"
        num_dyn_groups = dim // group_size

        self.cv1 = ConvBN(dim, dim // 2, k=1, s=1, p=0, act=True)
        self.cv2 = ConvBN(dim // 2, dim // 2, k=lks, s=1, p=(lks - 1) // 2, g=dim // 2, act=True)
        self.cv3 = ConvBN(dim // 2, dim // 2, k=1, s=1, p=0, act=True)
        self.cv4 = nn.Conv2d(dim // 2, sks * sks * num_dyn_groups, kernel_size=1, stride=1, padding=0, bias=True)
        self.norm = nn.GroupNorm(num_groups=num_dyn_groups, num_channels=sks * sks * num_dyn_groups)

        self.sks = sks
        self.num_dyn_groups = num_dyn_groups

    def forward(self, x):
        x = self.cv3(self.cv2(self.cv1(x)))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.shape
        w = w.view(b, self.num_dyn_groups, self.sks * self.sks, h, width)
        return w


class LSBlockOfficial(nn.Module):
    """
    Closer to official LSNet idea: large-kernel perception + dynamic small-kernel aggregation.
    Heavier but more faithful to LS design.
    """
    def __init__(self, dim, lks=7, sks=3, group_size=8, weight_act="softmax"):
        super().__init__()
        self.lkp = LKP(dim, lks=lks, sks=sks, group_size=group_size)
        self.ska = DynamicSKA(weight_act=weight_act)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        out = self.ska(x, self.lkp(x))
        out = self.bn(out)
        return x + out


class LSBlockLite(nn.Module):
    """
    Lighter LS-inspired block:
    - large-kernel DW conv for context perception
    - small-kernel DW conv for local aggregation
    - sigmoid gate from large context to modulate local aggregation
    Much cheaper than official dynamic SKA.
    """
    def __init__(self, dim, lks=7, sks=3, act="silu"):
        super().__init__()
        self.large = nn.Conv2d(dim, dim, kernel_size=lks, stride=1,
                               padding=(lks - 1) // 2, groups=dim, bias=False)
        self.large_bn = nn.BatchNorm2d(dim)
        self.small = nn.Conv2d(dim, dim, kernel_size=sks, stride=1,
                               padding=(sks - 1) // 2, groups=dim, bias=False)
        self.small_bn = nn.BatchNorm2d(dim)
        self.proj = BaseConv(dim, dim, 1, 1, act=act)
        self.out_bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        ctx = self.large_bn(self.large(x))
        gate = torch.sigmoid(ctx)
        agg = self.small_bn(self.small(x)) * gate
        out = self.out_bn(self.proj(agg))
        return x + out


class IdentityBlock(nn.Module):
    def forward(self, x):
        return x


def build_ls_block(dim, ls_type="lite", lks=7, sks=3, group_size=8,
                   weight_act="softmax", act="silu"):
    if ls_type in [None, "none"]:
        return IdentityBlock()
    if ls_type == "lite":
        return LSBlockLite(dim, lks=lks, sks=sks, act=act)
    if ls_type == "official":
        return LSBlockOfficial(dim, lks=lks, sks=sks, group_size=group_size, weight_act=weight_act)
    raise ValueError(f"Unsupported ls_type: {ls_type}")


class LCSF(nn.Module):
    """
    Large-context scale-focused fusion for WTPNet.

    Inputs:
        feat1: [B, C1, 80, 80]
        feat2: [B, C2, 40, 40]
        feat3: [B, C3, 20, 20]
    Output:
        p3_out: [B, C1, 80, 80]

    Supported insertion points:
        - cat_p4 : after cat([upsample(feat3), feat2])  -> channel = C2 + C3
        - p4     : after reducing to P4                 -> channel = C2
        - cat_p3 : after cat([upsample(P4), feat1])     -> channel = C1 + C2
        - p3     : after reducing to P3_out             -> channel = C1

    Recommended starting configs:
        1) ls_type='lite',     ls_stages=('p3',)
        2) ls_type='lite',     ls_stages=('p4', 'p3')
        3) ls_type='official', ls_stages=('p3',)

    LCSF uses large-kernel perception before/after feature pyramid aggregation
    so compact target responses are not diluted by broad contextual semantics.
    """
    def __init__(
        self,
        in_channels=(128, 256, 512),
        depthwise=False,
        act="silu",
        ls_type="lite",                  # 'lite' | 'official' | 'none'
        ls_stages=("p3",),               # subset of {'cat_p4', 'p4', 'cat_p3', 'p3'}
        lks=7,
        # lks=11,
        sks=3,
        group_size=8,
        weight_act="softmax",
        use_c3_p4=False,
        use_c3_p3=False,
    ):
        super().__init__()
        c1, c2, c3 = in_channels
        ls_stages = set(ls_stages)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.use_c3_p4 = use_c3_p4
        self.use_c3_p3 = use_c3_p3

        # stage-1: feat3 up + feat2 -> P4
        self.ls_cat_p4 = build_ls_block(c2 + c3, ls_type, lks, sks, group_size, weight_act,
                                        act) if "cat_p4" in ls_stages else IdentityBlock()
        self.lateral_conv0 = BaseConv(c2 + c3, c2, 1, 1, act=act)
        self.c3_p4 = FusionLayer(c2 + c3, c2, depthwise=depthwise, act=act)
        self.ls_p4 = build_ls_block(c2, ls_type, lks, sks, group_size, weight_act,
                                    act) if "p4" in ls_stages else IdentityBlock()

        # stage-2: P4 up + feat1 -> P3_out
        self.ls_cat_p3 = build_ls_block(c1 + c2, ls_type, lks, sks, group_size, weight_act,
                                        act) if "cat_p3" in ls_stages else IdentityBlock()
        self.reduce_conv1 = BaseConv(c1 + c2, c1, 1, 1, act=act)
        self.c3_p3 = FusionLayer(c1 + c2, c1, depthwise=depthwise, act=act)
        self.ls_p3 = build_ls_block(c1, ls_type, lks, sks, group_size, weight_act,
                                    act) if "p3" in ls_stages else IdentityBlock()

    def forward(self, inputs):
        feat1, feat2, feat3 = inputs

        # 20 -> 40
        p5_up = self.upsample(feat3)
        p5_cat = torch.cat([p5_up, feat2], dim=1)
        p5_cat = self.ls_cat_p4(p5_cat)

        if self.use_c3_p4:
            p4 = self.c3_p4(p5_cat)
        else:
            p4 = self.lateral_conv0(p5_cat)
        p4 = self.ls_p4(p4)

        # 40 -> 80
        p4_up = self.upsample(p4)
        p4_cat = torch.cat([p4_up, feat1], dim=1)
        p4_cat = self.ls_cat_p3(p4_cat)

        if self.use_c3_p3:
            p3_out = self.c3_p3(p4_cat)
        else:
            p3_out = self.reduce_conv1(p4_cat)
        p3_out = self.ls_p3(p3_out)

        return p3_out


if __name__ == "__main__":
    # quick shape check
    fusion = LCSF(
        in_channels=(128, 256, 512),
        ls_type="lite",
        ls_stages=("p3",),
        use_c3_p4=False,
        use_c3_p3=False,
    )
    x1 = torch.randn(2, 128, 80, 80)
    x2 = torch.randn(2, 256, 40, 40)
    x3 = torch.randn(2, 512, 20, 20)
    y = fusion([x1, x2, x3])
    print("output shape:", y.shape)

