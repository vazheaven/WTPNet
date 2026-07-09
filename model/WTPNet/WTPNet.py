import torch
import torch.nn as nn

from model.WTPNet.attention import CrossAttention, SelfAttention
# from model.WTPNet.attention_freq import CrossAttention, SelfAttention
# from model.WTPNet.freq_prior import FrequencyPriorGenerator2D
# from model.WTPNet.freq_prior_fft import FFTPriorGenerator3D

from model.WTPNet.backbone3d import Backbone3D
from model.WTPNet.motion_backbone import BackboneTD
from model.WTPNet.WTCP import (
    WTCPMotionPrior,
    WTCPContextPrior,
    WTCPFusion,
    WTCPPyramid,
    WTCPQInjector,
    WTCPQKVetoInjector,
    WTCPKInjector,
    WTCPMotionOnlyInjector
)
from model.WTPNet.DPRE import BaseConv, CSPDarknet, DWConv, CSPDarknetDPRE
from model.WTPNet.LCSF import LCSF

# class Feature_Backbone(nn.Module):
#     def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"), in_channels=[256, 512, 1024], depthwise=False, act="silu"):
#         super().__init__()
#         self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
#         self.in_features = in_features
#
#     def forward(self, input):
#         out_features = self.backbone.forward(input)
#         [feat1, feat2, feat3] = [out_features[f] for f in self.in_features]
#         return [feat1, feat2, feat3]

class Feature_Backbone(nn.Module):
    """Current-frame spatial branch with DPRE inserted into shallow YOLO-style stages."""
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"),
                 in_channels=[256, 512, 1024], depthwise=False, act="silu",
                 enhance_stages=(), enhance_alpha=0.1):
        super().__init__()
        self.backbone = CSPDarknetDPRE(
            depth, width,
            depthwise=depthwise,
            act=act,
            enhance_stages=enhance_stages,
            enhance_alpha=enhance_alpha
        )
        self.in_features = in_features
    def forward(self, input):
        out_features = self.backbone.forward(input)
        [feat1, feat2, feat3] = [out_features[f] for f in self.in_features]
        return [feat1, feat2, feat3]


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class FusionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=0.5, depthwise=False, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(hidden_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(hidden_channels, out_channels, 1, stride=1, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


class Feature_Fusion(nn.Module):
    def __init__(self, in_channels=[128, 256, 512], depthwise=False, act="silu"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(in_channels[1] + in_channels[2], in_channels[1], 1, 1, act=act)
        self.C3_p4 = FusionLayer(
            int(2 * in_channels[1]),
            int(in_channels[1]),
            depthwise=depthwise,
            act=act,
        )
        self.reduce_conv1 = BaseConv(int(in_channels[0] + in_channels[1]), int(in_channels[0]), 1, 1, act=act)
        self.C3_p3 = FusionLayer(
            int(2 * in_channels[0]),
            int(in_channels[0]),
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        out_features = input
        [feat1, feat2, feat3] = out_features

        P5_upsample = self.upsample(feat3)
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        P4 = self.lateral_conv0(P5_upsample)

        P4_upsample = self.upsample(P4)
        P4_upsample = torch.cat([P4_upsample, feat1], 1)
        P3_out = self.reduce_conv1(P4_upsample)

        return P3_out


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, in_channels=[16, 32, 64], act="silu"):
        super().__init__()
        Conv = BaseConv

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(in_channels=int(in_channels[i]), out_channels=int(256 * width), ksize=1, stride=1, act=act)
            )
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            )

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, inputs):
        outputs = []
        for k, x in enumerate(inputs):
            x = self.stems[k](x)
            cls_feat = self.cls_convs[k](x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = self.reg_convs[k](x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs


model_config = {
    'backbone_2d': 'yolo_free_nano',
    'pretrained_2d': True,
    'stride': [8, 16, 32],
    'backbone_3d': 'shufflenetv2',
    'backbone_td': 'shufflenetv2',
    'model_size': '1.0x',
    'pretrained_3d': True,
    'memory_momentum': 0.9,
    'head_dim': 128,
    'head_norm': 'BN',
    'head_act': 'lrelu',
    'num_cls_heads': 2,
    'num_reg_heads': 2,
    'head_depthwise': True,
}


def build_backbone_3d(cfg, pretrained=False):
    backbone = Backbone3D(cfg, pretrained)
    return backbone, backbone.feat_dim


mcfg = model_config

class WTPNet(nn.Module):
    """
    WTPNet has three feature sources:
        1. motion branch: aligned temporal-difference frames -> Q features
        2. context branch: raw temporal frames -> K features
        3. current-frame branch: last raw frame -> V features

    WTCP generates prior maps from shallow temporal features and injects them
    into selected Q/K stages. DPRE is used inside the 2D current-frame branch.
    LCSF performs the final scale-focused multi-level fusion.

    wtcp_mode:
        'q_only'   -> inject WTCP into the Q/motion branch only
        'k_only'   -> inject WTCP into the K/context branch only
        'q_kveto'  -> Q prior injection with K-branch veto
        'qk_both'  -> prior injection on both Q and K branches
        'q_motion' -> motion-only Q prior injection
    """
    def __init__(self, num_classes, fp16=False, num_frame=5,
                 use_wtcp=True,
                 # use_wtcp=False,
                 wtcp_inject_stages=(True, False, False),
                 wtcp_mode='k_only'):
        super(WTPNet, self).__init__()
        assert wtcp_mode in ['q_only', 'k_only','q_kveto', 'qk_both', 'q_motion']
        self.use_wtcp = use_wtcp
        self.wtcp_inject_stages = wtcp_inject_stages
        self.wtcp_mode = wtcp_mode

        # DPRE preserves weak shallow photometric responses in the current frame.
        self.backbone2d = Feature_Backbone(
            0.33, 0.50, enhance_stages=("dark2", "dark3"), enhance_alpha=0.1)

        # Q branch uses TDC/RepConv3D on aligned motion frames; K branch uses
        # regular 3D convolution on raw temporal frames.
        self.backbone3d, _ = build_backbone_3d(mcfg, pretrained=mcfg['pretrained_3d'] and True)
        self.backbonetd = BackboneTD(mcfg, pretrained=mcfg['pretrained_3d'] and True)

        if self.use_wtcp:
            # WTCP combines temporal DCT motion cues with wavelet context cues.
            self.motion_prior = WTCPMotionPrior(in_channels=24, hidden=8, prior_channels=16)
            self.context_prior = WTCPContextPrior(in_channels=24, hidden=8, prior_channels=16)
            self.shared_prior = WTCPFusion(in_channels=16, out_channels=16)

            self.pyramid_q = WTCPPyramid(channels=16)
            self.pyramid_k = WTCPPyramid(channels=16)
            self.pyramid_g = WTCPPyramid(channels=16)

            if wtcp_mode == 'q_only':
                self.q_prior2 = WTCPQInjector(128, prior_channels=16) if wtcp_inject_stages[0] else None
                self.q_prior3 = WTCPQInjector(256, prior_channels=16) if wtcp_inject_stages[1] else None
                self.q_prior4 = WTCPQInjector(512, prior_channels=16) if wtcp_inject_stages[2] else None
                self.k_prior2 = self.k_prior3 = self.k_prior4 = None

            elif wtcp_mode == 'k_only':
                self.k_prior2 = WTCPKInjector(128, prior_channels=16) if wtcp_inject_stages[0] else None
                self.k_prior3 = WTCPKInjector(256, prior_channels=16) if wtcp_inject_stages[1] else None
                self.k_prior4 = WTCPKInjector(512, prior_channels=16) if wtcp_inject_stages[2] else None
                self.q_prior2 = self.q_prior3 = self.q_prior4 = None

            elif wtcp_mode == 'q_kveto':
                self.q_prior2 = WTCPQKVetoInjector(128, prior_channels=16) if wtcp_inject_stages[0] else None
                self.q_prior3 = WTCPQKVetoInjector(256, prior_channels=16) if wtcp_inject_stages[1] else None
                self.q_prior4 = WTCPQKVetoInjector(512, prior_channels=16) if wtcp_inject_stages[2] else None
                self.k_prior2 = self.k_prior3 = self.k_prior4 = None

            elif wtcp_mode == 'q_motion':
                self.q_prior2 = WTCPMotionOnlyInjector(128, prior_channels=16) if wtcp_inject_stages[0] else None
                self.q_prior3 = WTCPMotionOnlyInjector(256, prior_channels=16) if wtcp_inject_stages[1] else None
                self.q_prior4 = None

            else:  # qk_both
                self.q_prior2 = WTCPQInjector(128, prior_channels=16) if wtcp_inject_stages[0] else None
                self.q_prior3 = WTCPQInjector(256, prior_channels=16) if wtcp_inject_stages[1] else None
                self.q_prior4 = WTCPQInjector(512, prior_channels=16) if wtcp_inject_stages[2] else None
                self.k_prior2 = WTCPKInjector(128, prior_channels=16) if wtcp_inject_stages[0] else None
                self.k_prior3 = WTCPKInjector(256, prior_channels=16) if wtcp_inject_stages[1] else None
                self.k_prior4 = WTCPKInjector(512, prior_channels=16) if wtcp_inject_stages[2] else None
        else:
            self.motion_prior = self.context_prior = self.shared_prior = None
            self.pyramid_q = self.pyramid_k = self.pyramid_g = None
            self.q_prior2 = self.q_prior3 = self.q_prior4 = None
            self.k_prior2 = self.k_prior3 = self.k_prior4 = None

        self.q_sa1 = SelfAttention(128, window_size=(2, 8, 8), num_heads=4, use_shift=True, mlp_ratio=1.5)
        self.k_sa1 = SelfAttention(128, window_size=(2, 8, 8), num_heads=4, use_shift=True, mlp_ratio=1.5)
        self.v_sa1 = SelfAttention(128, window_size=(2, 8, 8), num_heads=4, use_shift=True, mlp_ratio=1.5)

        self.q_sa2 = SelfAttention(256, window_size=(2, 4, 4), num_heads=4, use_shift=True, mlp_ratio=1.5)
        self.k_sa2 = SelfAttention(256, window_size=(2, 4, 4), num_heads=4, use_shift=True, mlp_ratio=1.5)
        self.v_sa2 = SelfAttention(256, window_size=(2, 4, 4), num_heads=4, use_shift=True, mlp_ratio=1.5)

        self.q_sa3 = SelfAttention(512, window_size=(2, 2, 2), num_heads=4, use_shift=True, mlp_ratio=1.5)
        self.k_sa3 = SelfAttention(512, window_size=(2, 2, 2), num_heads=4, use_shift=True, mlp_ratio=1.5)
        self.v_sa3 = SelfAttention(512, window_size=(2, 2, 2), num_heads=4, use_shift=True, mlp_ratio=1.5)

        self.ca1 = CrossAttention(128, window_size=(2, 8, 8), num_heads=4)
        self.ca2 = CrossAttention(256, window_size=(2, 4, 4), num_heads=4)
        self.ca3 = CrossAttention(512, window_size=(2, 2, 2), num_heads=4)

        # LCSF refocuses cross-attention outputs before the YOLOX-style head.
        self.feature_fusion = LCSF(
            in_channels=(128, 256, 512),
            ls_type="official",
            ls_stages=("p3", "p4"),
            use_c3_p4=True,
            use_c3_p3=True,
            act="silu"
        )
        self.head = YOLOXHead(num_classes=num_classes, width=1.0, in_channels=[128], act="silu")

    def forward(self, inputs):
        # inputs: [B, 3, 2T, H, W], organized as aligned frames followed by raw frames.
        if len(inputs.shape) == 5:
            T = inputs.shape[2]
            diff_imgs = inputs[:, :, :T // 2, :, :]
            mt_imgs = inputs[:, :, T // 2:, :, :]
        else:
            diff_imgs = inputs
            mt_imgs = inputs

        # Motion-sensitive Q features.
        q_3d = self.backbonetd(diff_imgs)
        q_stem = q_3d['stem']
        q_3d1, q_3d2, q_3d3 = q_3d['stage2'], q_3d['stage3'], q_3d['stage4']

        # Context-aware K features.
        k_3d = self.backbone3d(mt_imgs)
        k_stem = k_3d['stem']
        k_3d1, k_3d2, k_3d3 = k_3d['stage2'], k_3d['stage3'], k_3d['stage4']

        if self.use_wtcp:
            p_q0 = self.motion_prior(q_stem)
            p_k0 = self.context_prior(k_stem)
            p_g0 = self.shared_prior(p_q0, p_k0)

            p_q2, p_q3, p_q4 = self.pyramid_q(p_q0)
            p_k2, p_k3, p_k4 = self.pyramid_k(p_k0)
            p_g2, p_g3, p_g4 = self.pyramid_g(p_g0)

            if self.wtcp_mode == 'q_only':
                if self.q_prior2 is not None:
                    q_3d1 = self.q_prior2(q_3d1, p_q2, p_g2)
                if self.q_prior3 is not None:
                    q_3d2 = self.q_prior3(q_3d2, p_q3, p_g3)
                if self.q_prior4 is not None:
                    q_3d3 = self.q_prior4(q_3d3, p_q4, p_g4)

            if self.wtcp_mode == 'k_only':
                if self.k_prior2 is not None:
                    k_3d1 = self.k_prior2(k_3d1, p_k2, p_g2)
                    # k_3d1 = self.k_prior2(k_3d1, p_k2, p_q2)
                    # k_3d1 = self.k_prior2(k_3d1, p_g2, p_g2)
                    # k_3d1 = self.k_prior2(k_3d1, p_g2)
                if self.k_prior3 is not None:
                    k_3d2 = self.k_prior3(k_3d2, p_k3, p_g3)
                if self.k_prior4 is not None:
                    k_3d3 = self.k_prior4(k_3d3, p_k4, p_g4)

            elif self.wtcp_mode == 'q_kveto':
                if self.q_prior2 is not None:
                    q_3d1 = self.q_prior2(q_3d1, p_q2, p_k2, p_g2)
                if self.q_prior3 is not None:
                    q_3d2 = self.q_prior3(q_3d2, p_q3, p_k3, p_g3)
                if self.q_prior4 is not None:
                    q_3d3 = self.q_prior4(q_3d3, p_q4, p_k4, p_g4)

            elif self.wtcp_mode == 'q_motion':
                if self.q_prior2 is not None:
                    q_3d1 = self.q_prior2(q_3d1, p_q2)
                if self.q_prior3 is not None:
                    q_3d2 = self.q_prior3(q_3d2, p_q3)


            else:  # qk_both
                if self.q_prior2 is not None:
                    q_3d1 = self.q_prior2(q_3d1, p_q2, p_g2)
                if self.q_prior3 is not None:
                    q_3d2 = self.q_prior3(q_3d2, p_q3, p_g3)
                if self.q_prior4 is not None:
                    q_3d3 = self.q_prior4(q_3d3, p_q4, p_g4)

                if self.k_prior2 is not None:
                    k_3d1 = self.k_prior2(k_3d1, p_k2, p_g2)
                if self.k_prior3 is not None:
                    k_3d2 = self.k_prior3(k_3d2, p_k3, p_g3)
                if self.k_prior4 is not None:
                    k_3d3 = self.k_prior4(k_3d3, p_k4, p_g4)

        # Current-frame V features.
        [feat1, feat2, feat3] = self.backbone2d(inputs[:, :, -1, :, :])

        def to_5d(x):
            return x.permute(0, 2, 3, 4, 1)

        q_3d1 = to_5d(q_3d1)
        q_3d2 = to_5d(q_3d2)
        q_3d3 = to_5d(q_3d3)

        k_3d1 = to_5d(k_3d1)
        k_3d2 = to_5d(k_3d2)
        k_3d3 = to_5d(k_3d3)

        def expand_v(x, T):
            x = x.permute(0, 2, 3, 1).unsqueeze(1)
            x = x.expand(-1, T, -1, -1, -1)
            return x

        T1 = q_3d1.shape[1]
        T2 = q_3d2.shape[1]
        T3 = q_3d3.shape[1]
        v1 = expand_v(feat1, T1)
        v2 = expand_v(feat2, T2)
        v3 = expand_v(feat3, T3)

        q1 = self.q_sa1(q_3d1)
        k1 = self.k_sa1(k_3d1)
        v1 = self.v_sa1(v1)

        q2 = self.q_sa2(q_3d2)
        k2 = self.k_sa2(k_3d2)
        v2 = self.v_sa2(v2)

        q3 = self.q_sa3(q_3d3)
        k3 = self.k_sa3(k_3d3)
        v3 = self.v_sa3(v3)

        out1 = self.ca1(q1, k1, v1)
        out2 = self.ca2(q2, k2, v2)
        out3 = self.ca3(q3, k3, v3)

        out1 = out1.mean(1).permute(0, 3, 1, 2)
        out2 = out2.mean(1).permute(0, 3, 1, 2)
        out3 = out3.mean(1).permute(0, 3, 1, 2)

        feat_all = self.feature_fusion([out1, out2, out3])
        outputs = self.head([feat_all])
        return outputs

