import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DCTMixin:
    """Shared temporal-DCT utilities used by WTCP prior generators."""

    @staticmethod
    def _build_dct_basis(T, device, dtype):
        n = torch.arange(T, device=device, dtype=dtype).view(1, T)
        k = torch.arange(T, device=device, dtype=dtype).view(T, 1)
        basis = torch.cos(math.pi / T * (n + 0.5) * k)
        basis[0] *= math.sqrt(1.0 / T)
        if T > 1:
            basis[1:] *= math.sqrt(2.0 / T)
        return basis

    def _temporal_dct(self, x):
        """Apply DCT along the temporal dimension of a [B, T, H, W] tensor."""
        basis = self._build_dct_basis(x.shape[1], x.device, x.dtype)
        return torch.einsum("bthw,ft->bfhw", x, basis)

    @staticmethod
    def _split_energy(coeff):
        """
        Split DCT coefficients into DC, low-frequency, and high-frequency energy.
        WTCP uses the ratios between these bands to distinguish stable background
        variation from motion-sensitive temporal responses.
        """
        E_dc = coeff[:, 0:1].pow(2)
        T = coeff.shape[1]

        if T >= 3:
            E_lf = coeff[:, 1:min(3, T)].pow(2).mean(dim=1, keepdim=True)
        elif T == 2:
            E_lf = coeff[:, 1:2].pow(2)
        else:
            E_lf = torch.zeros_like(E_dc)

        if T >= 4:
            E_hf = coeff[:, 3:].pow(2).mean(dim=1, keepdim=True)
        else:
            E_hf = torch.zeros_like(E_dc)
        return E_dc, E_lf, E_hf


class WTCPMotionPrior(nn.Module, _DCTMixin):
    """
    Motion-side WTCP prior for the Q branch.

    Input:  shallow aligned motion features [B, C, T, H, W]
    Output: motion prior map [B, prior_channels, H, W]
    """

    def __init__(self, in_channels=24, hidden=8, prior_channels=16, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.squeeze = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, prior_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(prior_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        z = self.squeeze(x).mean(dim=1)
        coeff = self._temporal_dct(z)
        E_dc, E_lf, E_hf = self._split_energy(coeff)
        R_hf = E_hf / (E_dc + self.eps)
        R_lf = E_lf / (E_dc + self.eps)
        feat = torch.cat([E_lf, E_hf, R_hf, R_lf], dim=1)
        return self.head(feat)


class WTCPContextPrior(nn.Module, _DCTMixin):
    """
    Wavelet-guided context prior for the K branch.

    The key frame and previous temporal context are decomposed into Haar
    subbands. Low-frequency dominance describes stable background context,
    while high-frequency contrast and novelty highlight target-like local
    changes that should guide temporal context aggregation.
    """

    def __init__(self, in_channels=24, hidden=8, prior_channels=16, eps=1e-6):
        super().__init__()
        self.eps = eps

        self.squeeze = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden),
            nn.SiLU(inplace=True),
        )

        # Channel-wise scorer for adaptively combining LH/HL/HH detail bands.
        self.hf_score = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
        )

        # 7 groups: LL_ref, HF_ref, HF_key, key_contrast, low_dominance,
        # hf_novelty, key_ref_diff.
        self.head = nn.Sequential(
            nn.Conv2d(hidden * 7, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, prior_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(prior_channels),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _haar2d(x):
        """
        Haar decomposition for [B, C, H, W].

        Returns LL, LH, HL, HH at half spatial resolution. Odd borders are
        cropped because the downstream prior is interpolated back to the
        original feature size.
        """
        _, _, H, W = x.shape
        H2 = H // 2 * 2
        W2 = W // 2 * 2
        x = x[:, :, :H2, :W2]

        a = x[:, :, 0::2, 0::2]
        b = x[:, :, 0::2, 1::2]
        c = x[:, :, 1::2, 0::2]
        d = x[:, :, 1::2, 1::2]

        LL = (a + b + c + d) * 0.5
        LH = (a - b + c - d) * 0.5
        HL = (a + b - c - d) * 0.5
        HH = (a - b - c + d) * 0.5
        return LL, LH, HL, HH

    def _adaptive_hf_fusion(self, LH, HL, HH):
        """Fuse horizontal, vertical, and diagonal high-frequency subbands."""
        bands = torch.stack([LH.abs(), HL.abs(), HH.abs()], dim=1)
        B, N, C, h, w = bands.shape

        desc = bands.reshape(B * N, C, h, w).mean(dim=(2, 3), keepdim=True)
        score = self.hf_score(desc).reshape(B, N, C, 1, 1)
        weight = torch.softmax(score, dim=1)
        return (weight * bands).sum(dim=1)

    def forward(self, x):
        z = self.squeeze(x)
        _, _, T, H, W = z.shape

        # The last frame is the key frame; previous frames define context.
        z_key = z[:, :, -1]
        if T > 1:
            z_ref = z[:, :, :-1].mean(dim=2)
        else:
            z_ref = z_key

        z_diff = (z_key - z_ref).abs()
        LL_ref, LH_ref, HL_ref, HH_ref = self._haar2d(z_ref)
        LL_key, LH_key, HL_key, HH_key = self._haar2d(z_key)

        HF_ref = self._adaptive_hf_fusion(LH_ref, HL_ref, HH_ref)
        HF_key = self._adaptive_hf_fusion(LH_key, HL_key, HH_key)

        key_contrast = torch.log1p(HF_key / (LL_key.abs() + self.eps))
        low_dominance = torch.log1p(LL_ref.abs() / (HF_ref + self.eps))
        hf_novelty = torch.log1p((HF_key - HF_ref).abs() / (HF_ref + self.eps))
        key_ref_diff = F.avg_pool2d(z_diff, kernel_size=2, stride=2)

        feat = torch.cat(
            [LL_ref, HF_ref, HF_key, key_contrast, low_dominance, hf_novelty, key_ref_diff],
            dim=1,
        )
        p = self.head(feat)
        return F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)


class WTCPFusion(nn.Module):
    """Fuse motion and context priors into a shared WTCP guidance map."""

    def __init__(self, in_channels=16, out_channels=16):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, p_q, p_k):
        return self.fuse(torch.cat([p_q, p_k], dim=1))


class _DownBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class WTCPPyramid(nn.Module):
    """
    Build WTCP prior maps at the feature scales used by the 3D backbones:
    stage2 (/8), stage3 (/16), and stage4 (/32).
    """

    def __init__(self, channels=16):
        super().__init__()
        self.ds1 = _DownBlock(channels)
        self.ds2 = _DownBlock(channels)
        self.ds3 = _DownBlock(channels)
        self.ds4 = _DownBlock(channels)

    def forward(self, p0):
        p1 = self.ds1(p0)
        p2 = self.ds2(p1)
        p3 = self.ds3(p2)
        p4 = self.ds4(p3)
        return p2, p3, p4


class WTCPQInjector(nn.Module):
    """Add WTCP prior information to Q/motion features."""

    def __init__(self, feat_channels, prior_channels=16):
        super().__init__()
        self.map = nn.Sequential(
            nn.Conv2d(prior_channels * 2, feat_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.SiLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, q, p_q, p_g):
        prior = self.map(torch.cat([p_q, p_g], dim=1)).unsqueeze(2)
        return q + self.alpha * prior


class _PriorMap(nn.Module):
    """Resize and project one or more WTCP prior maps to a feature tensor."""

    def __init__(self, out_channels, in_priors):
        super().__init__()
        self.map = nn.Sequential(
            nn.Conv2d(in_priors, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, *priors, target_hw=None):
        xs = []
        for p in priors:
            if target_hw is not None and p.shape[-2:] != target_hw:
                p = F.interpolate(p, size=target_hw, mode="bilinear", align_corners=False)
            xs.append(p)
        return self.map(torch.cat(xs, dim=1))


class WTCPQKVetoInjector(nn.Module):
    """
    Q-branch prior injection gated by K-branch context.

    This variant lets the context prior suppress motion responses that are
    inconsistent with the temporal context branch.
    """

    def __init__(self, feat_channels, prior_channels=16):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.proposal = _PriorMap(out_channels=feat_channels, in_priors=prior_channels * 2)
        self.veto = nn.Sequential(
            nn.Conv2d(prior_channels * 2, feat_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, q, p_q, p_k, p_g):
        _, _, _, H, W = q.shape
        proposal = self.proposal(p_q, p_g, target_hw=(H, W))

        if p_k.shape[-2:] != (H, W):
            p_k = F.interpolate(p_k, size=(H, W), mode="bilinear", align_corners=False)
        if p_g.shape[-2:] != (H, W):
            p_g = F.interpolate(p_g, size=(H, W), mode="bilinear", align_corners=False)

        veto = self.veto(torch.cat([p_k, p_g], dim=1))
        return q + self.alpha * (proposal * veto).unsqueeze(2)


class WTCPKInjector(nn.Module):
    """
    Add WTCP context prior to K features with a bounded residual scale.

    The zero-initialized beta keeps the pretrained temporal backbone behavior
    unchanged at initialization, then learns how strongly to use WTCP guidance.
    """

    def __init__(self, feat_channels, prior_channels=16, max_scale=0.05):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.max_scale = max_scale
        mid_channels = max(16, min(64, feat_channels // 4))

        self.delta = nn.Sequential(
            nn.Conv2d(prior_channels * 2, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, feat_channels, kernel_size=1, bias=True),
        )

    def forward(self, k, p_k, p_g):
        _, _, _, H, W = k.shape
        if p_k.shape[-2:] != (H, W):
            p_k = F.interpolate(p_k, size=(H, W), mode="bilinear", align_corners=False)
        if p_g.shape[-2:] != (H, W):
            p_g = F.interpolate(p_g, size=(H, W), mode="bilinear", align_corners=False)

        delta = self.delta(torch.cat([p_k, p_g], dim=1)).unsqueeze(2)
        scale = self.max_scale * torch.tanh(self.beta)
        return k + scale * delta


class WTCPMotionOnlyInjector(nn.Module):
    """Ablation injector that uses only the motion-side WTCP prior."""

    def __init__(self, feat_channels, prior_channels=16):
        super().__init__()
        self.map = nn.Sequential(
            nn.Conv2d(prior_channels, feat_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.SiLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, q, p_q):
        _, _, _, H, W = q.shape
        if p_q.shape[-2:] != (H, W):
            p_q = F.interpolate(p_q, size=(H, W), mode="bilinear", align_corners=False)
        return q + self.alpha * self.map(p_q).unsqueeze(2)
