import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- helpers ----------
def conv_dw_pw(in_ch, out_ch, kernel_size, padding=0, dilation=1):
    # depthwise separable conv: depthwise then pointwise
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, dilation=dilation, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class SobelLayer(nn.Module):
    """Compute per-channel gradient magnitude approx via Sobel filters (works with groups conv)."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # 3x3 Sobel kernels for x and y
        kx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], dtype=torch.float32)
        ky = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]], dtype=torch.float32)
        kx = kx.view(1,1,3,3)
        ky = ky.view(1,1,3,3)
        # We'll use groups=channels to apply same kernel per channel
        self.register_buffer('kx', kx.repeat(channels,1,1,1))  # (C,1,3,3)
        self.register_buffer('ky', ky.repeat(channels,1,1,1))

    def forward(self, x):
        # x: (B,C,H,W)
        B,C,H,W = x.shape
        # pad with reflect to preserve size
        pad = 1
        # conv with groups=C: weight shape (C,1,3,3)
        gx = F.conv2d(x, self.kx, bias=None, stride=1, padding=pad, groups=C)
        gy = F.conv2d(x, self.ky, bias=None, stride=1, padding=pad, groups=C)
        # gradient magnitude per channel
        gm = torch.sqrt(gx*gx + gy*gy + 1e-6)  # (B,C,H,W)
        return gm

# ---------- FDC Block ----------
class LFDCBlock(nn.Module):
    """
    Improved Frequency Dynamic Convolution block:
      - LFBD: 3 parallel depthwise-separable convs (5x5, 3x3, 1x1) -> F_low, F_mid, F_high
      - LSM: compute G from input F (per-channel gradient magnitude averaged), produce pixel-wise beta weights
             and modulate the 3 band features per-pixel
      - FBM: compute global energy per band and fuse with alpha weights
    """
    def __init__(self, in_ch, out_ch, reduction=4):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        # LFBD: depthwise separable convs as band proxies
        self.band_low  = conv_dw_pw(in_ch, out_ch, kernel_size=5, padding=2)
        self.band_mid  = conv_dw_pw(in_ch, out_ch, kernel_size=3, padding=1)
        # high band uses a high-pass prefilter + small kernel; here we approximate via 3x3 dilated or 1x1
        # combined with subtractive average pooling for high-pass
        self.band_high_prepool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.band_high = conv_dw_pw(in_ch, out_ch, kernel_size=3, padding=2, dilation=2)

        # Sobel-like gradient estimator for LSM (compute on original input F)
        self.sobel = SobelLayer(in_ch)

        # LSM: map G (B,1,H,W) [+ optionally band features] -> pixel-wise 3 weights
        # We'll use small conv net that takes concatenation of G and channel-compressed band responses
        # to produce beta (B,3,H,W)
        self.compress = nn.Conv2d(in_ch, max(8, in_ch//reduction), 1, bias=False)  # for computational efficiency
        self.lsm_net = nn.Sequential(
            nn.Conv2d(1 + max(8, in_ch//reduction)*0 + 3 * (out_ch//reduction if False else 1), 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1)  # outputs raw scores for low/mid/high
        )
        # Note: to keep it simple & efficient, we'll only feed G and spatially pooled compressed features.
        # Simpler robust variant below (more practical)

        # Practical simpler LSM: G -> small conv -> 3-channel beta (softmax over channel dim)
        self.lsm_simple = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1)  # raw scores
        )

        # FBM: fusion conv after concatenation or energy-weighted fusion
        # We'll implement global-energy-based weights alpha (per-sample)
        self.fuse_conv = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        # small learnable scaling for residual
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, F):
        """
        F: (B, C, H, W)
        returns: F_fdc (B, out_ch, H, W)
        """
        B,C,H,W = F.shape

        # 1) LFBD: produce 3 band feature maps (B, out_ch, H, W)
        F_low = self.band_low(F)
        F_mid = self.band_mid(F)
        # high: prefilter (high-pass) then conv
        F_hp = F - self.band_high_prepool(F)  # high-pass approx
        F_high = self.band_high(F_hp)

        # 2) LSM: compute G(p) from input F
        gm = self.sobel(F)             # (B,C,H,W): per-channel gradient magnitude
        G = gm.mean(dim=1, keepdim=True)   # (B,1,H,W) average across channels -> local highfreq map

        # optional local smoothing: helps suppress isolated noise
        G_s = F.avg_pool2d = F  # placeholder: skip heavy operations; (we can smooth G if needed)
        # compute pixel-wise beta weights (B,3,H,W)
        beta_logits = self.lsm_simple(G)   # raw scores
        beta = torch.softmax(beta_logits, dim=1)  # softmax along channel (3 bands), sum=1 per pixel

        # 3) pixel-wise modulation: apply beta to bands
        # Ensure same channel dims (out_ch)
        # Each band is (B, out_ch, H, W); beta is (B,3,H,W) -> expand to match channels
        b_low  = beta[:,0:1,:,:].expand(-1, F_low.size(1), -1, -1)
        b_mid  = beta[:,1:2,:,:].expand(-1, F_mid.size(1), -1, -1)
        b_high = beta[:,2:3,:,:].expand(-1, F_high.size(1), -1, -1)

        F_low_t  = F_low  * b_low
        F_mid_t  = F_mid  * b_mid
        F_high_t = F_high * b_high

        # 4) FBM: global energy calculation and fusion
        # Compute per-band global energy per sample:
        # E_i shape (B,)
        E_low  = (F_low_t  ** 2).mean(dim=(1,2,3))   # (B,)
        E_mid  = (F_mid_t  ** 2).mean(dim=(1,2,3))
        E_high = (F_high_t ** 2).mean(dim=(1,2,3))
        E = torch.stack([E_low, E_mid, E_high], dim=1)  # (B,3)
        # normalize to get alpha per sample
        alpha = E / (E.sum(dim=1, keepdim=True) + 1e-6)  # (B,3)
        alpha = alpha.view(B,3,1,1)  # (B,3,1,1)

        # fuse using alpha weights (apply per-sample)
        F_fused = alpha[:,0:1,:,:].expand(-1, F_low_t.size(1), -1, -1) * F_low_t \
                + alpha[:,1:2,:,:].expand(-1, F_mid_t.size(1), -1, -1) * F_mid_t \
                + alpha[:,2:3,:,:].expand(-1, F_high_t.size(1), -1, -1) * F_high_t

        # optional 1x1 refine
        out = self.fuse_conv(F_fused)
        out = self.bn_out(out)
        out = self.act(out)

        # residual add (project if necessary)
        if out.shape[1] != C:
            # if channel mismatch, project F to out_ch and add scaled residual
            proj = nn.Conv2d(C, out.shape[1], 1).to(F.device)
            F_proj = proj(F)
            out = F_proj + self.gamma * out
        else:
            out = F + self.gamma * out

        return out
