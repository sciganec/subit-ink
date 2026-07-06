import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class CurvatureField2D(nn.Module):
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.smooth = self._make_gaussian_kernel(sigma)

    def _make_gaussian_kernel(self, sigma: float) -> nn.Conv2d:
        size = int(6 * sigma + 1) | 1
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-coords**2 / (2 * sigma**2))
        g = g / g.sum()
        kernel = g[:, None] * g[None, :]
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        conv = nn.Conv2d(1, 1, size, padding=size//2, bias=False)
        conv.weight = nn.Parameter(kernel, requires_grad=False)
        return conv

    def _second_derivative(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        pad = [0, 0, 0, 0]
        pad[-(dim*2+2)] = 1; pad[-(dim*2+1)] = 1
        x_pad = F.pad(x, pad, mode='replicate')
        slices_m = [slice(None)] * x_pad.dim(); slices_c = [slice(None)] * x_pad.dim(); slices_p = [slice(None)] * x_pad.dim()
        d = 2 + dim
        slices_m[d] = slice(0, -2); slices_c[d] = slice(1, -1); slices_p[d] = slice(2, None)
        return x_pad[tuple(slices_m)] - 2 * x_pad[tuple(slices_c)] + x_pad[tuple(slices_p)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W). Зводимо до 1 каналу для аналізу текстури
        x_gray = x.mean(dim=1, keepdim=True)
        xs = self.smooth(x_gray)
        hyy = self._second_derivative(xs, 1); hxx = self._second_derivative(xs, 2)
        
        # Власні значення 2D Гессе
        eigs = torch.stack([hxx.abs(), hyy.abs()], dim=1).squeeze(2)
        eigs_sorted, _ = torch.sort(eigs, dim=1)
        l1, l2 = eigs_sorted[:, 0], eigs_sorted[:, 1]
        
        eps = 1e-6; norm = l2 + eps
        flat_score = (1 - l1/norm).squeeze(1) # Пласкі ділянки (папірус)
        edge_score = (l1/norm).squeeze(1)     # Краї (межі літер або тріщини)
        convex_score = ((hxx < 0) & (hyy < 0)).float().squeeze(1)
        concave_score = ((hxx > 0) & (hyy > 0)).float().squeeze(1)
        return torch.stack([flat_score, edge_score, convex_score, concave_score], dim=1)

class SUBITv2Encoder2D(nn.Module):
    WHEN_SCALES = [1, 2, 4, 8]
    def __init__(self, in_channels=32, embed_dim=32):
        super().__init__()
        self.curvature_fields = nn.ModuleList([CurvatureField2D(sigma=s) for s in self.WHEN_SCALES])
        self.subit_proj = nn.Sequential(
            nn.Conv2d(16, embed_dim // 2, 1), nn.GroupNorm(8, embed_dim // 2), nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_features = [curv_field(x) for curv_field in self.curvature_fields]
        return self.subit_proj(torch.cat(scale_features, dim=1))

class LearnedEvolutionFtheta2D(nn.Module):
    def __init__(self, channels: int, n_steps: int = 3):
        super().__init__()
        self.n_steps = n_steps
        self.diffusion_net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1), nn.GroupNorm(8, channels), nn.GELU(),
            nn.Conv2d(channels, 3, 1), # Dxx, Dyy, Dxy
        )
        self.dt = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)
        self.proj = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(8, channels), nn.GELU())
        self.stability_gate = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def anisotropic_step(self, x: torch.Tensor, D_tensor: torch.Tensor) -> torch.Tensor:
        Dxx, Dyy = D_tensor[:, 0:1].abs(), D_tensor[:, 1:2].abs()
        pad = nn.ReplicationPad2d(1); xp = pad(x)
        d2y = (xp[:, :, 2:, 1:-1] - 2*x + xp[:, :, :-2, 1:-1])
        d2x = (xp[:, :, 1:-1, 2:] - 2*x + xp[:, :, 1:-1, :-2])
        return x + self.dt * (Dyy * d2y + Dxx * d2x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x; D_tensor = self.diffusion_net(x); gate = self.stability_gate(x); h = x
        for _ in range(self.n_steps): h = self.anisotropic_step(h, D_tensor)
        h = self.proj(h)
        return residual + gate * (h - residual)

class ConvBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(min(8, out_ch), out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(min(8, out_ch), out_ch), nn.GELU(),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x): return self.block(x) + self.skip(x)

class SUBITAttention2D(nn.Module):
    def __init__(self, feat_ch: int, subit_ch: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(feat_ch + subit_ch, feat_ch, 1), nn.GroupNorm(min(8, feat_ch), feat_ch), nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, 1), nn.Sigmoid(),
        )
    def forward(self, feat, subit): return feat * self.gate(torch.cat([feat, subit], dim=1))

class SUBITInkUNet(nn.Module):
    def __init__(self, in_channels=32, features=(32, 64, 128, 256), subit_embed_dim=32):
        super().__init__()
        self.subit_encoder = SUBITv2Encoder2D(in_channels=in_channels, embed_dim=subit_embed_dim)
        self.input_proj = nn.Conv2d(in_channels + subit_embed_dim, features[0], 1)
        self.encoders, self.pools, self.enc_attn = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        ch = features[0]
        for f in features:
            self.encoders.append(ConvBlock2D(ch, f)); self.pools.append(nn.MaxPool2d(2))
            self.enc_attn.append(SUBITAttention2D(f, subit_embed_dim)); ch = f
        self.bottleneck = ConvBlock2D(features[-1], features[-1] * 2)
        self.evolution = LearnedEvolutionFtheta2D(features[-1] * 2, n_steps=3)
        self.upconvs, self.decoders, self.dec_attn = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        rev_features = list(reversed(features)); ch = features[-1] * 2
        for f in rev_features:
            self.upconvs.append(nn.ConvTranspose2d(ch, f, 2, stride=2))
            self.decoders.append(ConvBlock2D(f * 2, f)); self.dec_attn.append(SUBITAttention2D(f, subit_embed_dim)); ch = f
        self.feature_proj = nn.Conv2d(features[0], features[0], 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        subit_emb = self.subit_encoder(x); subit_skips = [subit_emb]
        for _ in range(len(self.encoders) - 1): subit_skips.append(F.avg_pool2d(subit_skips[-1], 2))
        h = self.input_proj(torch.cat([x, subit_emb], dim=1)); skips = []
        for i, (enc, pool, attn) in enumerate(zip(self.encoders, self.pools, self.enc_attn)):
            h = attn(enc(h), subit_skips[i]); skips.append(h); h = pool(h)
        h = self.evolution(self.bottleneck(h))
        for i, (upconv, dec, attn, skip) in enumerate(zip(self.upconvs, self.decoders, self.dec_attn, reversed(skips))):
            h = upconv(h)
            if h.shape != skip.shape: h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)
            h = attn(dec(torch.cat([h, skip], dim=1)), F.interpolate(subit_emb, size=h.shape[2:], mode='bilinear', align_corners=False))
        return self.feature_proj(h), subit_emb

class AttractorDecoder2D(nn.Module):
    def __init__(self, channels: int, n_attractor_steps: int = 5):
        super().__init__()
        self.n_steps = n_attractor_steps
        self.ftheta = LearnedEvolutionFtheta2D(channels, n_steps=1)
        self.omega_head = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1), nn.GroupNorm(min(8, channels), channels), nn.GELU(),
            nn.Conv2d(channels, 4, 1), nn.Softmax(dim=1)
        )
        self.ink_head = nn.Conv2d(channels, 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x; trajectory_delta = torch.zeros_like(x)
        for _ in range(self.n_steps):
            h_new = self.ftheta(h)
            trajectory_delta = trajectory_delta + (h_new - h).abs()
            h = h_new
        omega = self.omega_head(torch.cat([h, trajectory_delta], dim=1))
        # Стабільні атрактори (літери) підсилюємо, хаотичні (тріщини) пригнічуємо
        omega_weight = omega[:, 3:4] * 1.5 + omega[:, 2:3] * 1.2 + omega[:, 1:2] * 0.8 + omega[:, 0:1] * 0.3
        return self.ink_head(h * omega_weight), omega

class OmegaLossV2_2D(nn.Module):
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta

    def forward(self, logits, labels, omega, subit_emb):
        probs = F.softmax(logits, dim=1)
        ce = F.cross_entropy(logits, labels)
        smooth = 1e-5; p, l = probs[:, 1], labels.float()
        inter = (p * l).sum(dim=(1, 2)); union = p.sum(dim=(1, 2)) + l.sum(dim=(1, 2))
        dice = (1 - (2 * inter + smooth) / (union + smooth)).mean()
        s_emb = F.interpolate(subit_emb, size=probs.shape[2:], mode='bilinear', align_corners=False) if subit_emb.shape[2:] != probs.shape[2:] else subit_emb
        curv = (torch.sigmoid(s_emb[:, 0]) * (1 - probs[:, 1])).mean()
        stab = (omega[:, 0] * probs[:, 1]).mean()
        omega_mean = omega.mean(dim=(0, 2, 3))
        omega_reg = -(-(omega_mean * (omega_mean + 1e-8).log()).sum()) * 0.01
        total = self.alpha * ce + (1 - self.alpha) * dice + self.gamma * curv + self.beta * stab + self.delta * omega_reg
        return total, {"ce": ce.item(), "dice": dice.item(), "curvature": curv.item(), "stability": stab.item(), "omega_reg": omega_reg.item(), "total": total.item()}

class VesuviusInkPipeline:
    def __init__(self, device='cuda', lr=1e-4):
        self.device = device
        self.model = SUBITInkUNet(in_channels=32).to(device)
        self.attractor = AttractorDecoder2D(channels=32).to(device)
        self.loss_fn = OmegaLossV2_2D()
        params = list(self.model.parameters()) + list(self.attractor.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)

    def train_step_batch(self, ct_batch, mask_batch):
        self.model.train(); self.attractor.train(); self.optimizer.zero_grad()
        x, y = ct_batch.to(self.device), mask_batch.to(self.device)
        features, subit_emb = self.model(x)
        logits, omega = self.attractor(features)
        if logits.shape[2:] != y.shape[1:]:
            logits = F.interpolate(logits, size=y.shape[1:], mode='bilinear', align_corners=False)
        loss, components = self.loss_fn(logits, y, omega, subit_emb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        with torch.no_grad():
            pred = (F.softmax(logits, dim=1)[:, 1] > 0.5).long()
            p, t = pred.float().flatten(), y.float().flatten()
            components["dice_metric"] = ((2*(p*t).sum()+1e-5) / (p.sum()+t.sum()+1e-5)).item()
        return components