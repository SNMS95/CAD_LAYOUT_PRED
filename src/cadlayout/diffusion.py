"""DDPM training objective and DDIM sampling (with constraint guidance) over object poses."""
from __future__ import annotations

import math

import torch

from . import CIRCLE
from .data import channel_mask, decode_pose
from .geometry import layout_penalty


class Diffusion:
    def __init__(self, T: int = 500, device="cpu"):
        self.T = T
        s = 0.008
        t = torch.linspace(0, T, T + 1, dtype=torch.float64)
        f = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
        abar = f / f[0]
        betas = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
        self.abar = torch.cumprod(1 - betas, 0).float().to(device)  # abar[t], t = 0..T-1

    def to(self, device):
        self.abar = self.abar.to(device)
        return self

    def loss(self, model, batch, overlap_weight: float = 0.0):
        """ε-prediction MSE, optionally + λ·mean_b[ᾱ_t · P(x̂0)].

        x̂0 is the network's implied clean layout. The ᾱ_t weight applies the constraint
        only where x̂0 is meaningful (low noise) and cancels the 1/√ᾱ_t blow-up of x̂0.
        Returns (total, {"eps": ..., "overlap": ...}); "overlap" is the weighted penalty.
        """
        types, sizes, mask = batch["types"], batch["sizes"], batch["mask"]
        x0 = batch["x0"]
        cm = channel_mask(types, mask)
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        a = self.abar[t][:, None, None]
        eps = torch.randn_like(x0)
        x_t = (a.sqrt() * x0 + (1 - a).sqrt() * eps) * cm
        eps_hat = model(types, sizes, mask, x_t, t)
        l_eps = (((eps_hat - eps) ** 2) * cm).sum() / cm.sum()
        parts = {"eps": l_eps.detach()}
        if overlap_weight > 0:
            x0_hat = (x_t - (1 - a).sqrt() * eps_hat) / a.sqrt() * cm
            pen = self.penalty_on_encoded(x0_hat, types, sizes, mask)  # (B,)
            l_ovl = (self.abar[t] * pen).mean()
            parts["overlap"] = l_ovl.detach()
            return l_eps + overlap_weight * l_ovl, parts
        return l_eps, parts

    def penalty_on_encoded(self, z, types, sizes, mask):
        """Constraint penalty evaluated on encoded poses (differentiable)."""
        p = decode_pose(z)
        return layout_penalty(types, sizes, p[..., :2], p[..., 2], mask)

    @torch.no_grad()
    def sample(
        self,
        model,
        types,
        sizes,
        mask,
        steps: int = 50,
        guidance: float = 0.0,
        guide_frac: float = 0.3,
        guide_iters: int = 5,
        generator: torch.Generator | None = None,
    ):
        """DDIM (eta=0). Returns decoded poses (B, N, 3) in room coords (x, y, theta).

        If guidance > 0, during the last `guide_frac` of the steps the predicted clean
        sample x̂0 is pushed down the gradient of the overlap/boundary penalty before
        taking the DDIM step (penalty measured in room units).
        """
        dev = sizes.device
        cm = channel_mask(types, mask)
        B, N = types.shape
        x = torch.randn(B, N, 4, device=dev, generator=generator) * cm
        ts = torch.linspace(self.T - 1, 0, steps, device=dev).round().long()
        guide_start = int((1 - guide_frac) * steps)
        x0 = x
        for i, t in enumerate(ts):
            a = self.abar[t]
            eps = model(types, sizes, mask, x, t.expand(B))
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1.0, 1.0) * cm
            if guidance > 0 and i >= guide_start:
                x0 = self.guide(x0, types, sizes, mask, cm, guidance, guide_iters)
            a_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
            eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
            x = (a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps) * cm
        return self.finalize(x0, types)

    def guide(self, x0, types, sizes, mask, cm, step, iters):
        with torch.enable_grad():
            z = x0.detach().clone()
            for _ in range(iters):
                z.requires_grad_(True)
                pen = self.penalty_on_encoded(z, types, sizes, mask).sum()
                (g,) = torch.autograd.grad(pen, z)
                z = (z - step * g * cm).detach()
        return z.clamp(-1.0, 1.0)

    @staticmethod
    def finalize(z, types):
        poses = decode_pose(z)
        return torch.where((types == CIRCLE)[..., None], poses * torch.tensor([1.0, 1.0, 0.0], device=z.device), poses)
