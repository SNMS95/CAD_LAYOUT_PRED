"""Post-hoc projection: a few hundred Adam steps that remove residual overlaps/out-of-room
violations while staying close to the sampled poses. No learning involved."""
from __future__ import annotations

import torch

from .geometry import layout_penalty


def refine(types, sizes, poses, mask, steps: int = 300, lr: float = 2e-3, anchor: float = 1e-2):
    """poses (B, N, 3) room coords -> refined poses (B, N, 3)."""
    init = poses.detach()
    p = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    m = mask.float()[..., None]
    for _ in range(steps):
        pen = layout_penalty(types, sizes, p[..., :2], p[..., 2], mask)
        if float(pen.detach().max()) == 0.0:
            break
        loss = pen.sum() + anchor * (((p - init) ** 2) * m).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return p.detach()
