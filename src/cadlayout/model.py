"""SetGraphDenoiser: a graph transformer over the complete graph of objects.

* nodes  = objects (type, size) [+ noisy pose], plus one learned room token
* edges  = all pairs; edge features from the noisy poses become a per-head attention
           bias (so attention == attention-weighted message passing with edge features)
* no positional encoding -> permutation-equivariant, any number of objects
* time conditioning via adaLN/FiLM (scale & shift of each LayerNorm)

The same backbone with `diffusion=False` is the deterministic regression baseline
(no time, no noisy-pose input, predicts the encoded pose directly).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import CIRCLE
from .geometry import circumradius

POSE_DIM = 4


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ff_mult: int = 4):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(d, elementwise_affine=False)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = mlp(d, ff_mult * d, d)
        # adaLN: cond -> (shift1, scale1, gate1, shift2, scale2, gate2), zero-init
        self.mod = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, h, cond, attn_bias):
        B, N, D = h.shape
        sh1, sc1, g1, sh2, sc2, g2 = self.mod(cond)[:, None].chunk(6, dim=-1)
        x = self.ln1(h) * (1 + sc1) + sh1
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        h = h + (1 + g1) * self.proj(a.transpose(1, 2).reshape(B, N, D))
        x = self.ln2(h) * (1 + sc2) + sh2
        return h + (1 + g2) * self.ff(x)


class SetGraphDenoiser(nn.Module):
    def __init__(self, d: int = 128, layers: int = 6, heads: int = 8, diffusion: bool = True):
        super().__init__()
        self.d, self.heads, self.diffusion = d, heads, diffusion
        self.obj_embed = mlp(4, d, d)  # [onehot(type) (2), size, area]
        self.in_proj = nn.Linear(d + (POSE_DIM if diffusion else 0), d)
        self.room = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.time_embed = mlp(d, d, d) if diffusion else None
        self.cond0 = nn.Parameter(torch.zeros(1, d))  # cond when there is no timestep
        if diffusion:  # edge features: Δx, Δy, |Δ|, |Δ| - (ρi + ρj)
            self.edge_mlp = mlp(4, 64, heads)
            nn.init.zeros_(self.edge_mlp[-1].weight)
            nn.init.zeros_(self.edge_mlp[-1].bias)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, POSE_DIM)
        if diffusion:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, types, sizes, mask, x_t=None, t=None):
        """types (B,N) long, sizes (B,N), mask (B,N) bool, x_t (B,N,4) encoded poses, t (B,).

        Returns (B, N, 4): ε̂ (diffusion) or encoded pose (regressor).
        """
        B, N = types.shape
        area = torch.where(types == CIRCLE, math.pi * sizes**2, sizes**2)
        feat = torch.cat([F.one_hot(types, 2).float(), sizes[..., None] * 4, area[..., None] * 10], -1)
        h = self.obj_embed(feat)
        if self.diffusion:
            h = torch.cat([h, x_t], dim=-1)
        h = self.in_proj(h)
        h = torch.cat([self.room.expand(B, 1, self.d), h], dim=1)  # (B, 1+N, d)

        if self.diffusion:
            cond = self.time_embed(timestep_embedding(t, self.d))
        else:
            cond = self.cond0.expand(B, self.d)

        # attention bias: edge features + key padding (room token always visible)
        key_ok = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], 1)
        bias = torch.zeros(B, self.heads, N + 1, N + 1, device=h.device, dtype=h.dtype)
        if self.diffusion:
            pos = x_t[..., :2]  # normalised coords ([-1, 1] = room)
            delta = pos[:, :, None] - pos[:, None, :]
            dist = torch.sqrt((delta**2).sum(-1) + 1e-8)
            rho = 2 * circumradius(types, sizes)  # to normalised units
            gap = dist - (rho[:, :, None] + rho[:, None, :])
            e = torch.cat([delta, dist[..., None], gap[..., None]], -1)
            bias[:, :, 1:, 1:] = self.edge_mlp(e).permute(0, 3, 1, 2)
        bias = bias.masked_fill(~key_ok[:, None, None, :], float("-inf"))

        for blk in self.blocks:
            h = blk(h, cond, bias)
        return self.head(self.ln_out(h[:, 1:]))
