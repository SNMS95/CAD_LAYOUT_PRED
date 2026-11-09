"""Synthetic data: random object sets + random valid layouts (rejection sampling).

Usage:
    python -m cadlayout.data --out data --n-train 50000 --workers 8
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool

import numpy as np
import torch
from torch.utils.data import Dataset

from . import CIRCLE, SQUARE
from .geometry import fits, half_extent, to_shapely


@dataclass
class GenConfig:
    n_min: int = 2
    n_max: int = 8
    p_circle: float = 0.5
    r_range: tuple = (0.03, 0.15)  # circle radius
    s_range: tuple = (0.05, 0.25)  # square side
    max_fill: float = 0.40  # max total object area / room area
    max_tries: int = 200  # pose samples per object
    max_restarts: int = 20  # layout restarts before resampling the object set


def sample_objects(rng: np.random.Generator, cfg: GenConfig):
    while True:
        n = int(rng.integers(cfg.n_min, cfg.n_max + 1))
        types = np.where(rng.random(n) < cfg.p_circle, CIRCLE, SQUARE)
        sizes = np.where(
            types == CIRCLE, rng.uniform(*cfg.r_range, n), rng.uniform(*cfg.s_range, n)
        )
        if object_areas(types, sizes).sum() <= cfg.max_fill:
            return types, sizes


def object_areas(types, sizes):
    return np.where(types == CIRCLE, math.pi * sizes**2, sizes**2)


def place(types, sizes, rng: np.random.Generator, cfg: GenConfig):
    """Random valid placement of a given object set; None if it fails."""
    order = np.argsort(-object_areas(types, sizes))  # largest first
    for _ in range(cfg.max_restarts):
        poses = np.zeros((len(types), 3))
        placed = []
        ok = True
        for i in order:
            for _ in range(cfg.max_tries):
                th = 0.0 if types[i] == CIRCLE else rng.uniform(0, math.pi / 2)
                h = half_extent(types[i], sizes[i], th)
                x, y = rng.uniform(h, 1 - h, size=2)
                g = to_shapely(types[i], sizes[i], x, y, th)
                if fits(g, placed):
                    placed.append(g)
                    poses[i] = (x, y, th)
                    break
            else:
                ok = False
                break
        if ok:
            return poses
    return None


def generate_one(rng, cfg: GenConfig):
    while True:
        types, sizes = sample_objects(rng, cfg)
        poses = place(types, sizes, rng, cfg)
        if poses is not None:
            return types, sizes, poses


def _worker(args):
    seed, count, cfg = args
    rng = np.random.default_rng(seed)
    return [generate_one(rng, cfg) for _ in range(count)]


def generate(n: int, cfg: GenConfig, seed: int = 0, workers: int = 1):
    chunks = max(1, workers * 4)
    per = [n // chunks + (1 if k < n % chunks else 0) for k in range(chunks)]
    jobs = [(seed * 100_003 + k, c, cfg) for k, c in enumerate(per) if c > 0]
    if workers > 1:
        with Pool(workers) as pool:
            parts = pool.map(_worker, jobs)
    else:
        parts = [_worker(j) for j in jobs]
    return pack([lay for part in parts for lay in part], cfg.n_max)


def pack(layouts, n_max: int) -> dict:
    """Pad to n_max: types (L, n_max) long, sizes, poses (L, n_max, 3), mask."""
    L = len(layouts)
    types = torch.zeros(L, n_max, dtype=torch.long)
    sizes = torch.zeros(L, n_max)
    poses = torch.zeros(L, n_max, 3)
    mask = torch.zeros(L, n_max, dtype=torch.bool)
    for k, (t, s, p) in enumerate(layouts):
        n = len(t)
        types[k, :n] = torch.as_tensor(t)
        sizes[k, :n] = torch.as_tensor(s, dtype=torch.float32)
        poses[k, :n] = torch.as_tensor(p, dtype=torch.float32)
        mask[k, :n] = True
    return {"types": types, "sizes": sizes, "poses": poses, "mask": mask}


# --------------------------------------------------------------------------- encoding


def encode_pose(poses: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
    """(x, y, theta) room coords -> (2x-1, 2y-1, cos4θ, sin4θ); circles' rot channels = 0.

    The factor 4 makes the target invariant to the square's 90° symmetry.
    """
    x = 2 * poses[..., 0] - 1
    y = 2 * poses[..., 1] - 1
    sq = (types == SQUARE).to(poses.dtype)
    c = torch.cos(4 * poses[..., 2]) * sq
    s = torch.sin(4 * poses[..., 2]) * sq
    return torch.stack([x, y, c, s], dim=-1)


def decode_pose(z: torch.Tensor) -> torch.Tensor:
    """Inverse of encode_pose -> (x, y, theta) with theta in (-π/4, π/4]."""
    x = (z[..., 0] + 1) / 2
    y = (z[..., 1] + 1) / 2
    th = torch.atan2(z[..., 3], z[..., 2]) / 4
    return torch.stack([x, y, th], dim=-1)


def channel_mask(types: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B, N, 4) float: which pose channels are meaningful (circles have no rotation)."""
    cm = torch.ones(*types.shape, 4, device=types.device)
    cm[..., 2:] = (types == SQUARE).float()[..., None]
    return cm * mask.float()[..., None]


class LayoutDataset(Dataset):
    def __init__(self, path_or_dict):
        d = torch.load(path_or_dict) if isinstance(path_or_dict, str) else path_or_dict
        self.types, self.sizes, self.poses, self.mask = d["types"], d["sizes"], d["poses"], d["mask"]

    def __len__(self):
        return len(self.types)

    def __getitem__(self, i):
        return self.types[i], self.sizes[i], self.poses[i], self.mask[i]


def collate(batch):
    types, sizes, poses, mask = (torch.stack(x) for x in zip(*batch))
    n = int(mask.sum(1).max())  # trim padding to the largest set in the batch
    return {"types": types[:, :n], "sizes": sizes[:, :n], "poses": poses[:, :n], "mask": mask[:, :n]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=50_000)
    ap.add_argument("--n-val", type=int, default=5_000)
    ap.add_argument("--n-test", type=int, default=5_000)
    ap.add_argument("--n-test-ood", type=int, default=2_000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = GenConfig()
    ood = GenConfig(n_min=9, n_max=12)
    splits = [
        ("train", args.n_train, cfg, 1),
        ("val", args.n_val, cfg, 2),
        ("test", args.n_test, cfg, 3),
        ("test_ood", args.n_test_ood, ood, 4),
    ]
    for name, n, c, seed in splits:
        t0 = time.time()
        d = generate(n, c, seed=seed, workers=args.workers)
        d["config"] = asdict(c)
        torch.save(d, os.path.join(args.out, f"{name}.pt"))
        print(f"{name}: {n} layouts, N∈[{c.n_min},{c.n_max}], {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
