"""Generate CAD layouts for an object list: K samples -> PNG grid + DXF/JSON NURBS files.

Object list JSON: [{"type": "circle", "size": 0.1}, {"type": "square", "size": 0.2}, ...]
(size = radius for circles, side length for squares, room = unit square)

    python -m cadlayout.sample --objects objs.json --k 8 --out outputs/sample
    python -m cadlayout.sample --from-test 0 --k 8 --out outputs/sample
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from . import CIRCLE, SQUARE
from .data import LayoutDataset
from .geometry import layout_violations
from .nurbs import export_dxf, export_json
from .refine import refine
from .train import get_device, load_checkpoint
from .viz import plot_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/diffusion/best.pt")
    ap.add_argument("--objects", help="JSON file with the object list")
    ap.add_argument("--from-test", type=int, help="use object set #i of data/test.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=0.05)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/sample")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = get_device(args.device)

    if args.objects:
        objs = json.load(open(args.objects))
        types = np.array([CIRCLE if o["type"] == "circle" else SQUARE for o in objs])
        sizes = np.array([float(o["size"]) for o in objs])
    else:
        ds = LayoutDataset(os.path.join(args.data, "test.pt"))
        t, s, _, m = ds[args.from_test or 0]
        types, sizes = t[m].numpy(), s[m].numpy().astype(float)

    model, diff, _ = load_checkpoint(args.ckpt, dev)
    T = torch.as_tensor(types, device=dev).long()[None].repeat(args.k, 1)
    S = torch.as_tensor(sizes, dtype=torch.float32, device=dev)[None].repeat(args.k, 1)
    M = torch.ones_like(T, dtype=torch.bool)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    poses = diff.sample(model, T, S, M, steps=args.steps, guidance=args.guidance, generator=g)
    if not args.no_refine:
        poses = refine(T, S, poses, M)
    poses = poses.cpu().numpy().astype(float)

    layouts, titles = [], []
    for k in range(args.k):
        v = layout_violations(types, sizes, poses[k])
        layouts.append((types, sizes, poses[k]))
        titles.append(f"sample {k}: {'valid' if v['valid'] else 'INVALID'}")
        export_json(os.path.join(args.out, f"layout_{k}.json"), types, sizes, poses[k])
        export_dxf(os.path.join(args.out, f"layout_{k}.dxf"), types, sizes, poses[k])
    plot_grid(layouts, os.path.join(args.out, "samples.png"), titles=titles)
    print("\n".join(titles))
    print(f"wrote {args.k} x (layout_k.dxf, layout_k.json) and samples.png to {args.out}")


if __name__ == "__main__":
    main()
