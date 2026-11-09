"""Evaluate diffusion vs. regression baseline on validity, violation size and diversity.

    python -m cadlayout.evaluate --diffusion runs/diffusion/best.pt \
        --regressor runs/regressor/best.pt --split test --out outputs/eval_test
    # extra checkpoints (repeatable):
    python -m cadlayout.evaluate --ckpt diff_ovl1=runs/diffusion_ovl1/best.pt --split test
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .data import GenConfig, LayoutDataset, collate, decode_pose, place
from .diffusion import Diffusion
from .geometry import layout_violations
from .refine import refine
from .train import get_device, load_checkpoint
from .viz import plot_grid


def unpad(b, poses):
    """Batch tensors -> list of (types, sizes, poses) numpy tuples without padding."""
    out = []
    for k in range(b["types"].shape[0]):
        n = int(b["mask"][k].sum())
        out.append(
            (
                b["types"][k, :n].cpu().numpy(),
                b["sizes"][k, :n].cpu().numpy().astype(np.float64),
                poses[k, :n].cpu().numpy().astype(np.float64),
            )
        )
    return out


def metrics(layouts) -> dict:
    v = [layout_violations(*lay) for lay in layouts]
    centre = [np.linalg.norm(p[:, :2] - 0.5, axis=1).mean() for _, _, p in layouts]
    return {
        "valid_pct": 100.0 * np.mean([x["valid"] for x in v]),
        "overlap_area_mean": float(np.mean([x["overlap_area"] for x in v])),
        "oob_area_mean": float(np.mean([x["oob_area"] for x in v])),
        "overlap_pairs_mean": float(np.mean([x["overlap_pairs"] for x in v])),
        "mean_dist_to_centre": float(np.mean(centre)),
    }


def diversity(samples) -> float:
    """samples: (K, N, 3) for one input -> mean over objects of mean pairwise centre distance."""
    xy = samples[:, :, :2]
    K = xy.shape[0]
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)  # (K, K, N)
    iu = np.triu_indices(K, 1)
    return float(d[iu].mean())


def run_model(model, diff, b, guidance, steps, seed, do_refine):
    g = torch.Generator(device=b["sizes"].device).manual_seed(seed)
    if diff is None:
        with torch.no_grad():
            poses = decode_pose(model(b["types"], b["sizes"], b["mask"]))
    else:
        poses = diff.sample(model, b["types"], b["sizes"], b["mask"], steps=steps, guidance=guidance, generator=g)
    if do_refine:
        poses = refine(b["types"], b["sizes"], poses, b["mask"])
    return poses


def best_val(path):
    """Val fit loss (and val overlap, if logged) of the epoch saved in a checkpoint."""
    hist = torch.load(path, map_location="cpu")["history"]
    rec = min(hist, key=lambda r: r["val"])
    return {"epoch": rec["epoch"], "val_fit": rec["val"], "val_overlap": rec.get("val_overlap")}


def write_table(results, path):
    cols = ["valid_pct", "overlap_area_mean", "oob_area_mean", "overlap_pairs_mean", "mean_dist_to_centre"]
    lines = ["| method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, m in results.items():
        if name in ("diversity", "checkpoints"):
            continue
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.4g}" for c in cols) + " |")
    lines += ["", "| diversity | value |", "|---|---|"]
    lines += [f"| {k} | {v:.4f} |" for k, v in results["diversity"].items()]
    if results.get("checkpoints"):
        lines += ["", "| checkpoint | best epoch | val fit | val overlap |", "|---|---|---|---|"]
        for k, v in results["checkpoints"].items():
            vo = "" if v["val_overlap"] is None else f"{v['val_overlap']:.4f}"
            lines.append(f"| {k} | {v['epoch']} | {v['val_fit']:.4f} | {vo} |")
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffusion", default="runs/diffusion/best.pt")
    ap.add_argument("--regressor", default="runs/regressor/best.pt")
    ap.add_argument(
        "--ckpt", action="append", default=[], metavar="NAME=PATH",
        help="extra checkpoint to evaluate (repeatable), e.g. --ckpt diff_ovl1=runs/diffusion_ovl1/best.pt",
    )
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n-div", type=int, default=50)
    ap.add_argument("--k-div", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=250)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    out = args.out or f"outputs/eval_{args.split}"
    os.makedirs(out, exist_ok=True)
    dev = get_device(args.device)

    ds = LayoutDataset(os.path.join(args.data, f"{args.split}.pt"))
    idx = list(range(min(args.n, len(ds))))
    batches = [collate([ds[i] for i in idx[s : s + args.bs]]) for s in range(0, len(idx), args.bs)]
    batches = [{k: v.to(dev) for k, v in b.items()} for b in batches]

    paths = {}
    for name, path in [("diffusion", args.diffusion), ("regressor", args.regressor)]:
        if path and os.path.exists(path):
            paths[name] = path
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            raise SystemExit(f"checkpoint not found: {path}")
        paths[name] = path
    models = {name: load_checkpoint(path, dev) for name, path in paths.items()}

    methods = {"ground_truth": None}
    for name, (_, diff, _) in models.items():
        if diff is not None:
            methods[f"{name}_raw"] = (name, 0.0, False)
            methods[f"{name}_guided"] = (name, args.guidance, False)
            methods[f"{name}_guided_refined"] = (name, args.guidance, True)
        else:
            methods[name] = (name, 0.0, False)
            methods[f"{name}_refined"] = (name, 0.0, True)

    results = {}
    for name, spec in methods.items():
        layouts = []
        for b in batches:
            if spec is None:
                poses = b["poses"]
            else:
                (model, diff, _), guid, ref = models[spec[0]], spec[1], spec[2]
                poses = run_model(model, diff, b, guid, args.steps, seed=0, do_refine=ref)
            layouts += unpad(b, poses)
        results[name] = metrics(layouts)
        print(name, json.dumps({k: round(v, 5) for k, v in results[name].items()}), flush=True)
        plot_grid(layouts[:8], os.path.join(out, f"{name}.png"), titles=[f"{name} #{i}" for i in range(8)])

    # diversity: K samples for the same object set
    div = {}
    sub = collate([ds[i] for i in idx[: args.n_div]])
    sub = {k: v.to(dev) for k, v in sub.items()}
    rng = np.random.default_rng(0)
    cfg = GenConfig(n_max=99)
    ref = []
    for types, sizes, _ in unpad(sub, sub["poses"]):
        ks = []
        while len(ks) < args.k_div:
            p = place(types, sizes, rng, cfg)
            if p is not None:
                ks.append(p)
        ref.append(diversity(np.stack(ks)))
    div["generator_reference"] = float(np.mean(ref))
    for mname, (model, diff, _) in models.items():
        if diff is None:
            div[mname] = 0.0  # deterministic by construction
            continue
        for name, guid in [(f"{mname}_raw", 0.0), (f"{mname}_guided", args.guidance)]:
            ks = [run_model(model, diff, sub, guid, args.steps, seed=s, do_refine=False) for s in range(args.k_div)]
            ks = torch.stack(ks).cpu().numpy()  # (K, B, N, 3)
            vals = [diversity(ks[:, k, : int(sub["mask"][k].sum())]) for k in range(ks.shape[1])]
            div[name] = float(np.mean(vals))
            n0 = int(sub["mask"][0].sum())
            t0, s0 = sub["types"][0, :n0].cpu().numpy(), sub["sizes"][0, :n0].cpu().numpy().astype(float)
            plot_grid(
                [(t0, s0, ks[k, 0, :n0].astype(float)) for k in range(args.k_div)],
                os.path.join(out, f"{name}_diversity.png"),
                titles=[f"sample {k}" for k in range(args.k_div)],
            )
    results["diversity"] = div
    results["checkpoints"] = {name: best_val(path) for name, path in paths.items()}
    print("diversity", json.dumps({k: round(v, 4) for k, v in div.items()}))
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_table(results, os.path.join(out, "table.md"))
    print(f"wrote {out}/metrics.json and {out}/table.md")

if __name__ == "__main__":
    main()
