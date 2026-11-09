"""Train the diffusion denoiser or the deterministic regression baseline.

    python -m cadlayout.train --model diffusion --epochs 100 --out runs/diffusion
    python -m cadlayout.train --model regressor --epochs 100 --out runs/regressor
    python -m cadlayout.train --model diffusion --overlap-weight 1.0 --out runs/diffusion_ovl1

Each run folder gets last.pt (every epoch), best.pt (lowest val fit loss) and final.pt.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import torch
from torch.utils.data import DataLoader, Subset

from .data import LayoutDataset, channel_mask, collate, decode_pose, encode_pose
from .diffusion import Diffusion
from .geometry import layout_penalty
from .model import SetGraphDenoiser


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(batch, dev):
    b = {k: v.to(dev) for k, v in batch.items()}
    b["x0"] = encode_pose(b["poses"], b["types"])
    return b


def regression_loss(model, b, overlap_weight: float = 0.0):
    """MSE on encoded poses, optionally + λ·mean_b P(prediction). Returns (total, parts)."""
    cm = channel_mask(b["types"], b["mask"])
    pred = model(b["types"], b["sizes"], b["mask"])
    mse = (((pred - b["x0"]) ** 2) * cm).sum() / cm.sum()
    parts = {"mse": mse.detach()}
    if overlap_weight > 0:
        p = decode_pose(pred * cm)
        l_ovl = layout_penalty(b["types"], b["sizes"], p[..., :2], p[..., 2], b["mask"]).mean()
        parts["overlap"] = l_ovl.detach()
        return mse + overlap_weight * l_ovl, parts
    return mse, parts


def build_model(cfg: dict) -> SetGraphDenoiser:
    return SetGraphDenoiser(
        d=cfg["d"], layers=cfg["layers"], heads=cfg["heads"], diffusion=cfg["model"] == "diffusion"
    )


def load_checkpoint(path: str, device):
    ck = torch.load(path, map_location=device)
    model = build_model(ck["config"]).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    diff = Diffusion(ck["config"]["T"], device) if ck["config"]["model"] == "diffusion" else None
    return model, diff, ck["config"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["diffusion", "regressor"], default="diffusion")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N train samples")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overlap-weight", type=float, default=0.0, help="λ for the overlap loss")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing existing checkpoints")
    args = ap.parse_args()
    cfg = vars(args)
    out = args.out or f"runs/{args.model}"
    existing = [f for f in ("last.pt", "best.pt", "final.pt") if os.path.exists(os.path.join(out, f))]
    if existing and not args.overwrite:
        raise SystemExit(f"{out} already has {existing}; pass --overwrite or choose another --out")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = get_device(args.device)

    train = LayoutDataset(os.path.join(args.data, "train.pt"))
    if args.limit:
        train = Subset(train, range(min(args.limit, len(train))))
    val = LayoutDataset(os.path.join(args.data, "val.pt"))
    tl = DataLoader(train, batch_size=args.bs, shuffle=True, collate_fn=collate, drop_last=True)
    vl = DataLoader(val, batch_size=512, shuffle=False, collate_fn=collate)

    model = build_model(cfg).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={dev} model={args.model} params={n_params / 1e6:.2f}M train={len(train)}")

    diff = Diffusion(args.T, dev) if args.model == "diffusion" else None
    lam = args.overlap_weight

    def loss_fn(m, b, w):
        return diff.loss(m, b, overlap_weight=w) if diff else regression_loss(m, b, overlap_weight=w)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)

    def checkpoint(history):
        return {
            "config": cfg,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "history": history,
        }

    history, best = [], float("inf")
    for ep in range(args.epochs):
        model.train()
        t0, tot, fit, ovl, n = time.time(), 0.0, 0.0, 0.0, 0
        for batch in tl:
            b = to_device(batch, dev)
            loss, parts = loss_fn(model, b, lam)
            if not torch.isfinite(loss):
                torch.save(checkpoint(history), os.path.join(out, "crash.pt"))
                raise SystemExit(f"non-finite loss at epoch {ep + 1}: {parts}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - args.ema)
            tot += loss.item()
            fit += float(parts.get("eps", parts.get("mse")))
            ovl += float(parts.get("overlap", 0.0))
            n += 1

        # Val: fixed noise/timesteps so it is comparable across epochs (training RNG
        # untouched). The overlap term is always measured (weight 1) for diagnostics,
        # but best.pt is chosen on the fit loss alone so all runs are comparable.
        fork_kw = {} if dev.type == "cpu" else {"device_type": dev.type}
        with torch.no_grad(), torch.random.fork_rng(**fork_kw):
            torch.manual_seed(1234)
            vfit, vovl, vn = 0.0, 0.0, 0
            for batch in vl:
                b = to_device(batch, dev)
                _, parts = loss_fn(ema, b, 1.0)
                vfit += float(parts.get("eps", parts.get("mse")))
                vovl += float(parts["overlap"])
                vn += 1
        rec = {
            "epoch": ep + 1,
            "train": tot / n,
            "train_fit": fit / n,
            "train_overlap": ovl / n,
            "val": vfit / vn,
            "val_overlap": vovl / vn,
            "sec": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(json.dumps(rec), flush=True)
        ck = checkpoint(history)
        torch.save(ck, os.path.join(out, "last.pt"))
        if rec["val"] < best:
            best = rec["val"]
            torch.save(ck, os.path.join(out, "best.pt"))
    torch.save(checkpoint(history), os.path.join(out, "final.pt"))
    print(f"saved {out}/final.pt (best val {best:.5f})", flush=True)

if __name__ == "__main__":
    main()
