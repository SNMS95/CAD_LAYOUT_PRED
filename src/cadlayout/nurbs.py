"""NURBS templates for circles/squares, pose -> NURBS decoder, evaluation and export.

The network never predicts NURBS parameters. A circle/square NURBS is an exact,
deterministic function of (type, size, x, y, theta): a unit template whose control
points are scaled, rotated and translated (NURBS are invariant under affine maps of
their control points). Because the knot vectors are fixed per template, the basis
functions at a fixed set of parameter values are constants, so evaluation in torch
is a cheap, differentiable matrix product.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from . import CIRCLE, SQUARE, TYPE_NAMES


@dataclass(frozen=True)
class NurbsCurve:
    degree: int
    ctrl: np.ndarray  # (n, 2)
    weights: np.ndarray  # (n,)
    knots: np.ndarray  # (n + degree + 1,)

    def to_dict(self) -> dict:
        return {
            "degree": self.degree,
            "control_points": self.ctrl.tolist(),
            "weights": self.weights.tolist(),
            "knots": self.knots.tolist(),
        }


_W = math.sqrt(2.0) / 2.0

# Unit circle (radius 1): degree-2 rational, 9 control points on the bounding square.
CIRCLE_TEMPLATE = NurbsCurve(
    degree=2,
    ctrl=np.array(
        [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]],
        dtype=np.float64,
    ),
    weights=np.array([1, _W, 1, _W, 1, _W, 1, _W, 1], dtype=np.float64),
    knots=np.array([0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1], dtype=np.float64),
)

# Unit square (side 1), centred at origin: degree-1 closed polyline, 5 control points.
SQUARE_TEMPLATE = NurbsCurve(
    degree=1,
    ctrl=np.array(
        [[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]], dtype=np.float64
    ),
    weights=np.ones(5, dtype=np.float64),
    knots=np.array([0, 0, 0.25, 0.5, 0.75, 1, 1], dtype=np.float64),
)

TEMPLATES = {CIRCLE: CIRCLE_TEMPLATE, SQUARE: SQUARE_TEMPLATE}


def bspline_basis(knots: np.ndarray, degree: int, u: np.ndarray) -> np.ndarray:
    """Cox-de Boor recursion. Returns basis matrix of shape (len(u), n_ctrl)."""
    knots = np.asarray(knots, dtype=np.float64)
    u = np.atleast_1d(np.asarray(u, dtype=np.float64))
    n_spans = len(knots) - 1
    N = np.zeros((len(u), n_spans))
    for i in range(n_spans):
        N[:, i] = (knots[i] <= u) & (u < knots[i + 1])
    # u == last knot belongs to the last non-degenerate span
    last = max(i for i in range(n_spans) if knots[i] < knots[i + 1])
    N[u >= knots[-1], last] = 1.0
    for k in range(1, degree + 1):
        Nk = np.zeros((len(u), n_spans - k))
        for i in range(n_spans - k):
            d1 = knots[i + k] - knots[i]
            d2 = knots[i + k + 1] - knots[i + 1]
            a = (u - knots[i]) / d1 * N[:, i] if d1 > 0 else 0.0
            b = (knots[i + k + 1] - u) / d2 * N[:, i + 1] if d2 > 0 else 0.0
            Nk[:, i] = a + b
        N = Nk
    return N


def evaluate(curve: NurbsCurve, u: np.ndarray) -> np.ndarray:
    """Evaluate a rational curve at parameters u -> (len(u), 2)."""
    B = bspline_basis(curve.knots, curve.degree, u) * curve.weights  # (M, n)
    return (B @ curve.ctrl) / B.sum(axis=1, keepdims=True)


def pose_to_nurbs(typ: int, size: float, x: float, y: float, theta: float) -> NurbsCurve:
    """Deterministic decoder: place the unit template (scale=size, rotate, translate)."""
    t = TEMPLATES[int(typ)]
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    ctrl = (size * t.ctrl) @ R.T + np.array([x, y])
    return NurbsCurve(t.degree, ctrl, t.weights.copy(), t.knots.copy())


def layout_to_nurbs(types, sizes, poses) -> list[NurbsCurve]:
    """poses: (N, 3) array of (x, y, theta) in room coords."""
    return [pose_to_nurbs(t, s, *p) for t, s, p in zip(types, sizes, poses)]


# --------------------------------------------------------------------------- torch


@lru_cache(maxsize=None)
def _rational_basis(typ: int, m: int) -> np.ndarray:
    """Normalised rational basis R (m, n) at u = k/m, k = 0..m-1 (closed curve, no dup)."""
    t = TEMPLATES[typ]
    u = np.arange(m) / m
    B = bspline_basis(t.knots, t.degree, u) * t.weights
    return B / B.sum(axis=1, keepdims=True)


def boundary_points_torch(
    types: torch.Tensor, sizes: torch.Tensor, xy: torch.Tensor, theta: torch.Tensor, m: int = 64
) -> torch.Tensor:
    """Differentiable boundary samples of each shape from its NURBS.

    types (B, N) long, sizes (B, N), xy (B, N, 2) room coords, theta (B, N).
    Returns (B, N, m, 2). m should be divisible by 4 so square corners are included.
    """
    dev, dt = xy.device, xy.dtype
    out = None
    for typ in (CIRCLE, SQUARE):
        R = torch.as_tensor(_rational_basis(typ, m), dtype=dt, device=dev)  # (m, n)
        P = torch.as_tensor(TEMPLATES[typ].ctrl, dtype=dt, device=dev)  # (n, 2)
        local = R @ P  # (m, 2) template points (affine invariance => transform after)
        c, s = torch.cos(theta), torch.sin(theta)  # (B, N)
        lx, ly = local[:, 0], local[:, 1]
        px = sizes[..., None] * (c[..., None] * lx - s[..., None] * ly) + xy[..., 0:1]
        py = sizes[..., None] * (s[..., None] * lx + c[..., None] * ly) + xy[..., 1:2]
        pts = torch.stack([px, py], dim=-1)  # (B, N, m, 2)
        sel = (types == typ)[..., None, None]
        out = pts if out is None else torch.where(sel, pts, out)
    return out


# --------------------------------------------------------------------------- export


def export_json(path, types, sizes, poses) -> None:
    objs = []
    for t, s, p, crv in zip(types, sizes, poses, layout_to_nurbs(types, sizes, poses)):
        objs.append(
            {
                "type": TYPE_NAMES[int(t)],
                "size": float(s),
                "pose": {"x": float(p[0]), "y": float(p[1]), "theta": float(p[2])},
                "nurbs": crv.to_dict(),
            }
        )
    with open(path, "w") as f:
        json.dump({"room": [[0, 0], [1, 1]], "objects": objs}, f, indent=2)


def export_dxf(path, types, sizes, poses, scale: float = 1000.0) -> None:
    """Write room boundary + one rational SPLINE entity per object (units: scale * room)."""
    import ezdxf
    from ezdxf.math import BSpline

    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    doc.layers.add("ROOM", color=7)
    doc.layers.add("CIRCLES", color=1)
    doc.layers.add("SQUARES", color=5)
    msp.add_lwpolyline(
        [(0, 0), (scale, 0), (scale, scale), (0, scale)], close=True, dxfattribs={"layer": "ROOM"}
    )
    for t, crv in zip(types, layout_to_nurbs(types, sizes, poses)):
        bs = BSpline(
            [(scale * x, scale * y, 0.0) for x, y in crv.ctrl],
            order=crv.degree + 1,
            knots=crv.knots.tolist(),
            weights=crv.weights.tolist(),
        )
        layer = "CIRCLES" if int(t) == CIRCLE else "SQUARES"
        sp = msp.add_spline(dxfattribs={"layer": layer})
        sp.apply_construction_tool(bs)
    doc.saveas(path)
