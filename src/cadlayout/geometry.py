"""Exact (shapely) validity checks and differentiable (torch) constraint penalties.

Room is the unit square [0, 1]^2. Poses are (x, y, theta) in room coordinates.
"""
from __future__ import annotations

import math

import numpy as np
import shapely
import torch
from shapely import affinity
from shapely.geometry import Point, Polygon, box

from . import CIRCLE, SQUARE
from .nurbs import boundary_points_torch

ROOM = box(0.0, 0.0, 1.0, 1.0)
OVERLAP_TOL = 1e-6  # area (room = 1.0) below which an intersection counts as touching


def half_extent(typ: int, size: float, theta: float) -> float:
    """Half-width of the axis-aligned bounding box (same in x and y for both shapes)."""
    if typ == CIRCLE:
        return size
    return 0.5 * size * (abs(math.cos(theta)) + abs(math.sin(theta)))


def circumradius(types: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    return sizes * torch.where(types == CIRCLE, 1.0, 1.0 / math.sqrt(2.0)).to(sizes.dtype)


def to_shapely(typ: int, size: float, x: float, y: float, theta: float) -> Polygon:
    if int(typ) == CIRCLE:
        return Point(x, y).buffer(size, quad_segs=64)
    h = size / 2.0
    sq = box(-h, -h, h, h)
    sq = affinity.rotate(sq, theta, origin=(0, 0), use_radians=True)
    return affinity.translate(sq, x, y)


def layout_violations(types, sizes, poses) -> dict:
    """Exact violation measures for one layout (arrays of length N)."""
    geoms = [to_shapely(t, s, *p) for t, s, p in zip(types, sizes, poses)]
    overlap_area, n_pairs = 0.0, 0
    for i in range(len(geoms)):
        for j in range(i + 1, len(geoms)):
            if int(types[i]) == CIRCLE and int(types[j]) == CIRCLE:
                d = math.dist(poses[i][:2], poses[j][:2])
                if d >= sizes[i] + sizes[j]:
                    continue
            a = geoms[i].intersection(geoms[j]).area
            if a > OVERLAP_TOL:
                overlap_area += a
                n_pairs += 1
    oob_area = sum(g.difference(ROOM).area for g in geoms)
    return {
        "overlap_area": overlap_area,
        "overlap_pairs": n_pairs,
        "oob_area": oob_area,
        "valid": n_pairs == 0 and oob_area <= OVERLAP_TOL,
    }


def fits(geom: Polygon, placed: list[Polygon]) -> bool:
    if not placed:
        return True
    hits = shapely.intersects(geom, np.asarray(placed, dtype=object))
    return not bool(np.any(hits))


# --------------------------------------------------------------------------- torch SDFs


def sdf_circle(p: torch.Tensor, c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(p - c, dim=-1) - r


def sdf_box(p: torch.Tensor, c: torch.Tensor, theta: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    """Signed distance to a rotated square. p (..., 2); c (..., 2); theta, side (...)."""
    d = p - c
    ct, st = torch.cos(theta), torch.sin(theta)
    qx = ct * d[..., 0] + st * d[..., 1]  # rotate by -theta into the box frame
    qy = -st * d[..., 0] + ct * d[..., 1]
    q = torch.stack([qx.abs(), qy.abs()], dim=-1) - (side / 2.0)[..., None]
    outside = torch.linalg.norm(q.clamp(min=0.0), dim=-1)
    inside = q.max(dim=-1).values.clamp(max=0.0)
    return outside + inside


def sdf_shapes(p: torch.Tensor, types, sizes, xy, theta) -> torch.Tensor:
    """SDF of every shape j at every point. p (B, Np, 2) -> (B, Np, N)."""
    pp = p[:, :, None, :]  # (B, Np, 1, 2)
    c = xy[:, None, :, :]
    s = sizes[:, None, :]
    th = theta[:, None, :]
    d_circ = sdf_circle(pp, c, s)
    d_box = sdf_box(pp, c, th, s)
    return torch.where((types == CIRCLE)[:, None, :], d_circ, d_box)


def layout_penalty(
    types: torch.Tensor,
    sizes: torch.Tensor,
    xy: torch.Tensor,
    theta: torch.Tensor,
    mask: torch.Tensor,
    m: int = 64,
    margin: float = 1e-3,
) -> torch.Tensor:
    """Differentiable overlap + out-of-room penalty per layout -> (B,).

    Boundary points of each shape (from its NURBS) are tested against the SDFs of all
    other shapes; penetration depth is penalised. Circle-circle pairs also get the exact
    analytic term. All in room coordinates.
    """
    B, N = types.shape
    pts = boundary_points_torch(types, sizes, xy, theta, m)  # (B, N, m, 2)
    sd = sdf_shapes(pts.reshape(B, N * m, 2), types, sizes, xy, theta).reshape(B, N, m, N)
    pair = mask[:, :, None] & mask[:, None, :] & ~torch.eye(N, dtype=torch.bool, device=xy.device)
    depth = torch.relu(margin - sd)  # (B, Ni, m, Nj)
    overlap = (depth.mean(dim=2) * pair).sum(dim=(1, 2))

    # Inscribed-circle term for every pair (exact for circle-circle). Inscribed circles
    # overlapping => shapes overlap, so this never penalises a valid layout, and it
    # separates nested shapes (e.g. a circle inscribed in a square) where the
    # boundary-point term above is ~0 with cancelling gradients.
    inr = sizes * torch.where(types == CIRCLE, 1.0, 0.5).to(sizes.dtype)
    dist = torch.sqrt(((xy[:, :, None] - xy[:, None, :]) ** 2).sum(-1) + 1e-12)
    in_depth = torch.relu(inr[:, :, None] + inr[:, None, :] + margin - dist)
    overlap = overlap + 0.5 * (in_depth * pair).sum(dim=(1, 2))

    oob = (torch.relu(margin - pts) + torch.relu(pts - (1.0 - margin))).sum(-1).mean(-1)
    oob = (oob * mask).sum(dim=1)
    return overlap + oob
