"""Render layouts from their evaluated NURBS curves; overlaps highlighted in red."""
from __future__ import annotations

import sys

import matplotlib

if "ipykernel" not in sys.modules:  # headless for CLI scripts; keep inline plots in Jupyter
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import CIRCLE
from .geometry import ROOM, to_shapely
from .nurbs import evaluate, layout_to_nurbs

COLORS = {CIRCLE: "#2a7ab9", 1: "#d08a1f"}


def plot_layout(ax, types, sizes, poses, title: str | None = None):
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "k-", lw=1.5)
    u = np.linspace(0, 1, 200)
    for t, crv in zip(types, layout_to_nurbs(types, sizes, poses)):
        pts = evaluate(crv, u)
        ax.fill(pts[:, 0], pts[:, 1], color=COLORS[int(t)], alpha=0.35, lw=0)
        ax.plot(pts[:, 0], pts[:, 1], color=COLORS[int(t)], lw=1.2)
        ax.plot(crv.ctrl[:, 0], crv.ctrl[:, 1], ".", color="k", ms=1.5)
    geoms = [to_shapely(t, s, *p) for t, s, p in zip(types, sizes, poses)]
    for i in range(len(geoms)):
        for j in range(i + 1, len(geoms)):
            inter = geoms[i].intersection(geoms[j])
            if inter.area > 1e-6:
                _fill_geom(ax, inter)
        out = geoms[i].difference(ROOM)
        if out.area > 1e-6:
            _fill_geom(ax, out)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def _fill_geom(ax, g):
    for poly in getattr(g, "geoms", [g]):
        if hasattr(poly, "exterior"):
            x, y = poly.exterior.xy
            ax.fill(x, y, color="red", alpha=0.8, lw=0)


def plot_grid(layouts, path, titles=None, ncols: int = 4):
    """layouts: list of (types, sizes, poses) numpy arrays."""
    n = len(layouts)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for k, ax in enumerate(axes):
        if k < n:
            plot_layout(ax, *layouts[k], title=titles[k] if titles else None)
        else:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
