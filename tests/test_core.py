import math

import numpy as np
import pytest
import torch

from cadlayout import CIRCLE, SQUARE
from cadlayout.data import (
    GenConfig,
    collate,
    decode_pose,
    encode_pose,
    generate,
    LayoutDataset,
)
from cadlayout.diffusion import Diffusion
from cadlayout.geometry import layout_penalty, layout_violations, sdf_box, sdf_circle
from cadlayout.model import SetGraphDenoiser
from cadlayout.nurbs import (
    _rational_basis,
    boundary_points_torch,
    evaluate,
    export_dxf,
    pose_to_nurbs,
)


# ------------------------------------------------------------------ NURBS


def test_circle_nurbs_is_exact():
    crv = pose_to_nurbs(CIRCLE, 0.13, 0.4, 0.6, 1.234)
    pts = evaluate(crv, np.linspace(0, 1, 1001))
    r = np.linalg.norm(pts - [0.4, 0.6], axis=1)
    assert np.abs(r - 0.13).max() < 1e-12


def test_square_nurbs_vertices_are_rotated_corners():
    s, x, y, th = 0.2, 0.5, 0.3, 0.4
    crv = pose_to_nurbs(SQUARE, s, x, y, th)
    pts = evaluate(crv, np.array([0.0, 0.25, 0.5, 0.75]))
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    corners = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * s / 2 @ R.T + [x, y]
    assert np.allclose(pts, corners, atol=1e-12)
    # midpoint of an edge lies on the straight segment
    mid = evaluate(crv, np.array([0.125]))[0]
    assert np.allclose(mid, (corners[0] + corners[1]) / 2, atol=1e-12)


def test_basis_partition_of_unity():
    for typ in (CIRCLE, SQUARE):
        assert np.allclose(_rational_basis(typ, 64).sum(1), 1.0)


def test_torch_and_numpy_evaluators_agree():
    types = torch.tensor([[CIRCLE, SQUARE]])
    sizes = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
    xy = torch.tensor([[[0.3, 0.4], [0.7, 0.6]]], dtype=torch.float64)
    th = torch.tensor([[0.0, 0.3]], dtype=torch.float64)
    pts = boundary_points_torch(types, sizes, xy, th, m=64)[0].numpy()
    u = np.arange(64) / 64
    for i, (t, s, p, a) in enumerate([(CIRCLE, 0.1, (0.3, 0.4), 0.0), (SQUARE, 0.2, (0.7, 0.6), 0.3)]):
        assert np.allclose(pts[i], evaluate(pose_to_nurbs(t, s, *p, a), u), atol=1e-12)


def test_dxf_roundtrip(tmp_path):
    import ezdxf

    types = np.array([CIRCLE, SQUARE])
    sizes = np.array([0.1, 0.2])
    poses = np.array([[0.3, 0.3, 0.0], [0.7, 0.7, 0.5]])
    path = tmp_path / "x.dxf"
    export_dxf(str(path), types, sizes, poses, scale=1000.0)
    doc = ezdxf.readfile(str(path))
    splines = list(doc.modelspace().query("SPLINE"))
    assert len(splines) == 2
    circ = [s for s in splines if s.dxf.layer == "CIRCLES"][0]
    pts = np.array(list(circ.construction_tool().approximate(100)))[:, :2]
    assert np.abs(np.linalg.norm(pts - [300, 300], axis=1) - 100).max() < 1e-6


# ------------------------------------------------------------------ geometry


def test_sdf_signs():
    c = torch.tensor([0.5, 0.5])
    assert sdf_circle(torch.tensor([0.5, 0.5]), c, torch.tensor(0.1)) < 0
    assert sdf_circle(torch.tensor([0.7, 0.5]), c, torch.tensor(0.1)) > 0
    th, side = torch.tensor(math.pi / 4), torch.tensor(0.2)
    # rotated 45°: point at distance 0.13 along x is inside (corner reaches 0.1414)
    assert sdf_box(torch.tensor([0.63, 0.5]), c, th, side) < 0
    # axis aligned: same point is outside (half side 0.1)
    assert sdf_box(torch.tensor([0.63, 0.5]), c, torch.tensor(0.0), side) > 0
    assert torch.isclose(sdf_box(torch.tensor([0.65, 0.5]), c, torch.tensor(0.0), side), torch.tensor(0.05))


def test_penalty_zero_iff_valid_simple_cases():
    types = torch.tensor([[CIRCLE, SQUARE]])
    sizes = torch.tensor([[0.1, 0.2]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    th = torch.tensor([[0.0, 0.3]])
    ok = torch.tensor([[[0.2, 0.2], [0.7, 0.7]]])
    bad = torch.tensor([[[0.45, 0.5], [0.55, 0.5]]])
    assert layout_penalty(types, sizes, ok, th, mask).item() == 0.0
    assert layout_penalty(types, sizes, bad, th, mask).item() > 0.0
    oob = torch.tensor([[[0.05, 0.5], [0.7, 0.7]]])
    assert layout_penalty(types, sizes, oob, th, mask).item() > 0.0


def test_refine_separates_circle_inscribed_in_square():
    """Nested shapes: boundary-point penalty alone is ~0 with cancelling gradients."""
    from cadlayout.refine import refine

    types = torch.tensor([[CIRCLE, SQUARE]])
    sizes = torch.tensor([[0.04, 0.08]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    poses = torch.tensor([[[0.30, 0.30, 0.0], [0.301, 0.30, 0.2]]])
    assert layout_penalty(types, sizes, poses[..., :2], poses[..., 2], mask).item() > 0.01
    out = refine(types, sizes, poses, mask, steps=500)
    assert layout_violations(types[0].numpy(), sizes[0].double().numpy(), out[0].double().numpy())["valid"]


def test_violations_detect_overlap_and_oob():
    t, s = np.array([CIRCLE, SQUARE]), np.array([0.1, 0.2])
    assert layout_violations(t, s, np.array([[0.2, 0.2, 0], [0.7, 0.7, 0.3]]))["valid"]
    assert not layout_violations(t, s, np.array([[0.45, 0.5, 0], [0.55, 0.5, 0.3]]))["valid"]
    assert not layout_violations(t, s, np.array([[0.05, 0.5, 0], [0.7, 0.7, 0.3]]))["valid"]


# ------------------------------------------------------------------ data


@pytest.fixture(scope="module")
def small_data():
    return generate(40, GenConfig(), seed=7, workers=1)


def test_generated_layouts_are_valid(small_data):
    d = small_data
    for k in range(len(d["types"])):
        m = d["mask"][k]
        v = layout_violations(
            d["types"][k][m].numpy(), d["sizes"][k][m].numpy().astype(float), d["poses"][k][m].numpy().astype(float)
        )
        assert v["valid"], v


def test_pose_encoding_roundtrip_mod_90deg():
    types = torch.tensor([[SQUARE, CIRCLE]])
    poses = torch.tensor([[[0.3, 0.7, 1.2], [0.6, 0.2, 0.0]]])
    dec = decode_pose(encode_pose(poses, types))
    assert torch.allclose(dec[..., :2], poses[..., :2], atol=1e-6)
    # theta recovered modulo 90°
    d = (dec[0, 0, 2] - poses[0, 0, 2]) / (math.pi / 2)
    assert abs(d - torch.round(d)) < 1e-5
    # square at θ and θ + 90° encode identically
    p2 = poses.clone()
    p2[0, 0, 2] += math.pi / 2
    assert torch.allclose(encode_pose(poses, types), encode_pose(p2, types), atol=1e-6)


# ------------------------------------------------------------------ model


@pytest.mark.parametrize("diffusion", [True, False])
def test_model_permutation_equivariance_and_padding(small_data, diffusion):
    torch.manual_seed(0)
    model = SetGraphDenoiser(d=32, layers=2, heads=4, diffusion=diffusion).eval()
    # make the zero-initialised head/edge layers non-trivial for the test
    for p in model.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.1)
    b = collate([LayoutDataset(small_data)[i] for i in range(8)])
    x = encode_pose(b["poses"], b["types"]) + 0.1 * torch.randn(*b["types"].shape, 4)
    t = torch.randint(0, 500, (8,))
    kw = lambda bb, xx: dict(x_t=xx, t=t) if diffusion else {}
    out = model(b["types"], b["sizes"], b["mask"], **kw(b, x))

    # permute objects within each layout (only the real ones, padding stays at the end)
    perm = torch.arange(b["types"].shape[1]).repeat(8, 1)
    for k in range(8):
        n = int(b["mask"][k].sum())
        perm[k, :n] = torch.randperm(n)
    g = lambda a: torch.gather(a, 1, perm if a.dim() == 2 else perm[..., None].expand_as(a))
    out_p = model(g(b["types"]), g(b["sizes"]), g(b["mask"]), **kw(b, g(x)))
    m = b["mask"][..., None]
    assert torch.allclose((g(out) * m), (out_p * m), atol=1e-5)

    # extra padding must not change outputs of real nodes
    pad = lambda a: torch.cat([a, torch.zeros_like(a[:, :3])], 1)
    out_pad = model(pad(b["types"]), pad(b["sizes"]), pad(b["mask"]), **kw(b, pad(x)))
    assert torch.allclose(out_pad[:, : out.shape[1]] * m, out * m, atol=1e-5)


def test_diffusion_loss_and_sampling_shapes(small_data):
    torch.manual_seed(0)
    model = SetGraphDenoiser(d=32, layers=2, heads=4)
    diff = Diffusion(T=100)
    b = collate([LayoutDataset(small_data)[i] for i in range(8)])
    b["x0"] = encode_pose(b["poses"], b["types"])
    loss, parts = diff.loss(model, b)
    loss.backward()
    assert torch.isfinite(loss) and "overlap" not in parts
    poses = diff.sample(model.eval(), b["types"], b["sizes"], b["mask"], steps=5, guidance=0.02)
    assert poses.shape == (*b["types"].shape, 3)
    assert torch.isfinite(poses).all()
    assert (poses[..., 2][b["types"] == CIRCLE] == 0).all()


# ------------------------------------------------------------------ overlap loss


@pytest.mark.parametrize("kind", ["diffusion", "regressor"])
def test_overlap_loss_finite_with_gradients(small_data, kind):
    from cadlayout.train import regression_loss

    torch.manual_seed(0)
    model = SetGraphDenoiser(d=32, layers=2, heads=4, diffusion=kind == "diffusion")
    for p in model.parameters():  # non-trivial outputs so the penalty is active
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.1)
    b = collate([LayoutDataset(small_data)[i] for i in range(16)])
    b["x0"] = encode_pose(b["poses"], b["types"])
    if kind == "diffusion":
        loss, parts = Diffusion(T=100).loss(model, b, overlap_weight=1.0)
    else:
        loss, parts = regression_loss(model, b, overlap_weight=1.0)
    assert torch.isfinite(loss) and parts["overlap"] > 0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


def test_overlap_term_zero_for_valid_x0(small_data):
    """If the network predicted the noise perfectly, x̂0 = x0 (valid) -> penalty 0."""
    b = collate([LayoutDataset(small_data)[i] for i in range(16)])
    x0 = encode_pose(b["poses"], b["types"])
    pen = Diffusion().penalty_on_encoded(x0, b["types"], b["sizes"], b["mask"])
    # margin=1e-3 can make touching-close generator layouts slightly positive
    assert pen.max().item() < 1e-3
