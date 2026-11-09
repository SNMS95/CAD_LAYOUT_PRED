# cad_pred_2d: object set → 2D room layout → NURBS CAD

A prototype generative surrogate. The input is a variable-length list of objects (circles of radius r and squares of side s, in a unit-square room). The output is a valid arrangement of those objects, exported as exact NURBS curves (DXF/JSON).

## Idea
- **NURBS parameters are not predicted directly.** A circle's or square's NURBS is an exact function of (size, x, y, θ). The network predicts a **pose per object**, and a fixed, differentiable decoder transforms a unit NURBS template:
  - circle: degree 2, 9 weighted control points;
  - square: degree 1, 5 control points.
- **Variable N:** `SetGraphDenoiser` is a graph transformer on the complete graph of objects. It uses edge features (relative position and an overlap gap) as an attention bias and has no positional encoding, so it is permutation-equivariant and works for any N.
- **Many valid layouts per input** → a **diffusion model** over poses `(2x−1, 2y−1, cos4θ, sin4θ)`. The 4θ encodes the 90° symmetry of a square. A deterministic regressor with the same backbone is the baseline; it collapses toward the room centre.
- **Constraints:** a differentiable overlap/out-of-room penalty uses NURBS boundary samples and analytic SDFs. It is applied as sampling-time guidance and in an optional post-hoc refinement. Exact validity is checked with shapely.

## Pipeline
```
objects (type, size) ─► [ML] SetGraphDenoiser + DDIM (50 steps) ─► poses
     ─► guidance / refine (penalty optimisation) ─► pose→NURBS decoder ─► DXF / JSON
```

## Usage
```bash
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -e .   # + shapely, ezdxf
.venv/bin/python -m pytest -q
.venv/bin/python -m cadlayout.data --out data                      # 50k/5k/5k + 2k OOD (N=9..12)
.venv/bin/python -m cadlayout.train --model diffusion --epochs 100 --out runs/diffusion
.venv/bin/python -m cadlayout.train --model regressor --epochs 40  --out runs/regressor
.venv/bin/python -m cadlayout.evaluate --split test     --out outputs/eval_test
.venv/bin/python -m cadlayout.evaluate --split test_ood --out outputs/eval_ood
.venv/bin/python -m cadlayout.sample --objects objs.json --k 8 --out outputs/sample
```
`objs.json` is a list of objects: `[{"type": "circle", "size": 0.1}, {"type": "square", "size": 0.2}]`. For circles `size` is the radius; for squares it is the side length.

## Results (1000 test layouts each; ~25 min diffusion training on Apple MPS)
% of layouts that are exactly valid (shapely check: no overlap, inside the room).

| method | test (N=2–8) | OOD (N=9–12, unseen) |
|---|---|---|
| diffusion, raw | 75.8 | 17.5 |
| diffusion + guidance (λ=0.05) | 96.3 | 76.7 |
| diffusion + guidance + refine | **100.0** | **100.0** |
| regressor | 0.0 | 0.0 |
| regressor + refine | 89.1 | 12.0 |

- **Diversity** = mean pairwise centre distance over 8 samples of the same input:

  | source | diversity |
  |---|---|
  | diffusion | 0.433 |
  | generator reference | 0.443 |
  | regressor | 0 (deterministic) |

- **Mean object distance to the room centre:**

  | source | distance |
  |---|---|
  | ground truth | 0.328 |
  | diffusion | 0.316 |
  | regressor | 0.001 (every object at the centre) |

## Experiment: overlap penalty as a training loss (end-to-end)
The same differentiable penalty is added to the training loss. It is applied to the network's implied clean layout x̂₀ and weighted by ᾱ_t, so it acts mostly at low noise:
`loss = MSE(ε̂, ε) + λ · mean[ᾱ_t · P(x̂₀)]`.
Flag: `--overlap-weight λ`. All models and results are listed in `runs/SUMMARY.md`.

**Raw validity (network only, no guidance/refinement), % of 1000 layouts:**

| model | test (N=2–8) | OOD (N=9–12) | diversity |
|---|---|---|---|
| diffusion, λ=0 (baseline) | 75.8 | 17.5 | 0.433 |
| diffusion, λ=0.1 | 92.1 | 49.8 | 0.452 |
| **diffusion, λ=1** | **93.6** | **60.9** | 0.454 |
| diffusion, λ=10 | 92.8 | 56.5 | 0.457 |
| regressor, λ=1 | 34.1 | 0.7 | 0 |

Findings:
- **The loss almost eliminates overlaps.** Layouts failing because of overlaps:
  - normal test: 22.5% → 1.5% (λ=1);
  - OOD: 81% → 16%.
- **Diversity is kept.** The denoising (fit) loss worsens only slightly: 0.288 → 0.293.
- **Remaining failures are mostly objects poking slightly through a wall** (median protrusion ≈ 0.5% of the room width). The penalty pushes objects apart and therefore outward:
  - mean distance to the room centre rises from 0.316 to 0.337, vs 0.328 for the ground truth;
  - wall-only failures rise from 1.7% to 4.9% (normal test).
- **Larger λ trades a worse fit loss for fewer overlaps** but more wall failures. λ=1 is the best balance.
- **Regressor + loss:** its objects no longer pile up at the centre, but it is only 34% valid and has zero diversity. A deterministic model remains the wrong tool.
- With guidance and refinement, every model reaches 100% on test and ≥99.8% on OOD.

## Future ideas
- [`docs/ideas/three_stage_pipeline.md`](docs/ideas/three_stage_pipeline.md): a learned shape network (object → NURBS), then shape-aware arrangement, then deterministic room-CAD assembly, all trained end-to-end.
- [`docs/ideas/differentiable_operations_tree.md`](docs/ideas/differentiable_operations_tree.md): editable parametric CAD. The CAD operations tree is treated as a differentiable program, enabling validity-preserving edit propagation, networks that output the editable tree, and GNNs on the tree.

## Layout
| file | role |
|---|---|
| `src/cadlayout/nurbs.py` | templates, Cox–de Boor evaluator (numpy and torch), pose→NURBS, DXF/JSON export |
| `src/cadlayout/geometry.py` | shapely validity checks, torch SDFs, differentiable layout penalty |
| `src/cadlayout/data.py` | rejection-sampling generator, pose encoding, dataset/collate |
| `src/cadlayout/model.py` | `SetGraphDenoiser` (diffusion denoiser or regressor) |
| `src/cadlayout/diffusion.py` | cosine DDPM loss, DDIM sampler with penalty guidance |
| `src/cadlayout/refine.py` | post-hoc Adam projection onto the constraints |
| `src/cadlayout/{train,evaluate,sample,viz}.py` | CLI scripts and plotting |
