# Saved models

Valid % = exact shapely check (no overlap, inside room) on 1000 layouts. *Raw* = network only (no guidance, no refinement).
Each folder has `best.pt` (lowest val fit loss; its EMA weights are used for sampling) and `last.pt`; the overlap-loss runs also have `final.pt` (end of training, incl. optimizer state).

| name | folder | model | λ (overlap loss) | epochs | best epoch | val fit | val overlap | raw valid test | raw valid OOD (N=9–12) | full pipeline valid test / OOD | diversity (raw, test) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusion | `runs/diffusion/` | diffusion | 0 | 100 | 98 | 0.2884 | – | 75.8 | 17.5 | 100.0 / 100.0 | 0.433 |
| diff_ovl0.1 | `runs/diffusion_ovl0.1/` | diffusion | 0.1 | 100 | 96 | 0.2904 | 0.0226 | 92.1 | 49.8 | 100.0 / 100.0 | 0.452 |
| diff_ovl1 | `runs/diffusion_ovl1/` | diffusion | 1 | 100 | 96 | 0.2928 | 0.0009 | 93.6 | 60.9 | 100.0 / 100.0 | 0.454 |
| diff_ovl10 | `runs/diffusion_ovl10/` | diffusion | 10 | 100 | 100 | 0.2947 | 0.0001 | 92.8 | 56.5 | 100.0 / 100.0 | 0.457 |
| regressor | `runs/regressor/` | regressor | 0 | 40 | 36 | 0.3283 | – | 0.0 | 0.0 | 89.1 / 12.0 | 0.000 |
| regressor_ovl1 | `runs/regressor_ovl1/` | regressor | 1 | 40 | 40 | 0.3595 | 0.0017 | 34.1 | 0.7 | 100.0 / 99.8 | 0.000 |

Generator reference diversity (test): 0.443

Full tables and pictures: `outputs/ovl_eval_test/`, `outputs/ovl_eval_test_ood/` (`table.md`, `metrics.json`, `*.png`).
