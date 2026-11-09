# Object list → 2D room layout → NURBS CAD

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SNMS95/CAD_LAYOUT_PRED/blob/main/notebooks/results.ipynb)

## Run it

Click the **Open In Colab** badge above and choose **Runtime → Run all**. The notebook installs the package, loads the trained model and shows the results. No local setup is needed. Everything it reads (datasets, trained model, training curves, metrics) is in [`notebooks/data/`](notebooks/data).

## What it does

**Input:** a list of objects, each a circle (radius) or a square (side length).
**Output:** a CAD model (NURBS curves, exportable to DXF/JSON) with those objects placed in a 1 × 1 room. No two objects overlap, and all of them stay inside the room.

Many layouts are valid for the same list, so the model is generative: each run gives a different one.

## How it works

1. A graph transformer reads the object list (any length) and predicts a **pose** (x, y, angle) for each object, using a diffusion model.
2. A fixed decoder turns each pose into exact NURBS curves.
3. An overlap penalty keeps layouts valid. It is used as an extra training loss. It can also be used at sampling time and as a final cleanup step.

## Results

% of 1000 generated layouts that are fully valid (no overlap, inside the room):

| model | test (2–8 objects) | unseen (9–12 objects) |
|---|---|---|
| diffusion | 75.8 | 17.5 |
| diffusion + overlap loss | 93.6 | 60.9 |
| diffusion + overlap loss + guidance + refine | 100 | 100 |
| regressor (no diffusion) | 0 | 0 |

The regressor always places every object at the room centre, so it is not a useful baseline. The overlap loss raises raw validity a lot while keeping the layouts diverse.

## Run locally (optional)

```bash
pip install -e .
echo '[{"type": "circle", "size": 0.1}, {"type": "square", "size": 0.2}]' > objects.json
python -m cadlayout.sample --ckpt notebooks/data/diffusion_ovl1_best.pt --objects objects.json --k 8 --out outputs/sample
```

For circles `size` is the radius. For squares it is the side length.

