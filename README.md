# PyroVision AI

Real-time fire and smoke detection with YOLOv8. The project is being built and
verified one stage at a time. Dataset preparation is complete; model training
has not started.

## Step 1 — dataset preparation

The selected source is the **D-Fire** dataset. It contains two annotated object
classes (`smoke`, `fire`) and includes normal/negative images. In YOLO object
detection, a negative image has an empty `.txt` label; `normal` is therefore not
modeled as a bounding-box class.

The downloaded Kaggle source is stored locally at `archive/` and ignored by
Git. The prepared dataset is also ignored and has this layout:

```text
data/processed/dfire/
  images/{train,val,test}/
  labels/{train,val,test}/
```

Prepare a fresh deterministic 70/20/10 split, stratified across negative,
smoke-only, fire-only, and smoke+fire images:

```powershell
python scripts/prepare_dataset.py `
  --source archive `
  --output data/processed/dfire `
  --ratios 0.7 0.2 0.1 `
  --seed 42
```

Validate image decoding, YOLO labels, normalized box bounds, split isolation,
ratios, and category balance:

```powershell
python scripts/verify_dataset.py `
  --dataset data/processed/dfire `
  --report artifacts/dataset_verification.json
```

Run the tooling test:

```powershell
python -m unittest discover -s tests -v
```

Dataset files are intentionally ignored by Git. The preparation manifest and
verification report make the local build auditable and reproducible.

### Verified dataset results

The strict verification gate passed with **0 errors** after decoding and
content-hashing all 21,527 images, validating every YOLO annotation, checking
split isolation, and comparing category distributions.

| Split | Images | Ratio | Smoke images | Fire images | Negative images |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 15,068 | 69.996% | 7,367 | 4,075 | 6,886 |
| Validation | 4,306 | 20.003% | 2,105 | 1,165 | 1,968 |
| Test | 2,153 | 10.001% | 1,053 | 582 | 984 |

The source annotations contained boxes crossing normalized image boundaries
and a small number of zero-area boxes. Preparation preserves the raw archive,
clips 379 recoverable boxes, drops 18 degenerate boxes, and retains 26,539
valid boxes. The processed dataset contains 11,854 smoke boxes and 14,685 fire
boxes.

Source: [D-Fire dataset](https://github.com/gaia-solutions-on-demand/DFireDataset)
