# PyroVision AI

Real-time fire and smoke detection with YOLOv8. The project is being built and
verified one stage at a time; only the dataset-preparation stage is implemented
at present.

## Step 1 — dataset preparation

The selected source is the **D-Fire** dataset. It contains two annotated object
classes (`smoke`, `fire`) and includes normal/negative images. In YOLO object
detection, a negative image has an empty `.txt` label; `normal` is therefore not
modeled as a bounding-box class.

Expected local layout after downloading/extracting the source:

```text
data/raw/dfire/
  .../images/
  .../labels/
```

Prepare a fresh deterministic 70/20/10 split, stratified across negative,
smoke-only, fire-only, and smoke+fire images:

```powershell
python scripts/prepare_dataset.py `
  --source data/raw/dfire `
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

Source: [D-Fire dataset](https://github.com/gaia-solutions-on-demand/DFireDataset)

