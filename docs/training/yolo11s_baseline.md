# YOLO11s baseline experiment

Experiment ID: `yolo11s_baseline`

Status: **full training resumed from epoch 50; next review at epoch 60**

## Objective

Fine-tune a COCO-pretrained YOLO11s detector on the verified D-Fire training
split, while measuring the same validation metrics before and after training.
The held-out test split is reserved for Step 3.

## Model and dataset

- Initial checkpoint: `yolo11s.pt`
- Dataset configuration: `configs/dfire.yaml`
- Classes: smoke, fire
- Image size: 640 px
- Random seed: 42

## Required metric checkpoints

### Before fine-tuning

- Precision: 0.003907
- Recall: 0.006535
- mAP50: 0.000180
- mAP50–95: 0.0000645
- Inference: 4.46 ms/image

The raw COCO checkpoint names target IDs 0 and 1 as `person` and `bicycle`;
D-Fire uses those IDs for `smoke` and `fire`. These values intentionally record
the unadapted checkpoint's target-ID alignment before the detection head is
fine-tuned. Per-class reporting uses the D-Fire target names.

### During training

Ultralytics `results.csv` retains per-epoch box loss, classification loss, DFL
loss, precision, recall, mAP50, mAP50–95, and learning rates. The experiment
JSON is refreshed after every completed epoch, so partial progress survives an
interruption. It also records the start environment, failures, total duration,
peak VRAM, best epoch, checkpoints, and the final environment.

### After fine-tuning

- Best-checkpoint precision: pending
- Best-checkpoint recall: pending
- Best-checkpoint mAP50: pending
- Best-checkpoint mAP50–95: pending
- Last-checkpoint metrics: pending
- Before/after deltas: pending

## Environment audit

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB)
- Driver: 577.05
- Python: 3.11.4
- CUDA-enabled PyTorch: 2.8.0+cu129 (verified)
- CUDA runtime / cuDNN: 12.9 / 9.10.2
- Ultralytics: 8.4.37

## Experiment history

| Date | Event | Outcome |
| --- | --- | --- |
| 2026-07-28 | Initial hardware/software audit | GPU present; global PyTorch is CPU-only |
| 2026-07-28 | Isolated CUDA environment setup | CUDA tensor operation, dependencies, dataset loader, and tests passed |
| 2026-07-28 | Pre-fine-tuning validation | P 0.003907, R 0.006535, mAP50 0.000180, mAP50–95 0.0000645 |
| 2026-07-28 | Smoke-test attempt 1 | Rejected: filename-ordered fraction contained 300 negatives and 1 positive |
| 2026-07-28 | Stratified smoke-test attempt 2 | Accepted: finite losses; P 0.2172, R 0.2069, mAP50 0.1274, mAP50–95 0.04056 |
| 2026-07-28 | Rebuilt processed dataset | Repaired 91 JPEGs in the processed copy; strict verification passed with 0 errors |
| 2026-07-29 | User-requested pause after epoch 50 | Best epoch 49: P 0.7950, R 0.6981, mAP50 0.7842, mAP50–95 0.4643; resumable state verified |
| 2026-07-29 | Continue-training decision | Resume at epoch 51 and pause safely after epoch 60 |
