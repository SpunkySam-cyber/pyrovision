# YOLO11s baseline experiment

Experiment ID: `yolo11s_baseline`

Status: **complete — epoch 54 selected from 70 completed epochs**

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

| Checkpoint | Epoch | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected `best.pt` | 54 | 0.7905 | 0.7118 | 0.7893 | 0.4669 |
| `last.pt` | 70 | 0.7805 | 0.7231 | 0.7865 | 0.4640 |

Selected-checkpoint per-class validation:

| Class | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| Smoke | 0.8407 | 0.7833 | 0.8572 | 0.5421 |
| Fire | 0.7403 | 0.6404 | 0.7215 | 0.3917 |

Absolute improvement over the pre-fine-tuning target-ID baseline:

- Precision: +0.7866
- Recall: +0.7053
- mAP50: +0.7892
- mAP50–95: +0.4669

The last checkpoint has 1.12 percentage points more recall but lower precision
and lower mAP50–95. Epoch 54 is selected because mAP50–95 is the primary model
selection metric. The plateau from epochs 55–70 does not justify additional
epochs with the same configuration.

Checkpoint integrity:

- `best.pt` SHA-256: `21812ec7917bda5ad004fc085ba6a9d8ee1b375c95db2efe754463fc430d28c3`
- `last.pt` SHA-256: `da284cd76bd7c8150a45b8f5f89b22f656e6356b4282e5313445b8be0dfcceaf`

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
| 2026-07-29 | Final training decision | Stop after 70 epochs; validation plateaued after epoch 54 |
| 2026-07-29 | Explicit best/last validation | Epoch 54 selected with P 0.7905, R 0.7118, mAP50 0.7893, mAP50–95 0.4669 |
