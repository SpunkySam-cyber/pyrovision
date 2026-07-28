# PyroVision project log

This log records verified stage gates and important decisions. Generated data,
weights, and large run artifacts remain local; reproducible configuration and
summary metrics are versioned in Git.

## Step 1 — dataset preparation

Status: **complete**

- Source: Kaggle mirror of D-Fire
- Classes: `smoke` (0), `fire` (1)
- Total images: 21,527
- Split seed: 42
- Train/validation/test: 15,068 / 4,306 / 2,153
- Raw annotation cleanup: 379 boxes clipped and 18 degenerate boxes dropped
- Final verification: passed with 0 errors and no cross-split content leakage

## Step 2 — YOLO11 training

Status: **in progress — full training pending**

### 2026-07-28 environment audit

- Selected model: COCO-pretrained YOLO11s (`yolo11s.pt`)
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU
- VRAM: 6,141 MiB total; 5,641 MiB free during audit
- GPU driver: 577.05
- Driver-advertised CUDA compatibility: 12.9
- Compute capability: 8.9
- System RAM: 15.73 GiB total; 1.97 GiB available during audit
- Storage: 84.81 GiB free on the project drive
- Python: 3.11.4
- Existing global PyTorch: 2.8.0+cpu (CUDA unavailable)
- Existing global torchvision: 0.23.0+cpu
- Existing global Ultralytics: 8.4.37

Decision: create an isolated environment with a CUDA-enabled PyTorch build
before baseline validation or training. Use a conservative explicit batch size
and low worker count because the laptop has 6 GB VRAM and available system RAM
was low at audit time.

### 2026-07-28 CUDA environment verification

- Environment: `.venv`
- PyTorch: 2.8.0+cu129
- torchvision: 0.23.0+cu129
- CUDA runtime: 12.9
- cuDNN: 9.10.2
- Ultralytics: 8.4.37
- CUDA matrix-operation test: passed on the RTX 4050
- Dependency consistency (`pip check`): passed
- Dataset loading through Ultralytics: passed, 2 classes and 3 splits
- Automated project tests: 2 passed

### 2026-07-28 pre-fine-tuning validation baseline

- Validation images: 4,306 (test split untouched)
- Precision: 0.003907
- Recall: 0.006535
- mAP50: 0.000180
- mAP50–95: 0.0000645
- Inference: 4.46 ms/image on the validation run

The unadapted COCO checkpoint maps IDs 0 and 1 to `person` and `bicycle`, while
D-Fire maps those IDs to `smoke` and `fire`. The near-zero result is retained as
an explicit pre-fine-tuning ID-alignment baseline, not a claim that COCO
pretraining semantically detects hazards without adaptation.

### 2026-07-28 smoke-test attempt 1 — rejected

- Configuration: 2 epochs, batch 4, `fraction=0.02`
- GPU pipeline, checkpoints, and logging: operational
- Peak allocated CUDA memory: 1.20 GiB
- Gate result: rejected

Ultralytics applied `fraction` after filename sorting, selecting 300 negative
images and only one positive image. The run therefore did not meaningfully
exercise box regression. Its artifacts and metrics are retained as a failed
experiment. The replacement gate uses a deterministic 400-image subset
stratified by D-Fire image category.

### 2026-07-28 smoke-test attempt 2 — accepted

- Subset: 400 images, seed 42
- Categories: 183 negative, 109 smoke-only, 22 fire-only, 86 smoke+fire
- Epochs / batch: 2 / 4
- Peak allocated CUDA memory: 1.21 GiB
- Final train losses: box 2.127, class 3.087, DFL 1.940
- Best-checkpoint validation: P 0.2172, R 0.2069, mAP50 0.1274,
  mAP50–95 0.04056
- Checkpoints: best and last created and SHA-256 verified
- Gate result: accepted

### 2026-07-28 input immutability correction

Ultralytics' scan found source JPEGs without an end-of-image marker and repaired
them in place. A complete source audit found 91 such files. Dataset preparation
now performs the same deterministic repair in the processed copy, records the
count, and the verifier rejects missing markers. The raw archive remains
unchanged. The processed split must be rebuilt and re-verified before full
training. That rebuild completed on 2026-07-28 and passed strict verification
across all 21,527 images with zero errors and no cross-split content leakage.
The full-training input gate is accepted.
