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

Status: **complete**

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

### 2026-07-29 epoch-50 pause and continuation decision

- Training paused safely after epoch 50 with resumable optimizer, scheduler,
  scaler, and EMA state preserved in `last.pt`.
- Best checkpoint at the pause: epoch 49, precision 0.7950, recall 0.6981,
  mAP50 0.7842, and mAP50–95 0.4643.
- The validation trend was still improving slowly without clear overfitting.
- Decision: resume from epoch 51 and pause again after epoch 60 for review.
- The runner now supports `resume --stop-after-epoch N` and mirrors every
  resumed epoch into the same experiment history.

### 2026-07-29 training completion and finalization

- Training stopped after 70 completed epochs because the validation metric had
  plateaued; no epoch after 54 improved mAP50–95.
- Selected checkpoint: epoch 54 `best.pt`.
- Explicit best-checkpoint validation on 4,306 validation images: precision
  0.7905, recall 0.7118, mAP50 0.7893, mAP50–95 0.4669.
- Smoke: P 0.8407, R 0.7833, mAP50 0.8572, mAP50–95 0.5421.
- Fire: P 0.7403, R 0.6404, mAP50 0.7215, mAP50–95 0.3917.
- Epoch-70 `last.pt`: P 0.7805, R 0.7231, mAP50 0.7865,
  mAP50–95 0.4640.
- Best SHA-256: `21812ec7917bda5ad004fc085ba6a9d8ee1b375c95db2efe754463fc430d28c3`.
- Last SHA-256: `da284cd76bd7c8150a45b8f5f89b22f656e6356b4282e5313445b8be0dfcceaf`.
- The held-out test split remained untouched through Step 2.
- Gate result: accepted; Step 3 evaluation unlocked.

## Step 3 — held-out evaluation

Status: **complete**

### 2026-07-29 one-time test evaluation

- Selected checkpoint: epoch 54 `best.pt`, SHA-256 verified.
- Test split: 2,153 images and 2,710 instances; no training or model selection
  used this split.
- Overall: precision 0.7657, recall 0.6992, mAP50 0.7642, mAP50–95 0.4526.
- Smoke: P 0.8189, R 0.7806, mAP50 0.8368, mAP50–95 0.5378.
- Fire: P 0.7125, R 0.6178, mAP50 0.6916, mAP50–95 0.3675.
- Batched model inference: 4.30 ms/image.
- Generated artifacts: raw and normalized confusion matrices, PR, F1,
  precision-confidence and recall-confidence curves, and prediction batches.
- Deterministic sanity set: negative, smoke-only, fire-only, and combined cases
  all behaved as expected at confidence 0.25.
- Gate result: accepted for Step 4 local real-time inference; not approved for
  safety-critical deployment.

## Step 4 — local real-time inference

Status: **Milestone 1 complete; stopped before Milestone 2**

### 2026-07-31 Milestone 1 — reusable foundation

- Added an installable `src/pyrovision/` package; inference code does not import
  the training runner.
- Added a strict, versioned inference configuration with typed checkpoint,
  model, input, output, and device settings.
- Added automatic selected-checkpoint resolution with relocation fallback,
  streaming SHA-256 verification, and exact class-name/order validation.
- Reverified epoch-54 `best.pt` at SHA-256
  `21812ec7917bda5ad004fc085ba6a9d8ee1b375c95db2efe754463fc430d28c3`.
- Loaded checkpoint metadata and confirmed class order `smoke`, `fire`.
- Added deterministic `BoundingBox`, `Detection`, and `FrameResult` types that
  do not expose Ultralytics objects to downstream consumers.
- Added explicit `auto`, `cpu`, `cuda`, and `cuda:N` device resolution. The
  local gate selected `cuda:0` on the NVIDIA GeForce RTX 4050 Laptop GPU.
- Extracted the streaming file-hash helper for shared training/inference use;
  Steps 1–3 retain their existing behavior.
- Added seven foundation unit tests; all 13 project tests pass.
- Added ignore rules for local media, outputs, logs, and detection artifacts.
- Milestone gate: accepted. Image inference remains deliberately unimplemented
  until Milestone 2 is authorized.

### 2026-07-31 Milestone 2 — image inference

- Added `DetectorEngine` with verified checkpoint loading, exact class
  validation, explicit CPU/CUDA execution, FP16 policy, prediction locking, and
  framework-neutral results.
- Added global and per-class threshold filtering. YOLO is called at the lowest
  required candidate threshold before deterministic class-aware filtering.
- Added coordinate clipping, invalid-box rejection, and stable detection order.
- Added OpenCV annotation with labeled `smoke` and `fire` boxes and confidence.
- Added supported-image decoding, optional annotated-image output, atomic JSON
  output, and structured input/output errors.
- Added a thin CLI with device, confidence, per-class confidence, IoU, output,
  and save-policy overrides.
- Added six image inference tests. All 19 project tests pass.
- Real CPU FP32 and CUDA FP16 gates both passed on processed training image
  `AoF04009.jpg`; each found one smoke and three fire regions.
- Visually inspected the CUDA annotated image; all four boxes were plausible.
- Generated JPEG/JSON artifacts remain ignored under `outputs/milestone2/`.
- Machine-readable verification is versioned in
  `metrics/step4_milestone2.json`.
- Milestone gate: accepted. Video inference remains unimplemented pending
  Milestone 3 authorization.
