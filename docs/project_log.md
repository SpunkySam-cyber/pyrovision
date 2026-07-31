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

### 2026-07-31 Milestone 3 — video inference

- Added supported-media classification and sequential OpenCV video decoding.
- Preserved original source frame indices and monotonic timestamps, with an FPS
  fallback for missing or unreliable container timestamps.
- Added optional frame skipping while adjusting output FPS to preserve playback
  duration.
- Added validated annotated video writing with explicit codec/container policy.
- Added interruption-safe, per-frame JSONL detection records and atomic run
  summaries.
- Added programmatic stop and Ctrl+C handling; both retain decodable partial
  videos, valid complete JSONL lines, released resources, and an interrupted
  summary.
- Shared the atomic JSON serializer between image and video outputs.
- Added six video tests. All 25 project tests pass.
- Built a 12-frame development MP4 from four processed training images covering
  negative, smoke, fire, and combined categories; the held-out test set was not
  used.
- Real CUDA FP16 inference processed and wrote all 12 frames in order at
  640×360/4 FPS using the default `mp4v`/MP4 path.
- Verified 12 output frames decode, 12 JSONL records parse, timestamps span
  0–2750 ms, and summary counts match 6 smoke and 12 fire detections.
- Local video/media/log outputs remain ignored. Machine-readable verification
  is versioned in `metrics/step4_milestone3.json`.
- Milestone gate: accepted. Webcam inference remains unimplemented pending
  Milestone 4 authorization.

### 2026-07-31 Milestone 4 — webcam inference

- Added an explicit webcam CLI mode with configuration-backed or direct index
  selection; file and webcam modes are mutually exclusive.
- Added `WebcamReader` with first-frame validation, monotonic session
  timestamps, capture FPS reporting, a 30 FPS fallback, transient read retries,
  resolution consistency checks, and idempotent release.
- Reused `DetectorEngine`, project-owned frame/detection types, the annotation
  renderer, validated video writer, flushed JSONL writer, and atomic summaries.
- Added optional live display with `Q`/Escape shutdown and a clear headless
  error; annotation work is shared with optional recording.
- Added independent recording and detection-log controls, bounded
  `--max-frames` verification, UTC-stamped run names, and explicit termination
  reasons.
- Added nine webcam tests covering CLI selection, capture validation, FPS
  fallback, recording/log order, display shutdown, Ctrl+C, read failure, a
  300-frame simulated session, and cleanup. All 34 project tests pass.
- Probed physical webcam indices 0–3 through default, DirectShow, and Media
  Foundation paths. No device was exposed to the execution environment.
- Completed a real CUDA FP16 end-to-end gate by injecting the existing
  training-derived development stream at the webcam capture boundary. All 12
  640×360 frames were processed, written, decoded, and logged in order.
- The verification summary matched 6 smoke and 12 fire detections. Visual
  inspection of a combined frame showed one plausible smoke box and three
  plausible fire boxes. The held-out test set was not used.
- Local recordings and logs remain ignored. Machine-readable verification is
  versioned in `metrics/step4_milestone4.json`.
- Milestone gate: accepted with physical-camera availability documented as the
  remaining hardware limitation. Performance instrumentation remains pending
  Milestone 5 authorization.

### 2026-07-31 Milestone 5 — production hardening

- Followed the accepted revised milestone definition: production hardening,
  not the previously planned timing instrumentation.
- Added strict duplicate-key/schema/type/probability/codec configuration
  validation and strict automatic-checkpoint metrics validation.
- Hardened model-result conversion against fractional class IDs, non-finite
  values, malformed arrays, invalid channels, and model-loader errors while
  retaining exact checkpoint class order and device validation.
- Replaced duplicated video/webcam frame loops with a shared ordered processing
  and cleanup lifecycle. Every output resource and capture is attempted during
  cleanup, and setup/zero-frame failures produce summaries when metadata is
  available.
- Added strict sorted JSON/JSONL, collision-resistant atomic publication,
  atomic annotated-image writes, safe output stems, deterministic collision
  suffixes, and removal of zero-frame recordings.
- Added deterministic flat-directory image orchestration over the existing
  image pipeline and optional `--log-level` diagnostics. CLI errors now include
  `error_type`; existing fields and flags remain available.
- Expanded the suite from 34 to 50 tests. All tests, compile/import checks, and
  dependency checks pass; Steps 1–3 implementation/config/test files remain
  unchanged.
- Manual regressions passed for CPU image/video, CUDA image-directory and
  simulated-camera inference, empty detections, corrupt/unsupported inputs,
  unavailable camera/device, writer failure, interruption, output collisions,
  checkpoint mismatch, class mismatch, and invalid configuration.
- The held-out test set was not used. Local verification artifacts remain
  ignored. Machine-readable evidence is versioned in
  `metrics/step4_milestone5.json`.
- Formal latency and FPS benchmarking did not start. Milestone 6 remains
  unauthorized and unstarted; the full documentation audit remains reserved
  for Milestone 7.

### 2026-08-01 Step 5 — FastAPI backend

- Added typed configuration for host, port, CORS, maximum upload bytes,
  persistent output, temporary storage, and inference YAML.
- Added lifespan-owned initialization. Startup verifies the checkpoint, class
  order, and device once, then every request reuses the same `DetectorEngine`.
- Added `GET /health`, `POST /predict/image`, and `POST /predict/video` with
  Pydantic models and useful OpenAPI descriptions and examples.
- Kept routes thin through `InferenceService`; image/video work reuses the
  existing pipelines, annotations, collision-safe outputs, JSON/JSONL, and
  engine prediction lock.
- Added bounded uploads, extension/content-type validation, temporary-file
  cleanup, output references, CORS, and structured request errors.
- Reserved the `/predict/*` architecture for webcam without registering
  `POST /predict/webcam`; it currently returns structured 404.
- Added 10 CPU-only API tests. All 60 project tests, compile checks, and
  dependency checks pass.
- Started the real API with the epoch-54 checkpoint on `cuda:0`. Health and
  Swagger returned 200. The development image returned one smoke and three
  fire detections. The 12-frame video returned 12 ordered records and 18
  detections (6 smoke, 12 fire); both outputs were retrievable.
- Re-ran the existing CUDA CLI on a development negative image; its detection
  list stayed empty. The held-out test set was not used.
- Deployment, frontend work, live streaming, formal benchmarking, and the full
  documentation/release audit remain deferred.
