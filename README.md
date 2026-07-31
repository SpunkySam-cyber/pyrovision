# PyroVision AI

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Backend](https://img.shields.io/badge/backend-v1.0.0-009688.svg)](docs/api.md)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](RELEASE_NOTES.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Real-time fire and smoke detection with YOLO11. The project is being built and
verified one stage at a time. Dataset preparation, YOLO11s training, held-out
test evaluation, local inference engineering, the FastAPI backend, and the
Version 1.0 release audit (Steps 1–6) are complete. Deployment and frontend
work have not started.

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

Preparation also repairs 91 source JPEGs that lack an end-of-image marker in
the processed copy only. The raw archive is unchanged, and strict verification
loads every processed image successfully before training.

Source: [D-Fire dataset](https://github.com/gaia-solutions-on-demand/DFireDataset)

## Step 2 — YOLO11s training

The baseline uses a COCO-pretrained `yolo11s.pt` checkpoint, 640 px inputs,
seed 42, standard HSV/translation/scale/horizontal-flip/mosaic augmentation,
and the verified training and validation splits. The held-out test split is not
used during Step 2.

Create the local GPU environment on this RTX 4050 / CUDA 12.9 machine:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-cuda129.txt
.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.venv\Scripts\python.exe scripts\check_environment.py --require-cuda
```

The verified stage commands are deliberately separate:

```powershell
# Record validation metrics before fine-tuning
.venv\Scripts\python.exe scripts\train.py baseline

# Verify the training loop on a deterministic, category-stratified subset
.venv\Scripts\python.exe scripts\train.py smoke-test

# Run the full fine-tuning job only after both gates pass
.venv\Scripts\python.exe scripts\train.py train

# Resume a paused run and preserve all optimizer/scheduler state
.venv\Scripts\python.exe scripts\train.py resume --stop-after-epoch 60

# Validate and hash best/last checkpoints without touching the test split
.venv\Scripts\python.exe scripts\train.py finalize
```

Per-epoch box/class/DFL losses, precision, recall, mAP50, and mAP50–95 are
written to the Ultralytics `results.csv` and mirrored into
`metrics/yolo11s_baseline.json` after each epoch. The metrics record also keeps
the pre-training baseline, environment details, best/last checkpoint hashes,
peak GPU allocation, explicit best-checkpoint validation, and any failure.
Weights, local datasets, and generated run artifacts are intentionally ignored
by Git.

`--stop-after-epoch` pauses only after that epoch's metrics and resumable
`last.pt` have been fully written. It exits before Ultralytics strips optimizer
state during normal finalization, allowing another true resume later.

Training completed after 70 epochs, with epoch 54 selected by validation
mAP50–95. Its validation metrics are precision 0.7905, recall 0.7118, mAP50
0.7893, and mAP50–95 0.4669. Full before/during/after metrics and checkpoint
hashes are documented in `docs/training/yolo11s_baseline.md`.

## Step 3 — held-out evaluation

Run the selected checkpoint once on the untouched test split:

```powershell
.venv\Scripts\python.exe scripts\evaluate.py
```

The command refuses to run before Step 2 is finalized or after a completed
test result already exists. It records aggregate and per-class metrics,
generates confusion matrices and metric curves, and saves deterministic
negative/smoke/fire/combined sanity predictions.

| Scope | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| Overall | 0.7657 | 0.6992 | 0.7642 | 0.4526 |
| Smoke | 0.8189 | 0.7806 | 0.8368 | 0.5378 |
| Fire | 0.7125 | 0.6178 | 0.6916 | 0.3675 |

The complete evaluation record and limitations are documented in
`docs/evaluation.md`. Generated plots and annotated test images remain local
and Git-ignored with the run directory.

## Step 4 — local real-time inference

Status: **complete and formally benchmarked.**

The reusable package now lives under `src/pyrovision/`. Its versioned inference
configuration validates checkpoint selection, expected class names, global and
class-specific thresholds, device policy, media defaults, and output policy.
With `checkpoint.path: auto`, it reads the selected epoch-54 checkpoint and
expected SHA-256 from the Step 2 metrics record. Absolute-path relocation is
handled through a project-relative run fallback.

`device: auto` selects `cuda:0` when CUDA is available and otherwise uses CPU.
Explicit `cuda`, `cuda:N`, and `cpu` requests are validated and never silently
changed. The current local foundation gate resolved the RTX 4050 as `cuda:0`,
verified the selected checkpoint hash, and confirmed checkpoint classes
`smoke`, `fire` in the required order.

The initial global and per-class confidence values in
`configs/inference.yaml` are 0.35, based on the validation-only F1 curve rather
than the held-out test curve. They remain provisional until development-video
testing.

Run the complete foundation test gate:

```powershell
.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Milestone 2 adds a reusable `DetectorEngine`, class-aware confidence filtering,
OpenCV annotation, annotated-image output, deterministic JSON output, and a thin
image CLI. Run local image inference with automatic GPU selection:

```powershell
.venv\Scripts\python.exe -B scripts\infer.py `
  --source path\to\image.jpg
```

Useful overrides:

```powershell
# Force CPU and change the output directory
.venv\Scripts\python.exe -B scripts\infer.py `
  --source path\to\image.jpg `
  --device cpu `
  --output-dir outputs\manual

# Apply one global threshold, then lower only the fire threshold
.venv\Scripts\python.exe -B scripts\infer.py `
  --source path\to\image.jpg `
  --confidence 0.40 `
  --class-threshold fire=0.30
```

The command writes `<stem>_annotated.<ext>` and `<stem>_detections.json` under
the configured ignored output directory. CPU FP32 and CUDA FP16 inference were
both verified on a training-split smoke-and-fire image with consistent objects.

Milestone 3 extends the same command to supported local videos:

```powershell
.venv\Scripts\python.exe -B scripts\infer.py `
  --source path\to\video.mp4
```

Video output contains:

```text
outputs/inference/
  video_annotated.mp4
  video_detections.jsonl
  video_summary.json
```

Each JSONL line retains the original frame index and timestamp. Frame skipping
is optional; when enabled, output FPS is divided by the processing stride so
playback duration remains correct:

```powershell
.venv\Scripts\python.exe -B scripts\infer.py `
  --source path\to\video.mp4 `
  --frame-skip 1
```

The default Windows codec/container is `mp4v`/`.mp4`. An explicit alternative
can be requested with `--codec MJPG --video-extension .avi`. Writer creation is
validated before processing.

Milestone 4 adds an explicit webcam mode. The bare flag uses
`input.webcam_index` from the inference configuration; an integer selects a
different camera:

```powershell
# Use configured camera 0, show live annotations, record, and log detections
.venv\Scripts\python.exe -B scripts\infer.py --webcam --display

# Select camera 1 without recording; stop after 300 processed frames
.venv\Scripts\python.exe -B scripts\infer.py `
  --webcam 1 `
  --no-record `
  --max-frames 300
```

Display is off by default. With it enabled, press `Q` or Escape for a clean
normal shutdown; Ctrl+C produces an interruption-safe partial run. Recording
and JSONL logging can be controlled independently with `--record`/
`--no-record` and `--save-detections`/`--no-save-detections`. Every webcam run
writes an atomic summary and uses a UTC-stamped name so separate sessions do
not overwrite each other.

The Milestone 4 gate had 34 passing tests. Its real CUDA webcam pipeline retained and
decoded all 12 frames from the training-derived development stream, wrote 12
ordered JSONL records, and produced a matching summary. A physical webcam was
not exposed to this execution environment on indices 0–3 through DirectShow or
Media Foundation, so final live-camera/window verification remains a local
hardware check. See `docs/inference.md` and
`metrics/step4_milestone4.json` for that gate's full record.

Milestone 5 hardens every local inference mode. A flat directory of supported
images can now use the existing image pipeline in deterministic filename order:

```powershell
.venv\Scripts\python.exe -B scripts\infer.py `
  --source path\to\image-directory `
  --device cuda:0
```

Repeated runs no longer overwrite existing outputs. The first run keeps the
original names; later runs receive `_2`, `_3`, and subsequent deterministic
suffixes across the entire requested output group. JSON and JSONL are strict,
key-sorted, reject NaN/infinity, and image/JSON publication is atomic.

Expected CLI failures now include `error_type` while retaining `success` and
`error`. Optional diagnostics can be enabled without changing default output:

```powershell
.venv\Scripts\python.exe -B scripts\infer.py `
  --source path\to\image.jpg `
  --log-level INFO
```

All **50 tests** pass. CPU image/video and CUDA directory/simulated-camera
regressions passed with the selected checkpoint. Corrupt and unsupported
inputs, unavailable cameras/devices, writer failure, interruption, collisions,
checkpoint mismatch, class-order mismatch, and invalid configuration were all
verified. See `docs/inference.md` and `metrics/step4_milestone5.json` for this
gate. The held-out test set was not used. These were the pre-benchmark
hardening results; formal measurements are recorded in Step 6.

## Step 5 — FastAPI backend

Status: **complete locally; not deployed.**

The backend is a thin adapter over the existing `pyrovision` package. FastAPI
lifespan loads and verifies one `DetectorEngine`; image and video requests
reuse the existing inference, annotation, output, and deterministic
serialization pipelines. No training or Ultralytics result logic lives in the
routes.

Install and start from the repository root:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-backend.txt
$env:PYTHONPATH = (Resolve-Path src)
.venv\Scripts\python.exe -m pyrovision.api
```

Quick checks:

```powershell
curl.exe http://127.0.0.1:8000/health

curl.exe -X POST http://127.0.0.1:8000/predict/image `
  -F "file=@path\to\image.jpg;type=image/jpeg"

curl.exe -X POST http://127.0.0.1:8000/predict/video `
  -F "file=@path\to\video.mp4;type=video/mp4"
```

Swagger UI is available at `http://127.0.0.1:8000/docs`. Generated annotated
media is returned as a backend-relative `/outputs/...` reference. Host, port,
CORS origins, upload size, persistent output, temporary storage, and inference
configuration paths are environment-configurable.

All **60 tests** pass: the previous 50 inference/training/evaluation tests plus
10 backend tests. The real local API gate loaded the verified checkpoint once
on `cuda:0`; health, Swagger, development image/video inference, and annotated
output retrieval passed. See `docs/api.md` and `metrics/step5_backend.json`.

`POST /predict/webcam`, live streaming, deployment, and frontend development
remain out of scope for the backend release.

## Step 6 — benchmark and Version 1.0 release gate

Status: **complete; recommended tag `v1.0.0`.**

Run the versioned CPU/CUDA benchmark:

```powershell
.venv\Scripts\python.exe -B scripts\benchmark.py `
  --config configs\benchmark.yaml
```

Steady-state summary on the local RTX 4050 development machine:

| Measurement | CPU | CUDA |
| --- | ---: | ---: |
| Image model inference | 293.546 ms | 16.308 ms |
| Full image pipeline | 307.801 ms | 55.890 ms |
| Image pipeline FPS | 3.249 | 17.892 |
| Full 12-frame video | 3208.373 ms | 318.795 ms |
| Video pipeline FPS | 3.740 | 37.642 |
| Steady HTTP image request | 264.680 ms | 79.668 ms |
| HTTP 12-frame video request | 4382.351 ms | 463.772 ms |

Warm-up, every component stage, hardware/software, methodology, input hashes,
and limitations are documented in [docs/benchmark.md](docs/benchmark.md) and
`metrics/step6_benchmark.json`.

The final release gate passed **69/69 automated tests** plus real-engine CPU,
CUDA, image, directory, video, simulated-camera, and local API checks. The
machine-readable release record is `metrics/step6_release.json`.

Release and deployment material:

- [API contract](docs/api.md)
- [Deployment readiness](docs/deployment.md)
- [Release notes](RELEASE_NOTES.md)
- [Changelog](CHANGELOG.md)
- [MIT license](LICENSE)

Install the test client dependency before running the complete suite:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

PyroVision remains an engineering detection baseline, not a certified
life-safety system. Public deployment still requires the controls in the
deployment readiness review.
