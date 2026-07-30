# Step 4 — local inference

Status: **Milestone 1 foundation complete**

Milestone 1 establishes the reusable boundary that later image, video, webcam,
and FastAPI adapters will share. It does not run detection or write annotated
media yet.

## Package structure

```text
src/pyrovision/
  __init__.py       Public package surface and version
  config.py         Typed YAML configuration and strict validation
  checkpoints.py    Checkpoint resolution, integrity, and class contracts
  device.py         Explicit CPU/CUDA resolution
  errors.py         Project-owned expected exception hierarchy
  hashing.py        Streaming SHA-256 utility
  types.py          Framework-independent detection and frame results
```

The package never imports `scripts/train.py`. The training runner imports only
the neutral hash helper so checkpoint hashing has one implementation while all
training, pause/resume, and evaluation behavior remains unchanged.

## Configuration contract

The versioned contract is `configs/inference.yaml`, schema version 1. Unknown
keys and invalid field types fail immediately instead of being ignored.

| Section | Configurable values |
| --- | --- |
| `checkpoint` | explicit/automatic path, metrics record, hash enforcement, expected classes |
| `device` | `auto`, `cpu`, `cuda`, or `cuda:N` |
| `model` | image size, global/class thresholds, IoU, maximum detections, FP16 policy |
| `input` | source placeholder, webcam index, frame skipping |
| `output` | output directory, media/log saving, display, four-character codec |

Project-owned paths are resolved against the repository root. The source is
left unset so a later CLI can supply an image, video, or webcam without editing
the YAML file.

## Checkpoint contract

The default `path: auto` reads
`training.selected_checkpoint` from `metrics/yolo11s_baseline.json`. Resolution
checks the recorded path first and then a project-relative
`runs/pyrovision/<experiment>_train/weights/best.pt` fallback so a relocated
repository does not depend solely on a stored absolute Windows path.

The selected file is streamed through SHA-256 before use. Verification fails
closed if the expected digest is absent or different. The current verified
checkpoint is:

```text
Epoch:   54
SHA-256: 21812ec7917bda5ad004fc085ba6a9d8ee1b375c95db2efe754463fc430d28c3
Classes: 0 smoke, 1 fire
```

Class validation requires exact count, ID order, spelling, and casing. The
future detector engine will invoke the same validation after loading the model.

## Device policy

- `auto`: select `cuda:0` when PyTorch reports CUDA; otherwise select CPU.
- `cpu`: always use CPU, including inside a CUDA-enabled environment.
- `cuda`: require `cuda:0`; fail clearly if CUDA is unavailable.
- `cuda:N`: require that exact device index; never silently choose another.

FP16 eligibility is recorded only for CUDA. The later detector engine will
combine this capability with the configured `half` policy.

## Stable result boundary

`BoundingBox`, `Detection`, and `FrameResult` are immutable, framework-neutral
types. Their dictionaries use stable field order and documented precision. A
future backend can serialize these objects without importing Ultralytics or
depending on its internal `Results` representation.

## Threshold policy

The initial confidence value is 0.35. The selected-checkpoint validation curve
peaked at aggregate F1 0.75 near confidence 0.351. The held-out test optimum is
not used for operating-threshold selection. Both class overrides currently
equal the global value and will remain provisional until external development
video is available.

## Verification

Milestone 1 verification completed on 2026-07-31:

- Real checkpoint resolution: passed
- SHA-256 verification: passed
- Checkpoint class inspection: passed (`smoke`, `fire`)
- Device auto-selection: passed (`cuda:0`, RTX 4050 Laptop GPU)
- Dependency consistency: previously verified and unchanged
- Existing Step 1–3 tests: 6 passed
- New foundation tests: 7 passed
- Total automated tests: 13 passed

Milestone 2 will add `DetectorEngine` and still-image inference. Video, webcam,
performance instrumentation, hardening, backend work, and deployment remain
out of scope at this gate.
