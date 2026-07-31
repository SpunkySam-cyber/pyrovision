# Step 4 — local inference

Status: **Milestone 3 video inference complete**

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
  model.py          Verified YOLO loading and frame detection engine
  annotation.py     OpenCV boxes, labels, confidence, and colors
  images.py         Image decode, orchestration, persistence, and JSON output
  sources.py        Media classification and ordered video decoding
  outputs.py        Validated video writer, JSONL sink, atomic JSON summaries
  video.py          Video frame pipeline, interruption handling, run summaries
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

## Milestone 2 — image inference

`DetectorEngine.from_config()` performs the complete acceptance sequence before
an image is processed:

1. Resolve the epoch-54 checkpoint.
2. Recalculate and verify its SHA-256.
3. Resolve the requested CPU or CUDA device.
4. Load YOLO through an injectable model factory.
5. Validate exact checkpoint class count, ID order, spelling, and casing.
6. Resolve FP32/FP16 behavior from the device and `half` policy.

`predict_frame()` accepts a BGR NumPy frame and returns a project-owned
`FrameResult`. It calls YOLO with the configured image size, IoU, maximum
detections, device, and precision mode. The YOLO candidate threshold is the
lowest configured global/class threshold; each returned detection is then
filtered against its own class threshold. Coordinates are clipped to the frame,
degenerate boxes are discarded, and results receive a deterministic ordering.

The engine uses a prediction lock. This avoids concurrent access to mutable
Ultralytics model state and establishes a safe boundary for the later FastAPI
adapter.

### Image command

```powershell
.venv\Scripts\python.exe -B scripts\infer.py `
  --source path\to\image.jpg
```

Supported inputs are BMP, JPEG, PNG, TIFF, and WebP. Unsupported, missing, or
undecodable files produce structured errors and a non-zero exit code.

Overrides include:

- `--device auto|cpu|cuda|cuda:N`
- `--confidence 0.40`
- repeatable `--class-threshold fire=0.30`
- `--iou 0.70`
- `--output-dir outputs/manual`
- `--save-media` / `--no-save-media`
- `--save-detections` / `--no-save-detections`

A global CLI confidence override applies to all expected classes first;
repeatable class overrides are applied afterward.

### Output contract

For `sample.jpg`, the default output contains:

```text
outputs/inference/
  sample_annotated.jpg
  sample_detections.json
```

The JSON record contains checkpoint identity, resolved device, output paths,
source dimensions, frame index/timestamp, and detections with stable fields:

```json
{
  "class_id": 1,
  "class": "fire",
  "confidence": 0.912346,
  "bbox": [20.0, 15.0, 100.0, 75.0]
}
```

Boxes use pixel-space `[x_min, y_min, x_max, y_max]`. Rendering operates on a
copy and never mutates the input frame.

### Real CPU/CUDA verification

The hardware gate used
`data/processed/dfire/images/train/AoF04009.jpg`, a 1280×720 `smoke+fire`
training-split image. The held-out test split was not used.

| Device | Precision mode | Smoke | Fire | Output |
| --- | --- | ---: | ---: | --- |
| CPU | FP32 | 1 | 3 | Annotated JPEG and JSON passed |
| RTX 4050 `cuda:0` | FP16 | 1 | 3 | Annotated JPEG and JSON passed |

Minor confidence/coordinate differences are expected between FP32 and FP16.
Both devices found the same four objects, and manual inspection confirmed
plausible annotations. Generated outputs remain local under
`outputs/milestone2/` and are ignored by Git. The structured verification record
is versioned at `metrics/step4_milestone2.json`.

### Milestone 2 verification

- Existing Step 1–3 tests: 6 passed
- Foundation tests: 7 passed
- Image-inference tests: 6 passed
- Total: 19 passed
- CPU execution: passed
- CUDA execution: passed
- Annotated image decode/manual inspection: passed
- Structured JSON decode: passed

Detailed latency and FPS are intentionally not reported yet; those measurements
belong to Milestone 5 after the video pipeline exists.

## Milestone 3 — video inference

The image CLI now classifies supported local paths before model loading and
routes videos through the ordered video pipeline:

```powershell
.venv\Scripts\python.exe -B scripts\infer.py `
  --source path\to\video.mp4
```

Supported video extensions are AVI, M4V, MKV, MOV, MP4, and WebM, subject to the
codecs available in the local OpenCV build.

### Source and frame contract

`VideoReader` validates container opening, dimensions, and FPS. Every yielded
frame includes:

- The original zero-based source frame index
- The container timestamp when it is valid and monotonic
- A deterministic `frame_index / source_fps` fallback timestamp
- The decoded BGR frame

Frame order is never parallelized or reordered. With `frame_skip: N`, every
`N + 1` frame is processed while skipped frames are still decoded and counted.
The annotated output FPS is divided by `N + 1`, preserving approximately the
same playback duration instead of accelerating the output.

### Outputs

For `video.mp4`, the default run writes:

```text
outputs/inference/
  video_annotated.mp4
  video_detections.jsonl
  video_summary.json
```

Each JSONL line is flushed immediately and contains `processed_index`, original
`frame_index`, timestamp, dimensions, and the stable detection list. This keeps
every complete record readable after an interruption.

The summary records completion/interruption status, checkpoint, device, codec,
input/output FPS, dimensions, source/read/processed/written frame counts,
timestamp range, frame skipping, and total/per-class detections. Detailed stage
timings are intentionally deferred to Milestone 5.

### Codec handling

The configured default is `mp4v` with `.mp4`. `VideoWriter.isOpened()` is
checked before any frame is processed, and failures recommend known Windows
fallbacks. The CLI can override both values:

```powershell
.venv\Scripts\python.exe -B scripts\infer.py `
  --source path\to\video.mp4 `
  --codec MJPG `
  --video-extension .avi
```

Verification covered real `mp4v`/MP4 writing, automated MJPG/AVI writing, and a
forced writer-open failure. GStreamer remains unavailable and H.264 writing is
not assumed.

### Graceful interruption

The pipeline supports both a programmatic stop callback and Ctrl+C. In either
case it:

1. Stops before beginning another inference.
2. Flushes and closes JSONL.
3. Finalizes and closes the partial video.
4. Releases the input capture.
5. Writes a summary with status `interrupted` and the reason.

The CLI returns exit code 130 for an interrupted video run. Automated tests
decode the partial output video and parse every retained JSONL line.

### Real video verification

No external video was present, so the reproducible development clip was built
from four processed training images: negative, smoke, fire, and smoke+fire. No
held-out test image was used.

| Property | Result |
| --- | --- |
| Input/output | 640×360, 4 FPS, MP4V/MP4 |
| Device | RTX 4050 `cuda:0`, FP16 |
| Frames read/processed/written | 12 / 12 / 12 |
| Output frames declared/decoded | 12 / 12 |
| JSONL records | 12 |
| Timestamp range | 0–2750 ms in 250 ms steps |
| Smoke/fire detections | 6 / 12 |
| Frame detection counts | `0,0,0,1,1,1,1,1,1,4,4,4` |

The negative segment remained empty, followed by consistent smoke, fire, and
combined segments. Local source and generated outputs remain ignored. The
machine-readable result is `metrics/step4_milestone3.json`.

### Milestone 3 verification

- Existing tests before Milestone 3: 19 passed
- New video tests: 6 passed
- Total: 25 passed
- Ordered frame/timestamp handling: passed
- Full video/JSONL/summary creation: passed
- MP4V/MP4 and MJPG/AVI writing: passed
- Codec-open failure handling: passed
- Stop callback and Ctrl+C partial-output handling: passed
- Real annotated output decode: 12/12 frames

Milestone 4 will add webcam index parsing, optional live display, camera failure
handling, and long-running resource validation. Webcam, performance
instrumentation, backend work, and deployment remain out of scope at this gate.
