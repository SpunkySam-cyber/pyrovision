# Step 4 — local inference

Status: **complete — image, directory, video, and webcam inference hardened and
formally benchmarked**

Milestone 1 established the reusable boundary now shared by image, directory,
video, webcam, benchmark, and FastAPI adapters. The sections below retain the
implementation history and finish with the current hardened contract.

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
  webcam.py         Webcam capture, display, recording, and run summaries
  streaming.py      Shared ordered frame processing and resource cleanup
  timing.py         Project-owned opt-in prediction timing results
  benchmarking.py   Reproducible CPU/CUDA component and pipeline benchmark
  yaml_utils.py     Strict duplicate-key-safe YAML loading
  api/              FastAPI lifecycle, schemas, upload handling, and services
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
detector engine invokes the same validation after loading the model.

## Device policy

- `auto`: select `cuda:0` when PyTorch reports CUDA; otherwise select CPU.
- `cpu`: always use CPU, including inside a CUDA-enabled environment.
- `cuda`: require `cuda:0`; fail clearly if CUDA is unavailable.
- `cuda:N`: require that exact device index; never silently choose another.

FP16 eligibility is recorded only for CUDA. The detector engine combines this
capability with the configured `half` policy.

## Stable result boundary

`BoundingBox`, `Detection`, and `FrameResult` are immutable, framework-neutral
types. Their dictionaries use stable field order and documented precision. The
FastAPI backend serializes these objects without exposing Ultralytics or
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
Ultralytics model state and provides the boundary reused by FastAPI.

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

This historical image gate intentionally deferred detailed latency and FPS.
Formal CPU/CUDA measurements are now available in `docs/benchmark.md`.

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
timings are reported separately in `docs/benchmark.md` so normal output schemas
remain unchanged.

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

## Milestone 4 — webcam inference

Webcam inference is an explicit CLI mode, so a missing file source never opens
a camera unexpectedly. The bare flag uses `input.webcam_index` from
`configs/inference.yaml`; passing an integer overrides it:

```powershell
# Camera index 0 from configuration, with a live window
.venv\Scripts\python.exe -B scripts\infer.py --webcam --display

# Explicit camera index, bounded headless verification run
.venv\Scripts\python.exe -B scripts\infer.py `
  --webcam 1 `
  --no-display `
  --max-frames 300
```

`--source` and `--webcam` are mutually exclusive. Invalid or unavailable camera
indices fail with a structured CLI error that identifies camera permissions and
device contention as likely causes. An opened camera must produce a valid frame
within three read attempts. The first real frame establishes capture dimensions;
subsequent resolution changes fail cleanly instead of corrupting the output.

### Capture and timestamp contract

`WebcamReader` owns the OpenCV capture and exposes the same `SourceFrame`
boundary used by video inference. It provides:

- Original zero-based capture indices, including skipped frames
- Monotonic timestamps relative to capture start
- Dimensions validated against the first captured frame
- Camera-reported FPS, or an explicit 30 FPS fallback when unavailable
- Idempotent capture release through a context manager

Frame skipping uses the existing `input.frame_skip` setting. Recording FPS is
divided by the processing stride, matching the video policy. Accurate
end-to-end and processed FPS measurements were completed in Step 6.

### Live display and shutdown

Live display is opt-in through `--display` or `output.display: true`. Annotation
is computed once and shared between display and recording. Press `Q`, `q`, or
Escape to finish normally. Ctrl+C or the reusable programmatic stop callback
marks the run interrupted. In every processing exit path, the pipeline closes:

1. The display window
2. The flushed JSONL writer
3. The annotated video writer
4. The webcam capture

A display initialization/runtime failure explains how to use `--no-display`
for headless environments. CLI interruptions return exit code 130.

### Optional recording and logs

Recording and structured detections are independent:

```powershell
# Display and JSONL only
.venv\Scripts\python.exe -B scripts\infer.py `
  --webcam 0 `
  --display `
  --no-record

# Display and recording only
.venv\Scripts\python.exe -B scripts\infer.py `
  --webcam 0 `
  --display `
  --no-save-detections
```

The defaults record and log without displaying. Each session receives a
UTC-stamped stem such as `webcam_0_20260731T174543_423615Z` and can produce:

```text
outputs/inference/
  <run>_annotated.mp4
  <run>_detections.jsonl
  <run>_summary.json
```

The JSONL frame schema is the same deterministic project-owned schema used for
videos. The atomic summary adds camera index, reported/fallback FPS source,
display/recording flags, termination reason, frame counts, timestamp range,
checkpoint/device identity, and per-class totals.

### Milestone 4 verification

- Existing tests before Milestone 4: 25 passed
- New webcam tests: 9 passed
- Total: 34 passed
- Index/config CLI selection and source exclusivity: passed
- Invalid, unavailable, and non-reading cameras: passed
- Display quit-key handling and display cleanup: passed
- Recording, ordered JSONL, and atomic summary: passed
- Ctrl+C and capture-failure partial-output handling: passed
- Simulated 300-frame long session with stable counts/cleanup: passed
- Real checkpoint on CUDA through the webcam capture boundary: passed

The real CUDA gate injected the existing 640×360, 4 FPS, 12-frame
training-derived development stream at the capture factory boundary. It used
the verified epoch-54 checkpoint on `cuda:0`, processed/wrote/decoded all 12
frames, wrote 12 ordered JSONL records, and counted 6 smoke plus 12 fire
detections. Visual inspection of a combined frame showed one plausible smoke
box and three plausible fire boxes. The held-out test set was not used.

Physical webcam probing was attempted on indices 0–3 with the OpenCV default,
DirectShow, and Media Foundation paths. No camera device was exposed to this
execution environment, so a true live-camera and GUI-window session remains a
documented hardware verification item. The machine-readable record is
`metrics/step4_milestone4.json`.

At this historical gate, timing, backend work, and deployment remained out of
scope. Timing and backend work are now complete; deployment remains deferred.

## Milestone 5 — production hardening

The accepted revised roadmap assigns Milestone 5 to production hardening.
Historical timing references above describe the earlier plan; no formal
latency or FPS benchmarking was performed in this milestone.

### Deterministic directory and output behavior

A flat directory is now a valid `--source`. Supported top-level images are
processed with the existing image pipeline in case-insensitive filename order;
nested directories and unsupported extensions are not included. An empty
directory fails before inference output is produced.

```powershell
.venv\Scripts\python.exe -B scripts\infer.py `
  --source path\to\images `
  --output-dir outputs\directory-run
```

Existing files are never silently overwritten. First-run names remain
unchanged. If any requested file in an output group exists, the whole group
receives the first free `_2`, `_3`, and subsequent suffix. Webcam UTC names and
programmatic `run_name` values use the same allocator; path-like or
Windows-invalid stems are rejected.

Atomic JSON now uses collision-resistant temporary files in the destination
directory. JSON and JSONL serialize with sorted keys, reject NaN/infinity, and
flush complete JSONL records. Annotated images are encoded to a same-extension
temporary file and atomically published.

### Validation and failure behavior

Configuration now rejects duplicate YAML keys, missing `schema_version`,
non-finite probabilities, non-string class names/threshold keys, punctuation in
four-character codecs, and malformed nested configuration types. Training
metrics used for automatic checkpoint selection must contain correctly typed
paths, epochs, experiment IDs, and SHA-256 values.

Checkpoint model results reject fractional/boolean/out-of-range class IDs,
non-finite confidence or box values, invalid channel counts, malformed result
arrays, and model-loader failures through project-owned errors. Checkpoint
hash, exact class order, and explicit CUDA device validation remain fail-closed.

Video and webcam processing now share one ordered frame-processing lifecycle.
Setup, inference, display, encoding, JSONL, interruption, and cleanup failures
all attempt to close the display, JSONL writer, video writer, and capture.
Setup and zero-frame failures write failed summaries when source metadata is
available. Zero-frame recordings are removed instead of being returned as
decodable media. Empty videos fail rather than reporting a successful run.

CLI errors retain the existing `success` and `error` fields and add a stable
`error_type`. `--log-level DEBUG|INFO|WARNING|ERROR` enables optional stderr
diagnostics; the default remains `WARNING`. `--max-frames` is rejected outside
webcam mode instead of being silently ignored.

### Milestone 5 verification

- Existing tests before hardening: 34
- Total tests after hardening: 50 passed
- Compile/import and dependency checks: passed
- CPU image inference and collision rerun: passed
- CUDA flat-directory inference over four training-derived categories: passed
- Negative image with zero detections: passed
- CPU video inference and 12/12-frame decode: passed
- CUDA simulated-camera inference, recording, and 12 JSONL records: passed
- Two-frame interrupted video, partial decode/JSONL/summary: passed
- Unsupported/corrupt input and unavailable camera: passed
- Forced writer-open failure and failed summary without invalid media: passed
- Checkpoint hash, class order, invalid config, and unavailable device: passed
- Held-out test data used: no
- Formal latency/FPS benchmarking performed: no

OpenCV's FFmpeg backend accepted two arbitrary alphanumeric FOURCC probes with
warnings and still produced decodable 12-frame files; deterministic writer
failure therefore uses an injected closed backend in tests and the manual
failure harness. Codec availability remains platform-dependent.

The machine-readable gate record is `metrics/step4_milestone5.json`. Step 6 has
now completed the formal benchmark and repository consistency audit without
changing the established inference algorithms.

## Step 6 — performance instrumentation

`DetectorEngine.predict_frame_timed()` is an additive measurement API. It runs
the same inference path as `predict_frame()` and returns the same `FrameResult`
paired with `PredictionTiming`. Existing image, video, webcam, CLI, and backend
callers continue to use `predict_frame()` unchanged.

The benchmark separates Ultralytics preprocessing/model/postprocessing,
project result conversion, annotation, encoding, JSON serialization, complete
image/video pipelines, and local HTTP requests. Warm-up and steady-state are
reported separately, with CPU and CUDA executed in isolated processes.

Run it with:

```powershell
.venv\Scripts\python.exe -B scripts\benchmark.py `
  --config configs\benchmark.yaml
```

See `docs/benchmark.md` and `metrics/step6_benchmark.json` for the complete
methodology and results.
