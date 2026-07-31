# PyroVision FastAPI backend

PyroVision 1.0 exposes the existing verified inference package through a
thin FastAPI adapter. It calls `DetectorEngine`, `infer_image`, and
`infer_video`, then maps project-owned results into Pydantic response models.
The API does not import training code or expose raw Ultralytics objects.

Deployment, webcam HTTP inference, live streaming, and frontend work are not
part of the backend release. Formal benchmarking was completed separately in
Step 6.

## Architecture and lifecycle

`create_app()` builds an independently testable application. Its lifespan
handler creates the configured directories, verifies the checkpoint SHA-256,
validates class order and device availability, and loads one
`DetectorEngine`. Every request reuses that engine.

The routes only persist bounded uploads and dispatch to `InferenceService`.
Blocking inference runs in FastAPI's thread pool. Uploads are removed on
success and all handled failures. Annotation, media encoding, collision-safe
naming, JSON, and JSONL remain owned by the existing inference pipelines.

Generated output references begin with `/outputs/` and are fetchable from the
same backend origin.

## Installation and startup

Install inference/CUDA dependencies as documented in the README, then add the
backend dependencies to the same virtual environment:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-backend.txt
```

Run from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path src)
.venv\Scripts\python.exe -m pyrovision.api
```

After installing the package, the equivalent entry point is
`pyrovision-api`.

The default address is `http://127.0.0.1:8000`. Startup fails closed if the
checkpoint is missing, its digest differs, class names are out of order, or
the requested device is unavailable.

OpenAPI is immediately available at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## Configuration

The backend reuses `configs/inference.yaml` for checkpoint, model, device,
threshold, frame-skip, and video-output policy. These environment variables
configure the HTTP service:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PYROVISION_API_HOST` | `127.0.0.1` | Uvicorn bind host |
| `PYROVISION_API_PORT` | `8000` | Uvicorn bind port |
| `PYROVISION_API_CORS_ORIGINS` | `http://localhost:3000` | Allowed origins |
| `PYROVISION_API_MAX_UPLOAD_SIZE_BYTES` | `262144000` | Upload limit |
| `PYROVISION_API_OUTPUT_DIR` | `outputs/api` | Persistent outputs |
| `PYROVISION_API_TEMP_DIR` | `outputs/api-temp` | Ephemeral uploads |
| `PYROVISION_API_INFERENCE_CONFIG` | `configs/inference.yaml` | Inference YAML |

`PYROVISION_API_CORS_ORIGINS` accepts comma-separated origins. Relative paths
resolve from the repository root. Persistent output and temporary directories
must be different.

```powershell
$env:PYROVISION_API_CORS_ORIGINS = "http://localhost:3000,https://example.test"
$env:PYROVISION_API_MAX_UPLOAD_SIZE_BYTES = "104857600"
```

## `GET /health`

This endpoint responds only after successful model startup:

```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_loaded": true,
  "checkpoint_sha256": "21812ec7917bda5ad004fc085ba6a9d8ee1b375c95db2efe754463fc430d28c3",
  "checkpoint_identifier": "21812ec7917b",
  "device": "cuda:0",
  "uptime_seconds": 9.359
}
```

```powershell
curl.exe http://127.0.0.1:8000/health
```

## `POST /predict/image`

Accepts one multipart field named `file`. Supported extensions are JPEG, PNG,
BMP, TIFF, and WebP. Extension and content type are checked before the existing
image pipeline performs the authoritative OpenCV decode.

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict/image `
  -F "file=@media\sample.jpg;type=image/jpeg"
```

Successful response:

```json
{
  "success": true,
  "original_filename": "sample.jpg",
  "width": 1280,
  "height": 720,
  "detections": [
    {
      "class_id": 1,
      "class": "fire",
      "confidence": 0.91,
      "bbox": [104.2, 82.5, 310.8, 287.1]
    }
  ],
  "processing": {
    "request_id": "8f6c7c0f4d35469183df7d7f71cc7e65",
    "media_type": "image",
    "duration_ms": 42.1,
    "device": "cuda:0",
    "checkpoint_sha256": "21812ec7917bda5ad004fc085ba6a9d8ee1b375c95db2efe754463fc430d28c3"
  },
  "annotated_output": {
    "url": "/outputs/images/8f6c7c0f4d35469183df7d7f71cc7e65_annotated.jpg",
    "filename": "8f6c7c0f4d35469183df7d7f71cc7e65_annotated.jpg",
    "content_type": "image/jpeg"
  },
  "detections_output": {
    "url": "/outputs/images/8f6c7c0f4d35469183df7d7f71cc7e65_detections.json",
    "filename": "8f6c7c0f4d35469183df7d7f71cc7e65_detections.json",
    "content_type": "application/json"
  }
}
```

## `POST /predict/video`

Accepts AVI, M4V, MKV, MOV, MP4, and WebM uploads. After extension and content
type checks, the existing `VideoReader` verifies that OpenCV can open the
container and decode frames. Codec support depends on the local OpenCV backend.

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict/video `
  -F "file=@media\sample.mp4;type=video/mp4"
```

The response includes:

- processed frame count;
- ordered frame records with source indices and timestamps;
- structured detections without Ultralytics objects;
- checkpoint, device, and request metadata;
- annotated video, JSONL, and summary output references;
- aggregate per-class detection totals.

Frame records use this shape:

```json
{
  "processed_index": 0,
  "frame_index": 0,
  "timestamp_ms": 0.0,
  "width": 1280,
  "height": 720,
  "detections": [
    {
      "class_id": 0,
      "class": "smoke",
      "confidence": 0.87,
      "bbox": [42.0, 58.1, 320.4, 280.0]
    }
  ]
}
```

The full upload is processed before responding. Live streaming is not
implemented.

## Errors

Handled HTTP, upload, media, inference, and output errors use one envelope:

```json
{
  "success": false,
  "error": {
    "code": "unsupported_media_type",
    "message": "Unsupported image file extension: .txt",
    "details": null
  }
}
```

| Status | Example codes |
| ---: | --- |
| 413 | `upload_too_large` |
| 415 | `unsupported_media_type` |
| 422 | `validation_error`, `empty_upload`, `invalid_media` |
| 500 | `inference_failed`, `output_generation_failed` |
| 503 | `model_unavailable` |

Checkpoint loading failure is logged and aborts startup, so the service cannot
expose a misleading healthy state.

## Webcam placeholder

The `/predict/*` router can accept a future `POST /predict/webcam` adapter. It
is intentionally not registered or advertised in OpenAPI; requests currently
receive the structured 404 response. The existing local webcam pipeline is
unchanged.

## Verification

The API suite uses CPU-only project-type stubs. It covers startup, one-time
model loading, health, image/video uploads, output retrieval, unsupported,
mismatched, corrupt, empty, oversized and missing files, inference failure,
configuration, OpenAPI, Swagger, and the webcam placeholder.

The manual gate loaded the verified epoch-54 checkpoint on `cuda:0`. Health and
Swagger returned HTTP 200. A training-derived development image returned one
smoke and three fire detections. The 12-frame development video returned 12
ordered records and 18 detections (6 smoke, 12 fire); both annotated output
URLs returned HTTP 200. The held-out test set was not used. Operational request
duration fields are not a formal benchmark. Formal CPU/CUDA component,
pipeline, and HTTP results are available in `docs/benchmark.md`.

Production hosting controls and unresolved public-deployment limits are
documented in `docs/deployment.md`.
