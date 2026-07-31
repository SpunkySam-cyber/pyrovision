# Backend deployment readiness

Status: **review complete; deployment not started**

PyroVision 1.0 is ready for controlled frontend integration. The following
production controls are intentionally documented rather than implemented in
Step 6.

## Required runtime configuration

Set these explicitly in the hosting environment:

- `PYROVISION_API_HOST=0.0.0.0`
- `PYROVISION_API_PORT=<platform port>`
- `PYROVISION_API_CORS_ORIGINS=<exact frontend origin>`
- `PYROVISION_API_MAX_UPLOAD_SIZE_BYTES=<approved limit>`
- `PYROVISION_API_OUTPUT_DIR=<persistent or managed output path>`
- `PYROVISION_API_TEMP_DIR=<ephemeral scratch path>`
- `PYROVISION_API_INFERENCE_CONFIG=configs/inference.yaml`

The checkpoint file and `metrics/yolo11s_baseline.json` must be available at
startup. Health checks should target `GET /health`; startup fails if checkpoint
integrity, class order, or device validation fails.

## Upload and execution limits

The current default byte limit is 250 MiB. Production should use the smallest
limit required by the frontend. A byte limit alone does not bound decoded
duration or frame count: a highly compressed video can still create a long GPU
job. Before public exposure, define and enforce maximum duration, dimensions,
and decoded frames at the application or trusted ingress layer.

The API processes one complete video before responding. Based on the local
benchmark, timeout scales primarily with frame count rather than file size.
Start with an ingress/application timeout appropriate to the accepted maximum
duration (often 120–300 seconds), then tune from deployment measurements.
Frontend requests should show progress/timeout errors and never retry a video
upload automatically without user confirmation.

## Workers and GPU use

Start with one Uvicorn worker per GPU. Each process loads a separate model and
consumes separate VRAM. The engine serializes predictions within a worker to
protect Ultralytics state; adding HTTP workers can therefore duplicate models
without improving single-GPU throughput. Scale only after measuring VRAM,
queueing, and concurrent request behavior on the target host.

CPU fallback is verified, but the local benchmark achieved only about 3.7
end-to-end video FPS. A GPU-backed target is recommended for interactive use.

## Storage and cleanup

The API currently returns backend-relative references to local generated
files. Local container filesystems may be ephemeral and are not a durable
public storage contract.

Choose one deployment policy before launch:

1. Store outputs in object storage and return signed, expiring URLs; or
2. Mount a persistent volume and serve outputs through a controlled proxy.

Apply a short retention period, such as 24 hours, unless product requirements
demand otherwise. Cleanup should delete an output group (annotated media,
detections, and summary) atomically by request ID and must never scan or remove
outside the configured output root. Monitor disk quota and reject new uploads
before exhaustion.

Temporary uploads already use random names and are removed after handled
success/failure paths. Platform-level cleanup should still cover process kills
and host crashes.

## Authentication and abuse controls

Before an internet-facing deployment:

- require an API key or authenticated frontend session;
- rate-limit by user/IP and separately cap concurrent video jobs;
- reject anonymous cross-origin requests;
- log request IDs, status, size, media kind, and duration without retaining
  uploaded media or sensitive filenames in logs;
- add request queue/backpressure behavior instead of allowing unbounded work;
- return generic internal errors while keeping server tracebacks private.

The detector is not a certified alarm system. User-facing language must not
claim guaranteed hazard detection or replace physical fire-safety systems.

## Reverse proxy and transport

Terminate TLS at the platform or reverse proxy. Configure the proxy to:

- enforce the same or smaller body limit as the application;
- use explicit read/send timeouts for video responses;
- preserve trusted forwarded host/protocol headers;
- disable response buffering only if required by the hosting platform;
- attach security headers and restrict methods to the documented API;
- forward `/outputs/` only under the selected retention/access policy.

Use exact CORS origins in production. Wildcard CORS is supported for controlled
testing but should not be combined with credentialed browser access.

## Observability and health

Track startup failures, request counts, HTTP status codes, upload bytes,
processing duration, frame counts, queue depth, GPU utilization/VRAM, disk
usage, and cleanup failures. Do not treat `/health` as a full inference probe;
run a separate scheduled smoke test with a known development image when the
platform supports it.

## Hosting decision

A GPU Hugging Face Space or comparable GPU container service remains the
preferred first deployment target. Verify codec support, persistent/object
storage, maximum request duration, and GPU availability on the selected tier
before connecting the frontend.
