# Changelog

All notable changes to PyroVision are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.0.0 — 2026-08-01

### Added

- Deterministic D-Fire preparation and strict dataset verification.
- Reproducible YOLO11s training, safe pause/resume, metric history, and
  checkpoint finalization.
- One-time held-out test evaluation with aggregate/per-class metrics, curves,
  confusion matrices, and sanity samples.
- Reusable image, directory, video, and webcam inference package with strict
  checkpoint, class-order, device, media, output, and serialization contracts.
- FastAPI backend with one-time lifespan model loading, image/video endpoints,
  output retrieval, CORS, upload limits, structured schemas, errors, and
  OpenAPI documentation.
- Reproducible CPU/CUDA component, pipeline, and HTTP benchmarking with hashed
  development inputs and machine-readable results.

### Hardened

- Collision-safe atomic output publication and interruption-safe video/webcam
  cleanup.
- Validation of corrupt, unsupported, oversized, malformed, or mismatched
  media/configuration/checkpoint inputs.
- Full automated regression coverage across Steps 1–6.

### Known limitations

- The detector is a project baseline, not a certified life-safety system.
- Fire recall remains lower than smoke recall on the held-out test split.
- The HTTP API processes complete videos synchronously and does not implement
  streaming or webcam uploads.
- Public deployment still requires authentication, rate limits, duration/frame
  limits, durable storage, output retention, and production proxy policy.
