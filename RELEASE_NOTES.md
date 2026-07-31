# PyroVision AI 1.0.0 — backend release candidate

PyroVision 1.0.0 is the first release-quality backend baseline. It includes the
complete dataset, training, evaluation, local inference, and FastAPI workflow,
plus CPU/CUDA benchmarks and reproducibility records.

## Release evidence

- Selected checkpoint: epoch 54
- Checkpoint SHA-256:
  `21812ec7917bda5ad004fc085ba6a9d8ee1b375c95db2efe754463fc430d28c3`
- Held-out test: precision 0.7657, recall 0.6992, mAP50 0.7642,
  mAP50–95 0.4526
- Formal benchmark hardware: NVIDIA GeForce RTX 4050 Laptop GPU and the local
  14-core/20-thread Intel CPU
- Automated release gate: documented in `metrics/step6_release.json`
- Benchmark evidence: `metrics/step6_benchmark.json`

## Public interfaces

- Python package: `pyrovision`
- Local CLI: `scripts/infer.py`
- Benchmark CLI: `scripts/benchmark.py`
- API launcher: `python -m pyrovision.api` or installed `pyrovision-api`
- HTTP: `GET /health`, `POST /predict/image`, `POST /predict/video`

## Upgrade and compatibility

This release does not change the established inference or HTTP contracts.
`DetectorEngine.predict_frame_timed()` is additive and returns the same
`FrameResult` alongside timing metadata. Existing callers can continue using
`predict_frame()` unchanged.

## Deployment status

The backend is release-ready for controlled integration and frontend
development. It is not yet approved for an unauthenticated public deployment.
Review `docs/deployment.md` before hosting it outside a trusted environment.

Recommended tag after final review: `v1.0.0`.
