# Step 6 performance benchmark

Status: **complete**

The stabilized PyroVision 1.0 inference and HTTP pipelines were benchmarked on
CPU and CUDA without changing model weights, thresholds, preprocessing, or
postprocessing algorithms. The complete machine-readable report is
`metrics/step6_benchmark.json`.

## Reproduce the benchmark

The versioned configuration is `configs/benchmark.yaml`:

```powershell
.venv\Scripts\python.exe -B scripts\benchmark.py `
  --config configs\benchmark.yaml
```

CPU and CUDA run in isolated Python processes. This prevents one device's
Ultralytics initialization from changing the other's CUDA visibility or
one-time initialization state.

## Inputs and controls

- Image: `media/dev/milestone5_images/04_combined.jpg`
- Image SHA-256:
  `6258179f096a2030c87215c4a7587f795b0755ae6871b9be8d5006d30311f30e`
- Video: `media/dev/milestone3_train_sequence.mp4`
- Video SHA-256:
  `c63efe196531df6eca1deb291143e6473771c1ea9713d63c9c58340089e7b606`
- Video frames: 12 at 640×360 and 4 FPS
- Image warm-ups: 3 per device
- Steady image iterations: 10 per component/pipeline
- Full video runs: 3 per device
- Steady HTTP image requests: 3 after one HTTP warm-up
- HTTP video requests: 1
- Execution: sequential, one frame/request at a time
- CUDA synchronization: before and after wall-clock measurements
- Held-out test split used: no

Ultralytics `Results.speed` supplies preprocessing, model inference, and
framework postprocessing time. Project conversion, annotation, encoding, JSON,
pipeline, and HTTP measurements use `perf_counter`. P95 uses linear
interpolation. Warm-up measurements are reported but excluded from steady-state
statistics.

## Hardware and software

- OS: Windows 10 build 26200, 64 bit
- CPU: Intel64 Family 6 Model 151, 14 physical / 20 logical cores
- RAM: 16.89 GB
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6.44 GB, compute capability 8.9
- Python: 3.11.4
- PyTorch: 2.8.0+cu129
- CUDA runtime: 12.9
- Ultralytics: 8.4.37
- OpenCV: 5.0.0
- FastAPI: 0.141.1
- Checkpoint SHA-256:
  `21812ec7917bda5ad004fc085ba6a9d8ee1b375c95db2efe754463fc430d28c3`

The report was generated from benchmark revision `9983e89`; the subsequent
Version 1.0 metadata/documentation changes do not modify inference algorithms.

## Image results

All times are steady-state means in milliseconds unless marked FPS.

| Stage | CPU | CUDA |
| --- | ---: | ---: |
| Model/checkpoint loading | 3416.791 | 1535.963 |
| Warm-up engine latency | 371.203 | 1701.732 |
| Preprocessing | 2.410 | 1.714 |
| Model inference | 293.546 | 16.308 |
| Framework postprocessing | 1.538 | 2.656 |
| Project postprocessing | 0.147 | 0.495 |
| Annotation | 4.045 | 4.279 |
| Image encoding | 3.896 | 3.881 |
| JSON serialization | 0.095 | 0.103 |
| Full image pipeline | 307.801 | 55.890 |
| Model-only FPS | 3.407 | 61.320 |
| End-to-end pipeline FPS | 3.249 | 17.892 |
| HTTP image warm-up | 349.910 | 2692.738 |
| HTTP image steady state | 264.680 | 79.668 |

The high first CUDA warm-up includes device/model initialization. The separate
HTTP warm-up includes thread-pool CUDA context initialization. Neither is
included in steady-state means.

## Video results

The full-pipeline latency is for the complete 12-frame clip.

| Stage | CPU | CUDA |
| --- | ---: | ---: |
| Decode total | 42.433 | 61.301 |
| Decode per frame | 3.536 | 5.108 |
| Preprocessing per frame | 1.873 | 1.569 |
| Model inference per frame | 276.824 | 16.309 |
| Combined postprocessing per frame | 1.540 | 2.329 |
| Annotation per frame | 0.257 | 0.254 |
| Video encoding per frame | 3.474 | 1.838 |
| JSON serialization per frame | 0.063 | 0.067 |
| Full video pipeline | 3208.373 | 318.795 |
| Model-only FPS | 3.612 | 61.316 |
| End-to-end pipeline FPS | 3.740 | 37.642 |
| HTTP video request | 4382.351 | 463.772 |
| HTTP video FPS | 2.738 | 25.875 |

## Interpretation

CUDA reduces steady image model latency by about 18× and raises full video
throughput from about 3.7 FPS to 37.6 FPS on this development clip. CPU remains
functional but is unsuitable for high-frame-rate real-time processing with
YOLO11s at 640 px on this machine.

HTTP overhead is modest after warm-up for images and includes multipart
handling, thread-pool dispatch, output publication, response validation, and
serialization. Video requests remain synchronous and scale with decoded frame
count; the 12-frame result must not be extrapolated directly to long uploads.

These measurements are a single-machine engineering baseline, not a universal
service-level guarantee. Power mode, thermals, driver, codec, input resolution,
detection count, storage, and concurrent load can materially change results.
