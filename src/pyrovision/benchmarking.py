"""Reproducible CPU/CUDA benchmark orchestration for stabilized inference."""

from __future__ import annotations

import json
import math
import os
import platform
import re
import statistics
import subprocess
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import cv2
import numpy as np

from . import __version__
from .annotation import annotate_frame
from .api.app import create_app
from .api.config import BackendConfig
from .config import InferenceConfig, load_inference_config
from .errors import ConfigurationError, InputMediaError, OutputMediaError
from .hashing import sha256_file
from .images import infer_image
from .model import DetectorEngine
from .outputs import AnnotatedVideoWriter, allocate_output_stem, write_json_atomic
from .sources import VideoReader
from .timing import PredictionTiming, TimedFrameResult
from .video import infer_video
from .yaml_utils import load_unique_yaml


BENCHMARK_SCHEMA_VERSION = 1


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class BenchmarkConfig:
    """Validated paths and iteration counts for one benchmark run."""

    schema_version: int
    inference_config: Path
    image: Path
    video: Path
    output_directory: Path
    report_path: Path
    devices: tuple[str, ...]
    warmup_iterations: int = 3
    image_iterations: int = 10
    video_runs: int = 3
    api_image_iterations: int = 3
    api_video_runs: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != BENCHMARK_SCHEMA_VERSION:
            raise ConfigurationError(
                f"Unsupported benchmark schema version {self.schema_version}; "
                f"expected {BENCHMARK_SCHEMA_VERSION}"
            )
        for name in (
            "inference_config",
            "image",
            "video",
            "output_directory",
            "report_path",
        ):
            if not isinstance(getattr(self, name), Path):
                raise ConfigurationError(f"benchmark.{name} must be a path")
        if not self.devices or len(set(self.devices)) != len(self.devices):
            raise ConfigurationError("benchmark.devices must be non-empty and unique")
        if not all(
            isinstance(device, str)
            and re.fullmatch(
                r"cpu|cuda(?::\d+)?",
                device.strip().lower(),
            )
            for device in self.devices
        ):
            raise ConfigurationError(
                "benchmark.devices may contain cpu, cuda, and cuda:N"
            )
        for name in (
            "warmup_iterations",
            "image_iterations",
            "video_runs",
            "api_image_iterations",
            "api_video_runs",
        ):
            _positive_integer(getattr(self, name), f"benchmark.{name}")
        object.__setattr__(
            self,
            "devices",
            tuple(device.strip().lower() for device in self.devices),
        )


def _resolve_path(value: Any, root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"benchmark.{name} must be a non-empty path")
    path = Path(value.strip())
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_benchmark_config(path: Path, project_root: Path) -> BenchmarkConfig:
    """Load the versioned benchmark YAML and reject unknown fields."""
    config_path = path.resolve()
    try:
        values = load_unique_yaml(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"Cannot load benchmark configuration: {exc}") from exc
    if not isinstance(values, dict):
        raise ConfigurationError("Benchmark configuration must be a mapping")
    allowed = {
        "schema_version",
        "inference_config",
        "image",
        "video",
        "output_directory",
        "report_path",
        "devices",
        "warmup_iterations",
        "image_iterations",
        "video_runs",
        "api_image_iterations",
        "api_video_runs",
    }
    unknown = set(values) - allowed
    if unknown:
        raise ConfigurationError(f"Unknown benchmark keys: {sorted(unknown)}")
    devices = values.get("devices", ["cpu", "cuda:0"])
    if not isinstance(devices, list) or not all(
        isinstance(device, str) for device in devices
    ):
        raise ConfigurationError("benchmark.devices must be a list of strings")
    root = project_root.resolve()
    return BenchmarkConfig(
        schema_version=values.get("schema_version", 0),
        inference_config=_resolve_path(
            values.get("inference_config", "configs/inference.yaml"),
            root,
            "inference_config",
        ),
        image=_resolve_path(values.get("image", ""), root, "image"),
        video=_resolve_path(values.get("video", ""), root, "video"),
        output_directory=_resolve_path(
            values.get("output_directory", "outputs/benchmark"),
            root,
            "output_directory",
        ),
        report_path=_resolve_path(
            values.get("report_path", "metrics/step6_benchmark.json"),
            root,
            "report_path",
        ),
        devices=tuple(devices),
        warmup_iterations=values.get("warmup_iterations", 3),
        image_iterations=values.get("image_iterations", 10),
        video_runs=values.get("video_runs", 3),
        api_image_iterations=values.get("api_image_iterations", 3),
        api_video_runs=values.get("api_video_runs", 1),
    )


def summarize_samples(values: Iterable[float]) -> dict[str, float | int]:
    """Return deterministic descriptive statistics using interpolated p95."""
    samples = [float(value) for value in values]
    if not samples or not all(math.isfinite(value) and value >= 0.0 for value in samples):
        raise ValueError("Timing samples must be non-empty, finite, and non-negative")
    ordered = sorted(samples)
    position = 0.95 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    p95 = ordered[lower]
    if upper != lower:
        p95 += (ordered[upper] - ordered[lower]) * (position - lower)
    return {
        "count": len(samples),
        "mean": round(statistics.fmean(samples), 3),
        "median": round(statistics.median(samples), 3),
        "p95": round(p95, 3),
        "min": round(min(samples), 3),
        "max": round(max(samples), 3),
        "stdev": round(statistics.pstdev(samples), 3),
    }


def _timing_summary(timings: list[PredictionTiming]) -> dict[str, Any]:
    if not timings:
        raise ValueError("At least one prediction timing is required")

    def present(name: str) -> dict[str, float | int] | None:
        samples = [getattr(timing, name) for timing in timings]
        values = [value for value in samples if value is not None]
        return summarize_samples(values) if values else None

    inference = present("inference_ms")
    model_only_fps = None
    if inference is not None and float(inference["mean"]) > 0.0:
        model_only_fps = round(1000.0 / float(inference["mean"]), 3)
    return {
        "preprocessing_ms": present("preprocessing_ms"),
        "model_inference_ms": inference,
        "framework_postprocessing_ms": present(
            "framework_postprocessing_ms"
        ),
        "project_postprocessing_ms": present("project_postprocessing_ms"),
        "combined_postprocessing_ms": summarize_samples(
            timing.postprocessing_ms for timing in timings
        ),
        "model_call_ms": summarize_samples(
            timing.model_call_ms for timing in timings
        ),
        "engine_total_ms": summarize_samples(
            timing.engine_total_ms for timing in timings
        ),
        "model_only_fps": model_only_fps,
    }


class _TimedEngineProxy:
    """Collect timings while preserving the engine contract used by pipelines."""

    def __init__(self, engine: DetectorEngine) -> None:
        self.engine = engine
        self.checkpoint = engine.checkpoint
        self.device = engine.device
        self.class_names = engine.class_names
        self.timings: list[PredictionTiming] = []

    def predict_frame(self, *args: Any, **kwargs: Any) -> Any:
        timed = self.engine.predict_frame_timed(*args, **kwargs)
        self.timings.append(timed.timing)
        return timed.result


def _synchronize(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize()


def _timed_call(device: str, operation: Any) -> tuple[Any, float]:
    _synchronize(device)
    started = perf_counter()
    result = operation()
    _synchronize(device)
    return result, (perf_counter() - started) * 1000.0


def _decode_image(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise InputMediaError(f"OpenCV could not decode benchmark image: {path}")
    return frame


def _benchmark_image_components(
    engine: DetectorEngine,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    frame = _decode_image(config.image)
    warmup: list[PredictionTiming] = []
    for index in range(config.warmup_iterations):
        warmup.append(
            engine.predict_frame_timed(
                frame,
                source=str(config.image),
                frame_index=index,
                timestamp_ms=0.0,
            ).timing
        )

    decode_ms: list[float] = []
    annotation_ms: list[float] = []
    encoding_ms: list[float] = []
    serialization_ms: list[float] = []
    timings: list[PredictionTiming] = []
    suffix = config.image.suffix.lower()
    for index in range(config.image_iterations):
        decoded, elapsed = _timed_call(
            engine.device.value,
            lambda: _decode_image(config.image),
        )
        decode_ms.append(elapsed)
        timed: TimedFrameResult = engine.predict_frame_timed(
            decoded,
            source=str(config.image),
            frame_index=index,
            timestamp_ms=0.0,
        )
        timings.append(timed.timing)
        annotated, elapsed = _timed_call(
            engine.device.value,
            lambda: annotate_frame(decoded, timed.result),
        )
        annotation_ms.append(elapsed)
        encoded, elapsed = _timed_call(
            engine.device.value,
            lambda: cv2.imencode(suffix, annotated),
        )
        if not encoded[0]:
            raise OutputMediaError("OpenCV could not encode benchmark image")
        encoding_ms.append(elapsed)
        _, elapsed = _timed_call(
            engine.device.value,
            lambda: json.dumps(
                timed.result.to_dict(),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
        serialization_ms.append(elapsed)
    return {
        "warmup": _timing_summary(warmup),
        "steady_state": _timing_summary(timings),
        "decode_ms": summarize_samples(decode_ms),
        "annotation_ms": summarize_samples(annotation_ms),
        "output_encoding_ms": summarize_samples(encoding_ms),
        "json_serialization_ms": summarize_samples(serialization_ms),
    }


def _benchmark_image_pipeline(
    engine: DetectorEngine,
    config: BenchmarkConfig,
    device_output: Path,
) -> dict[str, Any]:
    proxy = _TimedEngineProxy(engine)
    durations: list[float] = []
    for _ in range(config.image_iterations):
        _, elapsed = _timed_call(
            engine.device.value,
            lambda: infer_image(
                proxy,
                config.image,
                output_directory=device_output / "image-pipeline",
            ),
        )
        durations.append(elapsed)
    mean_ms = statistics.fmean(durations)
    return {
        "end_to_end_ms": summarize_samples(durations),
        "end_to_end_fps": round(1000.0 / mean_ms, 3),
        "prediction": _timing_summary(proxy.timings),
    }


def _video_frames(path: Path) -> tuple[list[Any], float]:
    started = perf_counter()
    with VideoReader(path) as reader:
        frames = list(reader.frames())
    if not frames:
        raise InputMediaError("Benchmark video contains no frames")
    return frames, (perf_counter() - started) * 1000.0


def _benchmark_video_components(
    engine: DetectorEngine,
    config: BenchmarkConfig,
    inference: InferenceConfig,
    device_output: Path,
) -> dict[str, Any]:
    frames, decode_total_ms = _video_frames(config.video)
    timings: list[PredictionTiming] = []
    annotation_ms: list[float] = []
    serialization_ms: list[float] = []
    annotated_frames: list[np.ndarray] = []
    for source_frame in frames:
        timed = engine.predict_frame_timed(
            source_frame.image,
            source=str(config.video),
            frame_index=source_frame.frame_index,
            timestamp_ms=source_frame.timestamp_ms,
        )
        timings.append(timed.timing)
        annotated, elapsed = _timed_call(
            engine.device.value,
            lambda: annotate_frame(source_frame.image, timed.result),
        )
        annotated_frames.append(annotated)
        annotation_ms.append(elapsed)
        record = {
            "record_type": "frame",
            "processed_index": len(annotated_frames) - 1,
            **timed.result.to_dict(),
        }
        _, elapsed = _timed_call(
            engine.device.value,
            lambda: json.dumps(
                record,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
        serialization_ms.append(elapsed)

    output_dir = device_output / "video-components"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = allocate_output_stem(
        output_dir,
        "encoding",
        (f"{inference.output.video_extension}",),
    )
    output_path = output_dir / f"{stem}{inference.output.video_extension}"
    metadata_fps = 1000.0 / max(
        frames[1].timestamp_ms - frames[0].timestamp_ms,
        1.0,
    ) if len(frames) > 1 else 30.0
    writer = AnnotatedVideoWriter(
        output_path,
        codec=inference.output.video_codec,
        fps=metadata_fps,
        width=annotated_frames[0].shape[1],
        height=annotated_frames[0].shape[0],
    )
    encoding_ms: list[float] = []
    try:
        for frame in annotated_frames:
            _, elapsed = _timed_call(
                engine.device.value,
                lambda frame=frame: writer.write(frame),
            )
            encoding_ms.append(elapsed)
        _, finalization_ms = _timed_call(engine.device.value, writer.close)
    finally:
        writer.close()
    return {
        "frames": len(frames),
        "decode_total_ms": round(decode_total_ms, 3),
        "decode_ms_per_frame": round(decode_total_ms / len(frames), 3),
        "prediction": _timing_summary(timings),
        "annotation_ms": summarize_samples(annotation_ms),
        "output_encoding_ms": summarize_samples(encoding_ms),
        "output_finalization_ms": round(finalization_ms, 3),
        "json_serialization_ms": summarize_samples(serialization_ms),
    }


def _benchmark_video_pipeline(
    engine: DetectorEngine,
    config: BenchmarkConfig,
    inference: InferenceConfig,
    device_output: Path,
) -> dict[str, Any]:
    proxy = _TimedEngineProxy(engine)
    durations: list[float] = []
    processed_frames: list[int] = []
    for _ in range(config.video_runs):
        output, elapsed = _timed_call(
            engine.device.value,
            lambda: infer_video(
                proxy,
                config.video,
                output_directory=device_output / "video-pipeline",
                frame_skip=inference.input.frame_skip,
                codec=inference.output.video_codec,
                video_extension=inference.output.video_extension,
            ),
        )
        durations.append(elapsed)
        processed_frames.append(output.summary.frames_processed)
    total_frames = sum(processed_frames)
    total_seconds = sum(durations) / 1000.0
    return {
        "runs": config.video_runs,
        "frames_per_run": processed_frames,
        "end_to_end_ms": summarize_samples(durations),
        "end_to_end_fps": round(total_frames / total_seconds, 3),
        "prediction": _timing_summary(proxy.timings),
    }


def _benchmark_api(
    engine: DetectorEngine,
    config: BenchmarkConfig,
    inference: InferenceConfig,
    device_output: Path,
) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    backend = BackendConfig(
        inference=inference,
        cors_origins=(),
        max_upload_size_bytes=max(
            config.image.stat().st_size,
            config.video.stat().st_size,
        ) + 1024,
        output_directory=device_output / "api",
        temporary_directory=device_output / "api-temp",
    )
    app = create_app(backend, engine_factory=lambda _: engine)
    image_bytes = config.image.read_bytes()
    video_bytes = config.video.read_bytes()
    image_durations: list[float] = []
    video_durations: list[float] = []
    video_frames: list[int] = []
    with TestClient(app) as client:
        health = client.get("/health")
        if health.status_code != 200:
            raise RuntimeError(f"Benchmark API health failed: {health.text}")
        warmup_response, api_warmup_ms = _timed_call(
            engine.device.value,
            lambda: client.post(
                "/predict/image",
                files={
                    "file": (
                        config.image.name,
                        image_bytes,
                        "image/jpeg",
                    )
                },
            ),
        )
        if warmup_response.status_code != 200:
            raise RuntimeError(
                f"Benchmark API warm-up failed: {warmup_response.text}"
            )
        for _ in range(config.api_image_iterations):
            response, elapsed = _timed_call(
                engine.device.value,
                lambda: client.post(
                    "/predict/image",
                    files={
                        "file": (
                            config.image.name,
                            image_bytes,
                            "image/jpeg",
                        )
                    },
                ),
            )
            if response.status_code != 200:
                raise RuntimeError(f"Benchmark image API failed: {response.text}")
            image_durations.append(elapsed)
        for _ in range(config.api_video_runs):
            response, elapsed = _timed_call(
                engine.device.value,
                lambda: client.post(
                    "/predict/video",
                    files={
                        "file": (
                            config.video.name,
                            video_bytes,
                            "video/mp4",
                        )
                    },
                ),
            )
            if response.status_code != 200:
                raise RuntimeError(f"Benchmark video API failed: {response.text}")
            video_durations.append(elapsed)
            video_frames.append(response.json()["processed_frames"])
    return {
        "warmup_image_request_ms": round(api_warmup_ms, 3),
        "image_request_end_to_end_ms": summarize_samples(image_durations),
        "video_request_end_to_end_ms": summarize_samples(video_durations),
        "video_end_to_end_fps": round(
            sum(video_frames) / (sum(video_durations) / 1000.0),
            3,
        ),
        "model_reused": app.state.model_load_count == 1,
    }


def _software() -> dict[str, Any]:
    import fastapi
    import numpy
    import pydantic
    import torch
    import ultralytics
    import uvicorn

    return {
        "python": platform.python_version(),
        "pyrovision": __version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "numpy": numpy.__version__,
        "fastapi": fastapi.__version__,
        "pydantic": pydantic.__version__,
        "uvicorn": uvicorn.__version__,
    }


def _hardware() -> dict[str, Any]:
    import psutil
    import torch

    cuda = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        properties = torch.cuda.get_device_properties(0)
        cuda = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
        }
    return {
        "platform": platform.platform(),
        "processor": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_bytes": psutil.virtual_memory().total,
        "cuda_device_0": cuda,
    }


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _report_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace(
            "\\",
            "/",
        )
    except ValueError:
        return str(path)


def merge_device_reports(
    reports: list[dict[str, Any]],
    config: BenchmarkConfig,
    project_root: Path,
) -> dict[str, Any]:
    """Merge isolated device reports without mutating their source objects."""
    if len(reports) != len(config.devices) or not reports:
        raise ValueError("One benchmark report is required for each device")
    merged = deepcopy(reports[0])
    merged["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    merged["devices"] = {}
    for report in reports:
        merged["devices"].update(report["devices"])
        if report["hardware"].get("cuda_device_0") is not None:
            merged["hardware"]["cuda_device_0"] = report["hardware"][
                "cuda_device_0"
            ]
    if set(merged["devices"]) != set(config.devices):
        raise ValueError("Isolated reports do not match configured devices")
    merged["configuration"]["devices"] = list(config.devices)
    merged["configuration"]["report_path"] = _report_path(
        config.report_path,
        project_root,
    )
    merged["methodology"]["device_process_isolation"] = True
    return merged


def run_benchmarks(config: BenchmarkConfig, project_root: Path) -> dict[str, Any]:
    """Run the configured sequential benchmark and publish its JSON report."""
    if len(config.devices) != 1:
        raise ConfigurationError(
            "Run each benchmark device in an isolated process"
        )
    for path, label in (
        (config.image, "image"),
        (config.video, "video"),
        (config.inference_config, "inference configuration"),
    ):
        if not path.is_file():
            raise InputMediaError(f"Benchmark {label} does not exist: {path}")
    config.output_directory.mkdir(parents=True, exist_ok=True)
    inference = load_inference_config(
        config.inference_config,
        project_root=project_root,
    )
    devices: dict[str, Any] = {}
    checkpoint_sha256: str | None = None
    for device in config.devices:
        device_config = replace(inference, device=device)
        engine, model_load_ms = _timed_call(
            device,
            lambda: DetectorEngine.from_config(device_config),
        )
        safe_device = device.replace(":", "-")
        device_output = config.output_directory / safe_device
        if checkpoint_sha256 is None:
            checkpoint_sha256 = engine.checkpoint.sha256
        elif checkpoint_sha256 != engine.checkpoint.sha256:
            raise RuntimeError("Benchmark devices resolved different checkpoints")
        devices[device] = {
            "model_loading_ms": round(model_load_ms, 3),
            "resolved_device": engine.device.value,
            "half_precision": engine.use_half,
            "image_components": _benchmark_image_components(engine, config),
            "image_pipeline": _benchmark_image_pipeline(
                engine,
                config,
                device_output,
            ),
            "video_components": _benchmark_video_components(
                engine,
                config,
                device_config,
                device_output,
            ),
            "video_pipeline": _benchmark_video_pipeline(
                engine,
                config,
                device_config,
                device_output,
            ),
            "api": _benchmark_api(
                engine,
                config,
                device_config,
                device_output,
            ),
        }

    report = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "checkpoint": {
            "sha256": checkpoint_sha256,
        },
        "configuration": {
            "schema_version": config.schema_version,
            "inference_config": _report_path(
                config.inference_config,
                project_root,
            ),
            "image": _report_path(config.image, project_root),
            "video": _report_path(config.video, project_root),
            "output_directory": _report_path(
                config.output_directory,
                project_root,
            ),
            "report_path": _report_path(config.report_path, project_root),
            "devices": list(config.devices),
            "warmup_iterations": config.warmup_iterations,
            "image_iterations": config.image_iterations,
            "video_runs": config.video_runs,
            "api_image_iterations": config.api_image_iterations,
            "api_video_runs": config.api_video_runs,
            "inference": {
                "image_size": inference.model.image_size,
                "confidence_threshold": inference.model.confidence_threshold,
                "class_thresholds": dict(inference.model.class_thresholds),
                "iou_threshold": inference.model.iou_threshold,
                "max_detections": inference.model.max_detections,
                "video_codec": inference.output.video_codec,
                "video_extension": inference.output.video_extension,
                "frame_skip": inference.input.frame_skip,
            },
        },
        "inputs": {
            "image": {
                "path": _report_path(config.image, project_root),
                "sha256": sha256_file(config.image),
                "bytes": config.image.stat().st_size,
            },
            "video": {
                "path": _report_path(config.video, project_root),
                "sha256": sha256_file(config.video),
                "bytes": config.video.stat().st_size,
            },
        },
        "hardware": _hardware(),
        "software": _software(),
        "methodology": {
            "execution": "sequential, one request/frame at a time",
            "warmup_excluded_from_steady_state": True,
            "cuda_synchronization": True,
            "framework_stage_source": "Ultralytics Results.speed",
            "percentile": "linearly interpolated p95",
            "test_split_used": False,
        },
        "devices": devices,
    }
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(config.report_path, report)
    return report
