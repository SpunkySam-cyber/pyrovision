"""Typed and validated configuration for local PyroVision inference."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from .errors import ConfigurationError
from .yaml_utils import load_unique_yaml


SUPPORTED_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_CODEC_PATTERN = re.compile(r"^[A-Za-z0-9]{4}$")


def _validate_probability(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ConfigurationError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint selection and integrity contract."""

    path: str | Path = "auto"
    metrics_file: Path = Path("metrics/yolo11s_baseline.json")
    verify_sha256: bool = True
    sha256: str | None = None
    expected_classes: tuple[str, ...] = ("smoke", "fire")

    def __post_init__(self) -> None:
        if not isinstance(self.path, (str, Path)):
            raise ConfigurationError("checkpoint.path must be 'auto' or a file path")
        if self.path != "auto" and not str(self.path).strip():
            raise ConfigurationError("checkpoint.path must be 'auto' or a file path")
        if not isinstance(self.metrics_file, Path) or self.metrics_file == Path("."):
            raise ConfigurationError("checkpoint.metrics_file must be a non-empty file path")
        if not isinstance(self.verify_sha256, bool):
            raise ConfigurationError("checkpoint.verify_sha256 must be true or false")
        if self.sha256 is not None and (
            not isinstance(self.sha256, str)
            or not _SHA256_PATTERN.fullmatch(self.sha256)
        ):
            raise ConfigurationError("checkpoint.sha256 must contain 64 hexadecimal characters")
        if not isinstance(self.expected_classes, (list, tuple)) or not all(
            isinstance(name, str) for name in self.expected_classes
        ):
            raise ConfigurationError(
                "checkpoint.expected_classes must contain only class-name strings"
            )
        normalized_classes = tuple(name.strip() for name in self.expected_classes)
        if not normalized_classes or any(not name for name in normalized_classes):
            raise ConfigurationError("checkpoint.expected_classes must contain class names")
        if len(set(normalized_classes)) != len(normalized_classes):
            raise ConfigurationError("checkpoint.expected_classes must be unique")
        object.__setattr__(self, "expected_classes", normalized_classes)
        if self.sha256 is not None:
            object.__setattr__(self, "sha256", self.sha256.lower())


@dataclass(frozen=True)
class ModelConfig:
    """Model execution and detection-filtering configuration."""

    image_size: int = 640
    confidence_threshold: float = 0.35
    class_thresholds: Mapping[str, float] = field(default_factory=dict)
    iou_threshold: float = 0.7
    max_detections: int = 300
    half: bool | str = "auto"

    def __post_init__(self) -> None:
        if isinstance(self.image_size, bool) or not isinstance(self.image_size, int):
            raise ConfigurationError("model.image_size must be an integer")
        if self.image_size <= 0:
            raise ConfigurationError("model.image_size must be positive")
        _validate_probability("model.confidence_threshold", self.confidence_threshold)
        _validate_probability("model.iou_threshold", self.iou_threshold)
        if isinstance(self.max_detections, bool) or not isinstance(self.max_detections, int):
            raise ConfigurationError("model.max_detections must be an integer")
        if self.max_detections <= 0:
            raise ConfigurationError("model.max_detections must be positive")
        if not isinstance(self.half, bool) and self.half != "auto":
            raise ConfigurationError("model.half must be true, false, or 'auto'")

        if not isinstance(self.class_thresholds, Mapping):
            raise ConfigurationError("model.class_thresholds must be a mapping")
        thresholds: dict[str, float] = {}
        for class_name, threshold in self.class_thresholds.items():
            if not isinstance(class_name, str):
                raise ConfigurationError(
                    "model.class_thresholds keys must be class-name strings"
                )
            normalized_name = class_name.strip()
            if not normalized_name:
                raise ConfigurationError("model.class_thresholds contains an empty class name")
            if normalized_name in thresholds:
                raise ConfigurationError(
                    f"model.class_thresholds contains duplicate class {normalized_name!r}"
                )
            _validate_probability(
                f"model.class_thresholds.{normalized_name}", threshold
            )
            thresholds[normalized_name] = float(threshold)
        object.__setattr__(self, "confidence_threshold", float(self.confidence_threshold))
        object.__setattr__(self, "iou_threshold", float(self.iou_threshold))
        object.__setattr__(self, "class_thresholds", MappingProxyType(thresholds))


@dataclass(frozen=True)
class InputConfig:
    """Media-source defaults shared by image, video, and webcam modes."""

    source: str | None = None
    webcam_index: int = 0
    frame_skip: int = 0

    def __post_init__(self) -> None:
        if self.source is not None:
            if not isinstance(self.source, str) or not self.source.strip():
                raise ConfigurationError("input.source must be null or a non-empty string")
            object.__setattr__(self, "source", self.source.strip())
        if isinstance(self.webcam_index, bool) or not isinstance(self.webcam_index, int):
            raise ConfigurationError("input.webcam_index must be an integer")
        if self.webcam_index < 0:
            raise ConfigurationError("input.webcam_index cannot be negative")
        if isinstance(self.frame_skip, bool) or not isinstance(self.frame_skip, int):
            raise ConfigurationError("input.frame_skip must be an integer")
        if self.frame_skip < 0:
            raise ConfigurationError("input.frame_skip cannot be negative")


@dataclass(frozen=True)
class OutputConfig:
    """Output-media, detection-log, and display defaults."""

    directory: Path = Path("outputs/inference")
    save_media: bool = True
    save_detections: bool = True
    display: bool = False
    video_codec: str = "mp4v"
    video_extension: str = ".mp4"

    def __post_init__(self) -> None:
        if not isinstance(self.directory, Path) or not str(self.directory):
            raise ConfigurationError("output.directory cannot be empty")
        for name in ("save_media", "save_detections", "display"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigurationError(f"output.{name} must be true or false")
        if not isinstance(self.video_codec, str) or not _CODEC_PATTERN.fullmatch(
            self.video_codec
        ):
            raise ConfigurationError(
                "output.video_codec must contain four ASCII letters or digits"
            )
        if (
            not isinstance(self.video_extension, str)
            or not self.video_extension.startswith(".")
            or self.video_extension.lower() not in {".avi", ".mkv", ".mov", ".mp4"}
        ):
            raise ConfigurationError(
                "output.video_extension must be .avi, .mkv, .mov, or .mp4"
            )
        object.__setattr__(self, "video_extension", self.video_extension.lower())


@dataclass(frozen=True)
class InferenceConfig:
    """Complete versioned local-inference configuration."""

    schema_version: int
    checkpoint: CheckpointConfig
    device: str
    model: ModelConfig
    input: InputConfig
    output: OutputConfig
    project_root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, CheckpointConfig):
            raise ConfigurationError("checkpoint must be a CheckpointConfig")
        if not isinstance(self.model, ModelConfig):
            raise ConfigurationError("model must be a ModelConfig")
        if not isinstance(self.input, InputConfig):
            raise ConfigurationError("input must be an InputConfig")
        if not isinstance(self.output, OutputConfig):
            raise ConfigurationError("output must be an OutputConfig")
        if not isinstance(self.project_root, Path):
            raise ConfigurationError("project_root must be a path")
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise ConfigurationError("schema_version must be an integer")
        if self.schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ConfigurationError(
                f"Unsupported inference schema version {self.schema_version}; "
                f"expected {SUPPORTED_SCHEMA_VERSION}"
            )
        if not isinstance(self.device, str) or not re.fullmatch(
            r"auto|cpu|cuda(?::\d+)?", self.device.strip().lower()
        ):
            raise ConfigurationError("device must be auto, cpu, cuda, or cuda:N")
        object.__setattr__(self, "device", self.device.strip().lower())
        object.__setattr__(self, "project_root", self.project_root.resolve())
        unknown_thresholds = set(self.model.class_thresholds) - set(
            self.checkpoint.expected_classes
        )
        if unknown_thresholds:
            raise ConfigurationError(
                "model.class_thresholds contains unknown classes: "
                f"{sorted(unknown_thresholds)}"
            )


def discover_project_root(start: Path) -> Path:
    """Find the nearest project root from a configuration file location."""
    resolved = start.resolve()
    directory = resolved if resolved.is_dir() else resolved.parent
    for candidate in (directory, *directory.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    return directory


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"{context} must be a mapping")
    return value


def _reject_unknown(values: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ConfigurationError(f"Unknown {context} keys: {sorted(unknown)}")


def _project_path(value: str | Path, project_root: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigurationError("Configured paths must be non-empty strings or paths")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_inference_config(
    path: Path, project_root: Path | None = None
) -> InferenceConfig:
    """Load inference YAML, reject typos, and resolve project-owned paths."""
    config_path = path.resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Inference configuration does not exist: {config_path}")
    try:
        raw = load_unique_yaml(config_path.read_text(encoding="utf-8"))
    except ConfigurationError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot load inference configuration: {exc}") from exc
    values = _mapping(raw, "configuration")
    if "schema_version" not in values:
        raise ConfigurationError("schema_version is required")
    _reject_unknown(
        values,
        {"schema_version", "checkpoint", "device", "model", "input", "output"},
        "top-level",
    )
    root = project_root.resolve() if project_root else discover_project_root(config_path)

    checkpoint_values = _mapping(values.get("checkpoint"), "checkpoint")
    _reject_unknown(
        checkpoint_values,
        {"path", "metrics_file", "verify_sha256", "sha256", "expected_classes"},
        "checkpoint",
    )
    checkpoint_path: str | Path = checkpoint_values.get("path", "auto")
    if checkpoint_path != "auto":
        checkpoint_path = _project_path(checkpoint_path, root)
    expected_classes = checkpoint_values.get("expected_classes", ("smoke", "fire"))
    if not isinstance(expected_classes, (list, tuple)):
        raise ConfigurationError("checkpoint.expected_classes must be a list")
    checkpoint = CheckpointConfig(
        path=checkpoint_path,
        metrics_file=_project_path(
            checkpoint_values.get("metrics_file", "metrics/yolo11s_baseline.json"), root
        ),
        verify_sha256=checkpoint_values.get("verify_sha256", True),
        sha256=checkpoint_values.get("sha256"),
        expected_classes=tuple(expected_classes),
    )

    model_values = _mapping(values.get("model"), "model")
    _reject_unknown(
        model_values,
        {
            "image_size",
            "confidence_threshold",
            "class_thresholds",
            "iou_threshold",
            "max_detections",
            "half",
        },
        "model",
    )
    model = ModelConfig(
        image_size=model_values.get("image_size", 640),
        confidence_threshold=model_values.get("confidence_threshold", 0.35),
        class_thresholds=_mapping(
            model_values.get("class_thresholds"), "model.class_thresholds"
        ),
        iou_threshold=model_values.get("iou_threshold", 0.7),
        max_detections=model_values.get("max_detections", 300),
        half=model_values.get("half", "auto"),
    )

    input_values = _mapping(values.get("input"), "input")
    _reject_unknown(input_values, {"source", "webcam_index", "frame_skip"}, "input")
    input_config = InputConfig(
        source=input_values.get("source"),
        webcam_index=input_values.get("webcam_index", 0),
        frame_skip=input_values.get("frame_skip", 0),
    )

    output_values = _mapping(values.get("output"), "output")
    _reject_unknown(
        output_values,
        {
            "directory",
            "save_media",
            "save_detections",
            "display",
            "video_codec",
            "video_extension",
        },
        "output",
    )
    output = OutputConfig(
        directory=_project_path(output_values.get("directory", "outputs/inference"), root),
        save_media=output_values.get("save_media", True),
        save_detections=output_values.get("save_detections", True),
        display=output_values.get("display", False),
        video_codec=output_values.get("video_codec", "mp4v"),
        video_extension=output_values.get("video_extension", ".mp4"),
    )

    configured_device = values.get("device", "auto")
    if not isinstance(configured_device, str):
        raise ConfigurationError("device must be auto, cpu, cuda, or cuda:N")
    return InferenceConfig(
        schema_version=values.get("schema_version", SUPPORTED_SCHEMA_VERSION),
        checkpoint=checkpoint,
        device=configured_device.strip().lower(),
        model=model,
        input=input_config,
        output=output,
        project_root=root,
    )
