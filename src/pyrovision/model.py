"""Reusable YOLO detector engine with a framework-independent output boundary."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any

import numpy as np

from .checkpoints import ResolvedCheckpoint, resolve_checkpoint, validate_class_names
from .config import InferenceConfig
from .device import ResolvedDevice, resolve_device
from .errors import ConfigurationError, InferenceError
from .types import BoundingBox, Detection, FrameResult


ModelFactory = Callable[[str], Any]


def _default_model_factory(checkpoint: str) -> Any:
    from ultralytics import YOLO

    return YOLO(checkpoint)


def _tensor_list(value: Any) -> list[Any]:
    """Convert PyTorch/NumPy/fake tensor values into ordinary Python lists."""
    converted = value
    for operation in ("detach", "cpu"):
        method = getattr(converted, operation, None)
        if callable(method):
            converted = method()
    tolist = getattr(converted, "tolist", None)
    return tolist() if callable(tolist) else list(converted)


class DetectorEngine:
    """Own checkpoint/device policy and convert YOLO results into project types."""

    def __init__(
        self,
        model: Any,
        config: InferenceConfig,
        checkpoint: ResolvedCheckpoint,
        device: ResolvedDevice,
    ) -> None:
        self.model = model
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        self.class_names = validate_class_names(
            model.names, config.checkpoint.expected_classes
        )
        if config.model.half is True and not device.is_cuda:
            raise ConfigurationError("model.half=true requires a CUDA device")
        self.use_half = (
            device.use_half if config.model.half == "auto" else bool(config.model.half)
        )
        self._predict_lock = Lock()

    @classmethod
    def from_config(
        cls,
        config: InferenceConfig,
        model_factory: ModelFactory | None = None,
        torch_module: Any | None = None,
    ) -> "DetectorEngine":
        """Resolve and verify all runtime dependencies before accepting input."""
        checkpoint = resolve_checkpoint(config.checkpoint, config.project_root)
        device = resolve_device(config.device, torch_module=torch_module)
        factory = model_factory or _default_model_factory
        model = factory(str(checkpoint.path))
        return cls(model, config, checkpoint, device)

    @property
    def candidate_confidence(self) -> float:
        """Lowest configured threshold needed before per-class filtering."""
        values = [self.config.model.confidence_threshold]
        values.extend(self.config.model.class_thresholds.values())
        return min(values)

    def threshold_for(self, class_name: str) -> float:
        return self.config.model.class_thresholds.get(
            class_name, self.config.model.confidence_threshold
        )

    def predict_frame(
        self,
        frame: np.ndarray,
        *,
        source: str,
        frame_index: int = 0,
        timestamp_ms: float | None = 0.0,
    ) -> FrameResult:
        """Run one BGR frame and return deterministic framework-neutral detections."""
        if not isinstance(frame, np.ndarray):
            raise InferenceError("Inference frame must be a NumPy array")
        if frame.ndim not in (2, 3) or frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise InferenceError(f"Unsupported frame shape: {frame.shape}")
        height, width = int(frame.shape[0]), int(frame.shape[1])
        arguments = {
            "source": frame,
            "imgsz": self.config.model.image_size,
            "conf": self.candidate_confidence,
            "iou": self.config.model.iou_threshold,
            "max_det": self.config.model.max_detections,
            "device": self.device.value,
            "half": self.use_half,
            "save": False,
            "verbose": False,
        }
        try:
            with self._predict_lock:
                raw_results = self.model.predict(**arguments)
        except Exception as exc:
            raise InferenceError(f"Model inference failed: {exc}") from exc
        if len(raw_results) != 1:
            raise InferenceError(
                f"Expected one result for one frame, received {len(raw_results)}"
            )

        raw_boxes = getattr(raw_results[0], "boxes", None)
        detections: list[Detection] = []
        if raw_boxes is not None:
            coordinates = _tensor_list(raw_boxes.xyxy)
            confidences = _tensor_list(raw_boxes.conf)
            classes = _tensor_list(raw_boxes.cls)
            if not (len(coordinates) == len(confidences) == len(classes)):
                raise InferenceError("Model result arrays have inconsistent lengths")
            for raw_bbox, raw_confidence, raw_class_id in zip(
                coordinates, confidences, classes, strict=True
            ):
                class_id = int(raw_class_id)
                if not 0 <= class_id < len(self.class_names):
                    raise InferenceError(f"Model returned unknown class ID {class_id}")
                class_name = self.class_names[class_id]
                confidence = float(raw_confidence)
                if confidence < self.threshold_for(class_name):
                    continue
                if len(raw_bbox) != 4:
                    raise InferenceError("Model bounding boxes must contain four values")
                x_min, y_min, x_max, y_max = (float(value) for value in raw_bbox)
                x_min = min(max(x_min, 0.0), float(width))
                y_min = min(max(y_min, 0.0), float(height))
                x_max = min(max(x_max, 0.0), float(width))
                y_max = min(max(y_max, 0.0), float(height))
                if x_max <= x_min or y_max <= y_min:
                    continue
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=BoundingBox(x_min, y_min, x_max, y_max),
                    )
                )

        detections.sort(
            key=lambda item: (
                -item.confidence,
                item.class_id,
                item.bbox.x_min,
                item.bbox.y_min,
                item.bbox.x_max,
                item.bbox.y_max,
            )
        )
        return FrameResult(
            source=source,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            width=width,
            height=height,
            detections=detections,
        )
