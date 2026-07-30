"""Stable project-owned types at the model/inference boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


def _rounded(value: float, precision: int) -> float:
    return round(float(value), precision)


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-space XYXY bounding box."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(_is_finite_number(value) for value in values):
            raise ValueError("Bounding-box coordinates must be finite")
        if self.x_min < 0 or self.y_min < 0:
            raise ValueError("Bounding-box minimum coordinates cannot be negative")
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("Bounding-box maximums must not be below minimums")

    def to_list(self, precision: int = 4) -> list[float]:
        return [
            _rounded(self.x_min, precision),
            _rounded(self.y_min, precision),
            _rounded(self.x_max, precision),
            _rounded(self.y_max, precision),
        ]


@dataclass(frozen=True)
class Detection:
    """Framework-independent object detection."""

    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

    def __post_init__(self) -> None:
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int):
            raise ValueError("Detection class_id must be an integer")
        if self.class_id < 0:
            raise ValueError("Detection class_id cannot be negative")
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("Detection class_name cannot be empty")
        if not _is_finite_number(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection confidence must be between 0 and 1")

    def to_dict(self, precision: int = 6) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class": self.class_name,
            "confidence": _rounded(self.confidence, precision),
            "bbox": self.bbox.to_list(precision=4),
        }


@dataclass(frozen=True)
class FrameResult:
    """Detections associated with one source frame."""

    source: str
    frame_index: int
    timestamp_ms: float | None
    width: int
    height: int
    detections: tuple[Detection, ...]

    def __init__(
        self,
        source: str,
        frame_index: int,
        timestamp_ms: float | None,
        width: int,
        height: int,
        detections: Iterable[Detection] = (),
    ) -> None:
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "timestamp_ms", timestamp_ms)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "detections", tuple(detections))
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("Frame source cannot be empty")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise ValueError("Frame index must be an integer")
        if self.frame_index < 0:
            raise ValueError("Frame index cannot be negative")
        if self.timestamp_ms is not None and (
            not _is_finite_number(self.timestamp_ms) or self.timestamp_ms < 0
        ):
            raise ValueError("Frame timestamp must be non-negative and finite")
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or self.width <= 0
            or self.height <= 0
        ):
            raise ValueError("Frame dimensions must be positive")
        if not all(isinstance(detection, Detection) for detection in self.detections):
            raise TypeError("Frame detections must contain Detection objects")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "frame_index": self.frame_index,
            "timestamp_ms": (
                _rounded(self.timestamp_ms, 3) if self.timestamp_ms is not None else None
            ),
            "width": self.width,
            "height": self.height,
            "detections": [detection.to_dict() for detection in self.detections],
        }
