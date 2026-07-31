"""Project-owned timing types for reproducible inference benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import FrameResult


@dataclass(frozen=True)
class PredictionTiming:
    """One frame's model and project postprocessing timings in milliseconds."""

    preprocessing_ms: float | None
    inference_ms: float | None
    framework_postprocessing_ms: float | None
    project_postprocessing_ms: float
    model_call_ms: float
    engine_total_ms: float

    @property
    def postprocessing_ms(self) -> float:
        return (self.framework_postprocessing_ms or 0.0) + (
            self.project_postprocessing_ms
        )

    @property
    def model_only_fps(self) -> float | None:
        if self.inference_ms is None or self.inference_ms <= 0.0:
            return None
        return 1000.0 / self.inference_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "preprocessing_ms": self.preprocessing_ms,
            "inference_ms": self.inference_ms,
            "framework_postprocessing_ms": self.framework_postprocessing_ms,
            "project_postprocessing_ms": self.project_postprocessing_ms,
            "postprocessing_ms": self.postprocessing_ms,
            "model_call_ms": self.model_call_ms,
            "engine_total_ms": self.engine_total_ms,
            "model_only_fps": self.model_only_fps,
        }


@dataclass(frozen=True)
class TimedFrameResult:
    """Framework-neutral detections paired with non-functional timing data."""

    result: FrameResult
    timing: PredictionTiming
