"""Deterministic OpenCV rendering for project-owned detection results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import cv2
import numpy as np

from .types import FrameResult


@dataclass(frozen=True)
class AnnotationStyle:
    """Visual settings for bounding boxes and labels in BGR color order."""

    colors: Mapping[str, tuple[int, int, int]] = field(
        default_factory=lambda: {
            "smoke": (160, 160, 160),
            "fire": (0, 0, 255),
        }
    )
    default_color: tuple[int, int, int] = (0, 255, 255)
    font_scale: float = 0.6
    line_thickness: int = 2
    text_thickness: int = 1


def annotate_frame(
    frame: np.ndarray,
    result: FrameResult,
    style: AnnotationStyle | None = None,
) -> np.ndarray:
    """Return an annotated copy without mutating the caller's frame."""
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
        raise ValueError("Annotation frame must be a two- or three-dimensional array")
    if frame.shape[0] != result.height or frame.shape[1] != result.width:
        raise ValueError("Frame dimensions do not match the detection result")
    selected_style = style or AnnotationStyle()
    annotated = frame.copy()
    for detection in result.detections:
        color = selected_style.colors.get(
            detection.class_name, selected_style.default_color
        )
        left = int(round(detection.bbox.x_min))
        top = int(round(detection.bbox.y_min))
        right = int(round(detection.bbox.x_max))
        bottom = int(round(detection.bbox.y_max))
        cv2.rectangle(
            annotated,
            (left, top),
            (right, bottom),
            color,
            selected_style.line_thickness,
        )
        label = f"{detection.class_name} {detection.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            selected_style.font_scale,
            selected_style.text_thickness,
        )
        label_top = max(0, top - text_height - baseline - 6)
        label_bottom = min(result.height - 1, label_top + text_height + baseline + 6)
        label_right = min(result.width - 1, left + text_width + 8)
        cv2.rectangle(
            annotated,
            (left, label_top),
            (label_right, label_bottom),
            color,
            -1,
        )
        text_origin = (left + 4, min(label_bottom - baseline - 3, result.height - 1))
        cv2.putText(
            annotated,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            selected_style.font_scale,
            (255, 255, 255),
            selected_style.text_thickness,
            cv2.LINE_AA,
        )
    return annotated
