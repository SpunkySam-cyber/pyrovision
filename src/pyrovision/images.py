"""Still-image input, annotation, and structured output orchestration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .annotation import AnnotationStyle, annotate_frame
from .errors import InputMediaError, OutputMediaError
from .model import DetectorEngine
from .types import FrameResult


SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class ImageInferenceOutput:
    """In-memory and on-disk products from one image inference request."""

    result: FrameResult
    annotated_frame: np.ndarray
    annotated_media: Path | None
    detections_file: Path | None
    checkpoint_sha256: str
    device: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": True,
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": self.device,
            "annotated_media": (
                str(self.annotated_media) if self.annotated_media is not None else None
            ),
            "detections_file": (
                str(self.detections_file) if self.detections_file is not None else None
            ),
            "result": self.result.to_dict(),
        }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise OutputMediaError(f"Cannot write detection output {path}: {exc}") from exc


def infer_image(
    engine: DetectorEngine,
    source: Path,
    *,
    output_directory: Path,
    save_media: bool = True,
    save_detections: bool = True,
    annotation_style: AnnotationStyle | None = None,
) -> ImageInferenceOutput:
    """Read, infer, annotate, and optionally persist one supported image."""
    source_path = source.resolve()
    if not source_path.is_file():
        raise InputMediaError(f"Input image does not exist: {source_path}")
    if source_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise InputMediaError(f"Unsupported image type: {source_path.suffix or '<none>'}")
    frame = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise InputMediaError(f"OpenCV could not decode image: {source_path}")

    result = engine.predict_frame(
        frame,
        source=str(source_path),
        frame_index=0,
        timestamp_ms=0.0,
    )
    annotated = annotate_frame(frame, result, style=annotation_style)
    output_root = output_directory.resolve()
    annotated_path: Path | None = None
    detections_path: Path | None = None
    if save_media or save_detections:
        try:
            output_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OutputMediaError(
                f"Cannot create output directory {output_root}: {exc}"
            ) from exc
    if save_media:
        annotated_path = output_root / f"{source_path.stem}_annotated{source_path.suffix.lower()}"
        try:
            written = cv2.imwrite(str(annotated_path), annotated)
        except cv2.error as exc:
            raise OutputMediaError(
                f"OpenCV could not write annotated image {annotated_path}: {exc}"
            ) from exc
        if not written:
            raise OutputMediaError(f"OpenCV could not write annotated image: {annotated_path}")
    if save_detections:
        detections_path = output_root / f"{source_path.stem}_detections.json"

    output = ImageInferenceOutput(
        result=result,
        annotated_frame=annotated,
        annotated_media=annotated_path,
        detections_file=detections_path,
        checkpoint_sha256=engine.checkpoint.sha256,
        device=engine.device.value,
    )
    if detections_path is not None:
        _write_json(detections_path, output.to_dict())
    return output
