"""Still-image input, annotation, and structured output orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .annotation import AnnotationStyle, annotate_frame
from .errors import InputMediaError, OutputMediaError
from .model import DetectorEngine
from .outputs import allocate_output_stem, write_image_atomic, write_json_atomic
from .sources import SUPPORTED_IMAGE_EXTENSIONS
from .types import FrameResult


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


@dataclass(frozen=True)
class ImageDirectoryInferenceOutput:
    """Deterministically ordered results from one flat image directory."""

    source_directory: Path
    images: tuple[ImageInferenceOutput, ...]

    def __init__(
        self,
        source_directory: Path,
        images: Iterable[ImageInferenceOutput],
    ) -> None:
        object.__setattr__(self, "source_directory", source_directory)
        object.__setattr__(self, "images", tuple(images))

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": True,
            "source_directory": str(self.source_directory),
            "images_processed": len(self.images),
            "images": [image.to_dict() for image in self.images],
        }


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
    requested_suffixes: list[str] = []
    if save_media:
        requested_suffixes.append(f"_annotated{source_path.suffix.lower()}")
    if save_detections:
        requested_suffixes.append("_detections.json")
    output_stem = (
        allocate_output_stem(
            output_root,
            source_path.stem,
            tuple(requested_suffixes),
        )
        if requested_suffixes
        else source_path.stem
    )
    if save_media:
        annotated_path = (
            output_root
            / f"{output_stem}_annotated{source_path.suffix.lower()}"
        )
        write_image_atomic(annotated_path, annotated)
    if save_detections:
        detections_path = output_root / f"{output_stem}_detections.json"

    output = ImageInferenceOutput(
        result=result,
        annotated_frame=annotated,
        annotated_media=annotated_path,
        detections_file=detections_path,
        checkpoint_sha256=engine.checkpoint.sha256,
        device=engine.device.value,
    )
    if detections_path is not None:
        write_json_atomic(detections_path, output.to_dict())
    return output


def infer_image_directory(
    engine: DetectorEngine,
    source_directory: Path,
    *,
    output_directory: Path,
    save_media: bool = True,
    save_detections: bool = True,
    annotation_style: AnnotationStyle | None = None,
) -> ImageDirectoryInferenceOutput:
    """Infer supported top-level images in deterministic filename order."""
    directory = source_directory.resolve()
    if not directory.is_dir():
        raise InputMediaError(f"Input image directory does not exist: {directory}")
    try:
        images = sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    except OSError as exc:
        raise InputMediaError(f"Cannot read image directory {directory}: {exc}") from exc
    if not images:
        raise InputMediaError(
            f"Image directory contains no supported top-level images: {directory}"
        )
    outputs = [
        infer_image(
            engine,
            image,
            output_directory=output_directory,
            save_media=save_media,
            save_detections=save_detections,
            annotation_style=annotation_style,
        )
        for image in images
    ]
    return ImageDirectoryInferenceOutput(directory, outputs)
