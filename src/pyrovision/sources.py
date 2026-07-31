"""Validated image/video classification and ordered OpenCV video reading."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import cv2
import numpy as np

from .errors import InputMediaError


SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
SUPPORTED_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
MediaKind = Literal["image", "video"]


def classify_media_path(path: Path) -> MediaKind:
    """Validate a local path and classify it without decoding model input."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise InputMediaError(f"Input media does not exist: {resolved}")
    extension = resolved.suffix.lower()
    if extension in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if extension in SUPPORTED_VIDEO_EXTENSIONS:
        return "video"
    raise InputMediaError(f"Unsupported media type: {extension or '<none>'}")


@dataclass(frozen=True)
class VideoMetadata:
    source: Path
    width: int
    height: int
    fps: float
    declared_frame_count: int


@dataclass(frozen=True)
class SourceFrame:
    frame_index: int
    timestamp_ms: float
    image: np.ndarray


class VideoReader:
    """Sequential video reader preserving source indices and timestamps."""

    def __init__(self, source: Path, capture_factory: Any | None = None) -> None:
        source_path = source.resolve()
        if classify_media_path(source_path) != "video":
            raise InputMediaError(f"Input is not a supported video: {source_path}")
        factory = capture_factory or cv2.VideoCapture
        self._capture = factory(str(source_path))
        if not self._capture.isOpened():
            self._capture.release()
            raise InputMediaError(f"OpenCV could not open video: {source_path}")
        width = int(round(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if width <= 0 or height <= 0:
            self._capture.release()
            raise InputMediaError(f"Video reports invalid dimensions: {width}x{height}")
        if not math.isfinite(fps) or fps <= 0:
            self._capture.release()
            raise InputMediaError(f"Video reports invalid FPS: {fps}")
        self.metadata = VideoMetadata(
            source=source_path,
            width=width,
            height=height,
            fps=fps,
            declared_frame_count=max(frame_count, 0),
        )
        self.frames_read = 0
        self._next_index = 0
        self._last_timestamp_ms = -1.0
        self._closed = False

    def _timestamp(self, frame_index: int) -> float:
        """Use container time when monotonic, otherwise reconstruct from FPS."""
        reported = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
        fallback = frame_index * 1000.0 / self.metadata.fps
        if not math.isfinite(reported) or reported < 0:
            return fallback
        if frame_index > 0 and reported <= self._last_timestamp_ms:
            return fallback
        return reported

    def read(self) -> SourceFrame | None:
        if self._closed:
            return None
        success, image = self._capture.read()
        if not success:
            return None
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise InputMediaError(
                f"Video returned an invalid frame at index {self._next_index}"
            )
        frame_index = self._next_index
        timestamp_ms = self._timestamp(frame_index)
        self._last_timestamp_ms = timestamp_ms
        self._next_index += 1
        self.frames_read += 1
        return SourceFrame(frame_index, timestamp_ms, image)

    def frames(self, frame_skip: int = 0) -> Iterator[SourceFrame]:
        if frame_skip < 0:
            raise ValueError("frame_skip cannot be negative")
        stride = frame_skip + 1
        while True:
            frame = self.read()
            if frame is None:
                break
            if frame.frame_index % stride == 0:
                yield frame

    def close(self) -> None:
        if not self._closed:
            self._capture.release()
            self._closed = True

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
