"""Validated media sources for local image, video, and webcam inference."""

from __future__ import annotations

import math
import time
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
MediaKind = Literal["image", "image_directory", "video"]


def classify_media_path(path: Path) -> MediaKind:
    """Validate a local path and classify it without decoding model input."""
    resolved = path.resolve()
    if resolved.is_dir():
        return "image_directory"
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
        try:
            self._capture = factory(str(source_path))
            opened = bool(self._capture.isOpened())
        except Exception as exc:
            raise InputMediaError(
                f"OpenCV could not initialize video capture for {source_path}: {exc}"
            ) from exc
        if not opened:
            try:
                self._capture.release()
            except Exception:
                pass
            raise InputMediaError(f"OpenCV could not open video: {source_path}")
        try:
            width = int(round(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = float(self._capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(round(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        except Exception as exc:
            self._capture.release()
            raise InputMediaError(
                f"Cannot read video metadata from {source_path}: {exc}"
            ) from exc
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
        try:
            reported = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
        except Exception:
            reported = float("nan")
        fallback = frame_index * 1000.0 / self.metadata.fps
        candidate = reported
        if not math.isfinite(candidate) or candidate < 0:
            candidate = fallback
        if frame_index > 0 and candidate <= self._last_timestamp_ms:
            candidate = max(
                fallback,
                self._last_timestamp_ms + 1000.0 / self.metadata.fps,
            )
        return candidate

    def read(self) -> SourceFrame | None:
        if self._closed:
            return None
        try:
            success, image = self._capture.read()
        except Exception as exc:
            raise InputMediaError(
                f"Video decoding failed at frame {self._next_index}: {exc}"
            ) from exc
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
        if (
            isinstance(frame_skip, bool)
            or not isinstance(frame_skip, int)
            or frame_skip < 0
        ):
            raise ValueError("frame_skip must be a non-negative integer")
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


@dataclass(frozen=True)
class WebcamMetadata:
    index: int
    width: int
    height: int
    fps: float
    fps_source: Literal["camera", "fallback"]


class WebcamReader:
    """Validated webcam reader with monotonic session-relative timestamps."""

    def __init__(
        self,
        index: int,
        *,
        capture_factory: Any | None = None,
        clock: Any | None = None,
        fallback_fps: float = 30.0,
        max_read_attempts: int = 3,
    ) -> None:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise InputMediaError("Webcam index must be a non-negative integer")
        if not math.isfinite(fallback_fps) or fallback_fps <= 0:
            raise ValueError("fallback_fps must be positive and finite")
        if (
            isinstance(max_read_attempts, bool)
            or not isinstance(max_read_attempts, int)
            or max_read_attempts <= 0
        ):
            raise ValueError("max_read_attempts must be a positive integer")

        factory = capture_factory or cv2.VideoCapture
        try:
            self._capture = factory(index)
            opened = bool(self._capture.isOpened())
        except Exception as exc:
            raise InputMediaError(
                f"OpenCV could not initialize webcam index {index}: {exc}"
            ) from exc
        if not opened:
            try:
                self._capture.release()
            except Exception:
                pass
            raise InputMediaError(
                f"OpenCV could not open webcam index {index}; verify the index, "
                "camera permissions, and that no other application is using it"
            )

        self._clock = clock or time.perf_counter
        self._started_at = float(self._clock())
        self._max_read_attempts = max_read_attempts
        self._closed = False
        self.frames_read = 0
        self._next_index = 0
        self._pending_frame = self._read_initial_frame(index)
        self._last_timestamp_ms = -1.0
        height, width = self._pending_frame.shape[:2]

        try:
            reported_fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        except Exception:
            reported_fps = 0.0
        if math.isfinite(reported_fps) and reported_fps > 0:
            fps = reported_fps
            fps_source: Literal["camera", "fallback"] = "camera"
        else:
            fps = float(fallback_fps)
            fps_source = "fallback"
        self.metadata = WebcamMetadata(index, width, height, fps, fps_source)

    def _read_initial_frame(self, index: int) -> np.ndarray:
        last_error: Exception | None = None
        for _ in range(self._max_read_attempts):
            try:
                success, image = self._capture.read()
            except Exception as exc:
                last_error = exc
                continue
            if success and isinstance(image, np.ndarray) and image.size > 0:
                return image
        self._capture.release()
        self._closed = True
        detail = f": {last_error}" if last_error is not None else ""
        raise InputMediaError(
            f"Webcam index {index} opened but did not return a valid frame{detail}"
        )

    def _capture_frame(self) -> np.ndarray:
        last_error: Exception | None = None
        for _ in range(self._max_read_attempts):
            try:
                success, image = self._capture.read()
            except Exception as exc:
                last_error = exc
                continue
            if success and isinstance(image, np.ndarray) and image.size > 0:
                return image
        detail = f": {last_error}" if last_error is not None else ""
        raise InputMediaError(
            f"Webcam index {self.metadata.index} failed to return a valid frame "
            f"after {self._max_read_attempts} attempts{detail}"
        )

    def read(self) -> SourceFrame | None:
        if self._closed:
            return None
        if self._pending_frame is not None:
            image = self._pending_frame
            self._pending_frame = None
        else:
            image = self._capture_frame()
        height, width = image.shape[:2]
        if width != self.metadata.width or height != self.metadata.height:
            raise InputMediaError(
                "Webcam frame dimensions changed during capture: "
                f"{width}x{height}, expected "
                f"{self.metadata.width}x{self.metadata.height}"
            )
        frame_index = self._next_index
        elapsed_ms = (float(self._clock()) - self._started_at) * 1000.0
        if not math.isfinite(elapsed_ms):
            raise InputMediaError("Webcam clock returned a non-finite timestamp")
        elapsed_ms = max(0.0, elapsed_ms)
        if elapsed_ms <= self._last_timestamp_ms:
            elapsed_ms = self._last_timestamp_ms + 0.001
        self._last_timestamp_ms = elapsed_ms
        self._next_index += 1
        self.frames_read += 1
        return SourceFrame(frame_index, elapsed_ms, image)

    def frames(self, frame_skip: int = 0) -> Iterator[SourceFrame]:
        if (
            isinstance(frame_skip, bool)
            or not isinstance(frame_skip, int)
            or frame_skip < 0
        ):
            raise ValueError("frame_skip must be a non-negative integer")
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

    def __enter__(self) -> "WebcamReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
