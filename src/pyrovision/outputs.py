"""Video, JSONL, and atomic-summary output sinks."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .errors import OutputMediaError


_INVALID_STEM_CHARACTERS = frozenset('<>:"/\\|?*')
SUPPORTED_OUTPUT_VIDEO_EXTENSIONS = frozenset({".avi", ".mkv", ".mov", ".mp4"})


def validate_video_extension(extension: str) -> str:
    if not isinstance(extension, str):
        raise OutputMediaError("Video extension must be a string")
    normalized = extension.lower()
    if normalized not in SUPPORTED_OUTPUT_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_VIDEO_EXTENSIONS))
        raise OutputMediaError(
            f"Unsupported output video extension {extension!r}; use {supported}"
        )
    return normalized


def validate_output_stem(stem: str) -> str:
    """Reject empty, path-like, control-character, and Windows-invalid stems."""
    if not isinstance(stem, str) or not stem:
        raise OutputMediaError("Output stem must be a non-empty string")
    if stem in {".", ".."} or stem != Path(stem).name:
        raise OutputMediaError(f"Output stem must not contain a path: {stem!r}")
    if stem[-1] in {" ", "."}:
        raise OutputMediaError("Output stem cannot end with a space or period")
    if any(ord(character) < 32 for character in stem):
        raise OutputMediaError("Output stem cannot contain control characters")
    invalid = sorted(set(stem) & _INVALID_STEM_CHARACTERS)
    if invalid:
        raise OutputMediaError(
            f"Output stem contains invalid character(s): {''.join(invalid)}"
        )
    return stem


def allocate_output_stem(
    directory: Path,
    preferred_stem: str,
    suffixes: tuple[str, ...],
) -> str:
    """Select the first deterministic stem whose requested outputs do not exist."""
    stem = validate_output_stem(preferred_stem)
    if not suffixes or any(not suffix for suffix in suffixes):
        raise OutputMediaError("At least one non-empty output suffix is required")
    for index in range(1, 100_001):
        candidate = stem if index == 1 else f"{stem}_{index}"
        if not any((directory / f"{candidate}{suffix}").exists() for suffix in suffixes):
            return candidate
    raise OutputMediaError(
        f"Could not allocate a collision-free output name for {stem!r}"
    )


class AnnotatedVideoWriter:
    """Validated OpenCV writer that tracks successfully submitted frames."""

    def __init__(
        self,
        path: Path,
        *,
        codec: str,
        fps: float,
        width: int,
        height: int,
        writer_factory: Any | None = None,
    ) -> None:
        if not isinstance(codec, str) or len(codec) != 4 or not codec.isascii():
            raise OutputMediaError("Video codec must contain exactly four characters")
        if any(ord(character) < 32 or ord(character) > 126 for character in codec):
            raise OutputMediaError("Video codec must contain printable ASCII characters")
        if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
            raise OutputMediaError("Video writer requires a positive finite FPS")
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
        ):
            raise OutputMediaError("Video writer requires positive FPS and dimensions")
        factory = writer_factory or cv2.VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*codec)
        try:
            self._writer = factory(str(path), fourcc, float(fps), (width, height))
        except Exception as exc:
            raise OutputMediaError(
                f"Cannot create video writer for {path} with codec {codec}: {exc}"
            ) from exc
        try:
            opened = bool(self._writer.isOpened())
        except Exception as exc:
            try:
                self._writer.release()
            except Exception:
                pass
            raise OutputMediaError(
                f"Cannot query video writer for {path} with codec {codec}: {exc}"
            ) from exc
        if not opened:
            try:
                self._writer.release()
            except Exception:
                pass
            raise OutputMediaError(
                f"Cannot open video writer for {path} with codec {codec}; "
                "try mp4v/.mp4 or MJPG/.avi"
            )
        self.path = path
        self.width = width
        self.height = height
        self.frames_written = 0
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise OutputMediaError("Cannot write to a closed video writer")
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.dtype != np.uint8
        ):
            raise OutputMediaError(
                "Video output frames must be uint8 BGR arrays with three channels"
            )
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise OutputMediaError(
                f"Output frame has {frame.shape[1]}x{frame.shape[0]}, expected "
                f"{self.width}x{self.height}"
            )
        try:
            self._writer.write(frame)
        except cv2.error as exc:
            raise OutputMediaError(f"Failed to encode video frame: {exc}") from exc
        self.frames_written += 1

    def close(self) -> None:
        if not self._closed:
            self._writer.release()
            self._closed = True


class JsonlWriter:
    """Flush every record so interrupted runs retain valid complete lines."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self._handle = path.open("w", encoding="utf-8", newline="\n")
        except OSError as exc:
            raise OutputMediaError(f"Cannot open JSONL output {path}: {exc}") from exc
        self.records_written = 0
        self._closed = False

    def write(self, value: dict[str, Any]) -> None:
        if self._closed:
            raise OutputMediaError("Cannot write to a closed JSONL writer")
        try:
            record = json.dumps(
                value,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise OutputMediaError(
                f"Cannot serialize JSONL output {self.path}: {exc}"
            ) from exc
        try:
            self._handle.write(record + "\n")
            self._handle.flush()
        except OSError as exc:
            raise OutputMediaError(f"Cannot write JSONL output {self.path}: {exc}") from exc
        self.records_written += 1

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True


class LiveDisplay:
    """Small OpenCV display adapter with an explicit stop-key contract."""

    def __init__(
        self,
        window_name: str = "PyroVision Webcam",
        *,
        wait_key_delay_ms: int = 1,
    ) -> None:
        if wait_key_delay_ms < 1:
            raise ValueError("wait_key_delay_ms must be positive")
        self.window_name = window_name
        self.wait_key_delay_ms = wait_key_delay_ms
        self._shown = False
        self._closed = False

    def show(self, frame: np.ndarray) -> bool:
        """Show one frame and return true when Q or Escape requests shutdown."""
        if self._closed:
            raise OutputMediaError("Cannot show a frame on a closed display")
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise OutputMediaError("Live display requires a non-empty NumPy frame")
        try:
            cv2.imshow(self.window_name, frame)
            self._shown = True
            key = cv2.waitKey(self.wait_key_delay_ms) & 0xFF
        except cv2.error as exc:
            raise OutputMediaError(
                "OpenCV could not create the live display; use --no-display "
                "in headless environments"
            ) from exc
        return key in (ord("q"), ord("Q"), 27)

    def close(self) -> None:
        if self._closed:
            return
        if self._shown:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
        self._closed = True


def write_image_atomic(path: Path, image: np.ndarray) -> None:
    """Encode an image beside its destination and atomically publish it."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=path.suffix,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        written = cv2.imwrite(str(temporary_path), image)
        if not written:
            raise OutputMediaError(f"OpenCV could not encode annotated image: {path}")
        os.replace(temporary_path, path)
    except cv2.error as exc:
        raise OutputMediaError(
            f"OpenCV could not write annotated image {path}: {exc}"
        ) from exc
    except OSError as exc:
        raise OutputMediaError(f"Cannot write annotated image {path}: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Serialize strict deterministic JSON and atomically publish it."""
    temporary_path: Path | None = None
    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise OutputMediaError(f"Cannot serialize JSON output {path}: {exc}") from exc
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        raise OutputMediaError(f"Cannot write JSON output {path}: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
