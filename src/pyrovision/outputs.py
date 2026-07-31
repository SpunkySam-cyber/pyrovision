"""Video, JSONL, and atomic-summary output sinks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .errors import OutputMediaError


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
        if len(codec) != 4:
            raise OutputMediaError("Video codec must contain exactly four characters")
        if fps <= 0 or width <= 0 or height <= 0:
            raise OutputMediaError("Video writer requires positive FPS and dimensions")
        factory = writer_factory or cv2.VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = factory(str(path), fourcc, float(fps), (width, height))
        if not self._writer.isOpened():
            self._writer.release()
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
            self._handle.write(json.dumps(value, separators=(",", ":")) + "\n")
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


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise OutputMediaError(f"Cannot write JSON output {path}: {exc}") from exc
