"""Shared ordered-frame processing and failure-safe resource cleanup."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .annotation import AnnotationStyle, annotate_frame
from .errors import InputMediaError, OutputMediaError
from .model import DetectorEngine
from .outputs import AnnotatedVideoWriter, JsonlWriter
from .sources import SourceFrame


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamProcessingResult:
    status: str
    termination_reason: str
    failure: Exception | None
    frames_processed: int
    first_timestamp_ms: float | None
    last_timestamp_ms: float | None
    detections_per_class: dict[str, int]

    @property
    def detections_total(self) -> int:
        return sum(self.detections_per_class.values())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def failed_processing_result(
    engine: DetectorEngine,
    failure: Exception,
) -> StreamProcessingResult:
    return StreamProcessingResult(
        status="failed",
        termination_reason="error",
        failure=failure,
        frames_processed=0,
        first_timestamp_ms=None,
        last_timestamp_ms=None,
        detections_per_class={name: 0 for name in engine.class_names},
    )


def process_frames(
    engine: DetectorEngine,
    frames: Iterable[SourceFrame],
    *,
    source: str,
    video_writer: AnnotatedVideoWriter | None = None,
    jsonl_writer: JsonlWriter | None = None,
    live_display: Any | None = None,
    annotation_style: AnnotationStyle | None = None,
    stop_requested: Callable[[], bool] | None = None,
    max_frames: int | None = None,
    require_frames: bool = False,
) -> StreamProcessingResult:
    """Process an ordered frame stream without owning its resources."""
    frames_processed = 0
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    detection_counts: Counter[str] = Counter()
    status = "complete"
    termination_reason = "source_exhausted"
    failure: Exception | None = None

    try:
        for source_frame in frames:
            if stop_requested is not None and stop_requested():
                status = "interrupted"
                termination_reason = "stop_requested"
                break
            processed_index = frames_processed
            result = engine.predict_frame(
                source_frame.image,
                source=source,
                frame_index=source_frame.frame_index,
                timestamp_ms=source_frame.timestamp_ms,
            )
            frames_processed += 1
            if first_timestamp is None:
                first_timestamp = result.timestamp_ms
            last_timestamp = result.timestamp_ms
            detection_counts.update(
                detection.class_name for detection in result.detections
            )

            annotated = None
            if video_writer is not None or live_display is not None:
                annotated = annotate_frame(
                    source_frame.image,
                    result,
                    style=annotation_style,
                )
            if video_writer is not None:
                video_writer.write(annotated)
            if jsonl_writer is not None:
                jsonl_writer.write(
                    {
                        "record_type": "frame",
                        "processed_index": processed_index,
                        **result.to_dict(),
                    }
                )
            if live_display is not None and live_display.show(annotated):
                termination_reason = "display_quit"
                break
            if max_frames is not None and frames_processed >= max_frames:
                termination_reason = "max_frames"
                break
    except KeyboardInterrupt:
        status = "interrupted"
        termination_reason = "keyboard_interrupt"
    except Exception as exc:
        status = "failed"
        termination_reason = "error"
        failure = exc

    if (
        require_frames
        and frames_processed == 0
        and failure is None
        and termination_reason == "source_exhausted"
    ):
        status = "failed"
        termination_reason = "error"
        failure = InputMediaError("Input video contains no decodable frames")

    return StreamProcessingResult(
        status=status,
        termination_reason=termination_reason,
        failure=failure,
        frames_processed=frames_processed,
        first_timestamp_ms=first_timestamp,
        last_timestamp_ms=last_timestamp,
        detections_per_class={
            name: detection_counts.get(name, 0) for name in engine.class_names
        },
    )


def close_resources(*resources: Any | None) -> OutputMediaError | None:
    """Attempt every close operation and report all cleanup failures together."""
    failures: list[str] = []
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.close()
        except Exception as exc:
            failures.append(f"{type(resource).__name__}: {exc}")
            LOGGER.exception("Failed to close %s", type(resource).__name__)
    if failures:
        return OutputMediaError("Resource cleanup failed: " + "; ".join(failures))
    return None


def discard_empty_media(
    path: Path | None,
    frames_written: int,
) -> OutputMediaError | None:
    """Remove a generated zero-frame recording that cannot be relied on to decode."""
    if path is None or frames_written > 0:
        return None
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        return OutputMediaError(f"Cannot remove empty video output {path}: {exc}")
    return None


def combine_output_failures(
    first: OutputMediaError | None,
    second: OutputMediaError | None,
) -> OutputMediaError | None:
    if first is None:
        return second
    if second is None:
        return first
    return OutputMediaError(f"{first}; additionally, {second}")


def failure_message(
    primary: Exception | None,
    cleanup: Exception | None,
) -> str | None:
    if primary is None and cleanup is None:
        return None
    if primary is None:
        return str(cleanup)
    if cleanup is None:
        return str(primary)
    return f"{primary}; additionally, {cleanup}"
