"""Real-time webcam inference with interruption-safe local outputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .annotation import AnnotationStyle
from .errors import OutputMediaError
from .model import DetectorEngine
from .outputs import (
    AnnotatedVideoWriter,
    JsonlWriter,
    LiveDisplay,
    allocate_output_stem,
    validate_video_extension,
    write_json_atomic,
)
from .sources import WebcamReader
from .streaming import (
    close_resources,
    combine_output_failures,
    discard_empty_media,
    failed_processing_result,
    failure_message,
    process_frames,
    utc_now,
)


@dataclass(frozen=True)
class WebcamRunSummary:
    status: str
    termination_reason: str
    error: str | None
    webcam_index: int
    annotated_media: str | None
    detections_file: str | None
    summary_file: str
    checkpoint_sha256: str
    device: str
    display_enabled: bool
    recording_enabled: bool
    codec: str | None
    width: int
    height: int
    capture_fps: float
    capture_fps_source: str
    recording_fps: float | None
    frames_read: int
    frames_processed: int
    frames_written: int
    frame_skip: int
    first_timestamp_ms: float | None
    last_timestamp_ms: float | None
    detections_total: int
    detections_per_class: dict[str, int]
    started_at_utc: str
    completed_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "termination_reason": self.termination_reason,
            "error": self.error,
            "annotated_media": self.annotated_media,
            "detections_file": self.detections_file,
            "summary_file": self.summary_file,
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": self.device,
            "display_enabled": self.display_enabled,
            "recording_enabled": self.recording_enabled,
            "codec": self.codec,
            "source": {
                "kind": "webcam",
                "index": self.webcam_index,
                "width": self.width,
                "height": self.height,
                "fps": self.capture_fps,
                "fps_source": self.capture_fps_source,
            },
            "recording_fps": self.recording_fps,
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "frames_written": self.frames_written,
            "frame_skip": self.frame_skip,
            "first_timestamp_ms": self.first_timestamp_ms,
            "last_timestamp_ms": self.last_timestamp_ms,
            "detections_total": self.detections_total,
            "detections_per_class": self.detections_per_class,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
        }


@dataclass(frozen=True)
class WebcamInferenceOutput:
    summary: WebcamRunSummary

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.summary.status != "failed", **self.summary.to_dict()}


def _run_id(index: int, started_at: str) -> str:
    timestamp = datetime.fromisoformat(started_at).astimezone(timezone.utc)
    return f"webcam_{index}_{timestamp.strftime('%Y%m%dT%H%M%S_%fZ')}"


def infer_webcam(
    engine: DetectorEngine,
    webcam_index: int,
    *,
    output_directory: Path,
    frame_skip: int = 0,
    record: bool = True,
    save_detections: bool = True,
    display: bool = False,
    codec: str = "mp4v",
    video_extension: str = ".mp4",
    max_frames: int | None = None,
    annotation_style: AnnotationStyle | None = None,
    stop_requested: Callable[[], bool] | None = None,
    capture_factory: Any | None = None,
    writer_factory: Any | None = None,
    display_factory: Any | None = None,
    clock: Any | None = None,
    run_name: str | None = None,
) -> WebcamInferenceOutput:
    """Run webcam inference until a stop condition and close every resource."""
    if (
        isinstance(frame_skip, bool)
        or not isinstance(frame_skip, int)
        or frame_skip < 0
    ):
        raise ValueError("frame_skip must be a non-negative integer")
    if max_frames is not None and (
        isinstance(max_frames, bool)
        or not isinstance(max_frames, int)
        or max_frames <= 0
    ):
        raise ValueError("max_frames must be a positive integer")
    if record:
        video_extension = validate_video_extension(video_extension)

    output_root = output_directory.resolve()
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputMediaError(
            f"Cannot create output directory {output_root}: {exc}"
        ) from exc

    started_at = utc_now()
    requested_suffixes = ["_summary.json"]
    if record:
        requested_suffixes.append(f"_annotated{video_extension}")
    if save_detections:
        requested_suffixes.append("_detections.jsonl")
    stem = allocate_output_stem(
        output_root,
        run_name or _run_id(webcam_index, started_at),
        tuple(requested_suffixes),
    )
    annotated_path = (
        output_root / f"{stem}_annotated{video_extension}" if record else None
    )
    detections_path = (
        output_root / f"{stem}_detections.jsonl" if save_detections else None
    )
    summary_path = output_root / f"{stem}_summary.json"
    video_writer: AnnotatedVideoWriter | None = None
    jsonl_writer: JsonlWriter | None = None
    live_display: Any | None = None

    with WebcamReader(
        webcam_index,
        capture_factory=capture_factory,
        clock=clock,
    ) as reader:
        metadata = reader.metadata
        recording_fps = metadata.fps / (frame_skip + 1) if record else None
        try:
            try:
                if annotated_path is not None:
                    video_writer = AnnotatedVideoWriter(
                        annotated_path,
                        codec=codec,
                        fps=float(recording_fps),
                        width=metadata.width,
                        height=metadata.height,
                        writer_factory=writer_factory,
                    )
                if detections_path is not None:
                    jsonl_writer = JsonlWriter(detections_path)
                if display:
                    factory = display_factory or LiveDisplay
                    live_display = factory(f"PyroVision Webcam {webcam_index}")
                processing = process_frames(
                    engine,
                    reader.frames(frame_skip=frame_skip),
                    source=f"webcam:{webcam_index}",
                    video_writer=video_writer,
                    jsonl_writer=jsonl_writer,
                    live_display=live_display,
                    annotation_style=annotation_style,
                    stop_requested=stop_requested,
                    max_frames=max_frames,
                )
            except Exception as exc:
                processing = failed_processing_result(engine, exc)
        finally:
            cleanup_failure = close_resources(
                live_display,
                jsonl_writer,
                video_writer,
                reader,
            )

        frames_written = video_writer.frames_written if video_writer is not None else 0
        cleanup_failure = combine_output_failures(
            cleanup_failure,
            discard_empty_media(annotated_path, frames_written),
        )
        published_video = video_writer is not None and frames_written > 0
        primary_failure = processing.failure
        effective_failure = primary_failure or cleanup_failure
        summary = WebcamRunSummary(
            status="failed" if cleanup_failure is not None else processing.status,
            termination_reason=(
                "error"
                if cleanup_failure is not None
                else processing.termination_reason
            ),
            error=failure_message(primary_failure, cleanup_failure),
            webcam_index=metadata.index,
            annotated_media=(
                str(annotated_path) if published_video else None
            ),
            detections_file=(
                str(detections_path) if jsonl_writer is not None else None
            ),
            summary_file=str(summary_path),
            checkpoint_sha256=engine.checkpoint.sha256,
            device=engine.device.value,
            display_enabled=display,
            recording_enabled=published_video,
            codec=codec if published_video else None,
            width=metadata.width,
            height=metadata.height,
            capture_fps=metadata.fps,
            capture_fps_source=metadata.fps_source,
            recording_fps=recording_fps,
            frames_read=reader.frames_read,
            frames_processed=processing.frames_processed,
            frames_written=frames_written,
            frame_skip=frame_skip,
            first_timestamp_ms=processing.first_timestamp_ms,
            last_timestamp_ms=processing.last_timestamp_ms,
            detections_total=processing.detections_total,
            detections_per_class=processing.detections_per_class,
            started_at_utc=started_at,
            completed_at_utc=utc_now(),
        )
        try:
            write_json_atomic(summary_path, summary.to_dict())
        except Exception as summary_failure:
            if effective_failure is not None:
                raise OutputMediaError(
                    f"{effective_failure}; additionally, could not write run "
                    f"summary: {summary_failure}"
                ) from effective_failure
            raise
        if effective_failure is not None:
            if primary_failure is not None and cleanup_failure is not None:
                primary_failure.add_note(str(cleanup_failure))
            raise effective_failure
        return WebcamInferenceOutput(summary=summary)
