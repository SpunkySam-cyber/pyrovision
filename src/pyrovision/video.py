"""Ordered local-video inference pipeline with interruption-safe outputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .annotation import AnnotationStyle
from .errors import OutputMediaError
from .model import DetectorEngine
from .outputs import (
    AnnotatedVideoWriter,
    JsonlWriter,
    allocate_output_stem,
    validate_video_extension,
    write_json_atomic,
)
from .sources import VideoReader
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
class VideoRunSummary:
    status: str
    interruption_reason: str | None
    error: str | None
    source: str
    annotated_media: str | None
    detections_file: str | None
    summary_file: str
    checkpoint_sha256: str
    device: str
    codec: str | None
    source_width: int
    source_height: int
    source_fps: float
    output_fps: float | None
    declared_source_frames: int
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
            "interruption_reason": self.interruption_reason,
            "error": self.error,
            "annotated_media": self.annotated_media,
            "detections_file": self.detections_file,
            "summary_file": self.summary_file,
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": self.device,
            "codec": self.codec,
            "source": {
                "path": self.source,
                "width": self.source_width,
                "height": self.source_height,
                "fps": self.source_fps,
                "declared_frames": self.declared_source_frames,
            },
            "output_fps": self.output_fps,
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
class VideoInferenceOutput:
    summary: VideoRunSummary

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.summary.status != "failed", **self.summary.to_dict()}


def infer_video(
    engine: DetectorEngine,
    source: Path,
    *,
    output_directory: Path,
    frame_skip: int = 0,
    save_media: bool = True,
    save_detections: bool = True,
    codec: str = "mp4v",
    video_extension: str = ".mp4",
    annotation_style: AnnotationStyle | None = None,
    stop_requested: Callable[[], bool] | None = None,
    capture_factory: Any | None = None,
    writer_factory: Any | None = None,
) -> VideoInferenceOutput:
    """Process a video sequentially and always close valid partial outputs."""
    if (
        isinstance(frame_skip, bool)
        or not isinstance(frame_skip, int)
        or frame_skip < 0
    ):
        raise ValueError("frame_skip must be a non-negative integer")
    if save_media:
        video_extension = validate_video_extension(video_extension)
    output_root = output_directory.resolve()
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputMediaError(f"Cannot create output directory {output_root}: {exc}") from exc

    source_path = source.resolve()
    requested_suffixes = ["_summary.json"]
    if save_media:
        requested_suffixes.append(f"_annotated{video_extension}")
    if save_detections:
        requested_suffixes.append("_detections.jsonl")
    stem = allocate_output_stem(
        output_root,
        source_path.stem,
        tuple(requested_suffixes),
    )
    annotated_path = (
        output_root / f"{stem}_annotated{video_extension}" if save_media else None
    )
    detections_path = (
        output_root / f"{stem}_detections.jsonl" if save_detections else None
    )
    summary_path = output_root / f"{stem}_summary.json"
    started_at = utc_now()
    video_writer: AnnotatedVideoWriter | None = None
    jsonl_writer: JsonlWriter | None = None

    with VideoReader(source_path, capture_factory=capture_factory) as reader:
        metadata = reader.metadata
        output_fps = metadata.fps / (frame_skip + 1) if save_media else None
        try:
            try:
                if annotated_path is not None:
                    video_writer = AnnotatedVideoWriter(
                        annotated_path,
                        codec=codec,
                        fps=float(output_fps),
                        width=metadata.width,
                        height=metadata.height,
                        writer_factory=writer_factory,
                    )
                if detections_path is not None:
                    jsonl_writer = JsonlWriter(detections_path)
                processing = process_frames(
                    engine,
                    reader.frames(frame_skip=frame_skip),
                    source=str(source_path),
                    video_writer=video_writer,
                    jsonl_writer=jsonl_writer,
                    annotation_style=annotation_style,
                    stop_requested=stop_requested,
                    require_frames=True,
                )
            except Exception as exc:
                processing = failed_processing_result(engine, exc)
        finally:
            cleanup_failure = close_resources(
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
        summary = VideoRunSummary(
            status="failed" if cleanup_failure is not None else processing.status,
            interruption_reason=(
                processing.termination_reason
                if processing.status == "interrupted"
                else None
            ),
            error=failure_message(primary_failure, cleanup_failure),
            source=str(source_path),
            annotated_media=(
                str(annotated_path) if published_video else None
            ),
            detections_file=(
                str(detections_path) if jsonl_writer is not None else None
            ),
            summary_file=str(summary_path),
            checkpoint_sha256=engine.checkpoint.sha256,
            device=engine.device.value,
            codec=codec if published_video else None,
            source_width=metadata.width,
            source_height=metadata.height,
            source_fps=metadata.fps,
            output_fps=output_fps,
            declared_source_frames=metadata.declared_frame_count,
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
        return VideoInferenceOutput(summary=summary)
