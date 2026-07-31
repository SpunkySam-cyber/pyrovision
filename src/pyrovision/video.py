"""Ordered local-video inference pipeline with interruption-safe outputs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .annotation import AnnotationStyle, annotate_frame
from .errors import OutputMediaError
from .model import DetectorEngine
from .outputs import AnnotatedVideoWriter, JsonlWriter, write_json_atomic
from .sources import VideoReader


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if frame_skip < 0:
        raise ValueError("frame_skip cannot be negative")
    output_root = output_directory.resolve()
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputMediaError(f"Cannot create output directory {output_root}: {exc}") from exc

    source_path = source.resolve()
    stem = source_path.stem
    annotated_path = (
        output_root / f"{stem}_annotated{video_extension}" if save_media else None
    )
    detections_path = (
        output_root / f"{stem}_detections.jsonl" if save_detections else None
    )
    summary_path = output_root / f"{stem}_summary.json"
    started_at = _utc_now()
    status = "complete"
    interruption_reason: str | None = None
    failure: Exception | None = None
    frames_processed = 0
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    detection_counts: Counter[str] = Counter()
    video_writer: AnnotatedVideoWriter | None = None
    jsonl_writer: JsonlWriter | None = None

    with VideoReader(source_path, capture_factory=capture_factory) as reader:
        metadata = reader.metadata
        output_fps = metadata.fps / (frame_skip + 1) if save_media else None
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

            try:
                for source_frame in reader.frames(frame_skip=frame_skip):
                    if stop_requested is not None and stop_requested():
                        status = "interrupted"
                        interruption_reason = "stop_requested"
                        break
                    result = engine.predict_frame(
                        source_frame.image,
                        source=str(source_path),
                        frame_index=source_frame.frame_index,
                        timestamp_ms=source_frame.timestamp_ms,
                    )
                    if video_writer is not None:
                        annotated = annotate_frame(
                            source_frame.image, result, style=annotation_style
                        )
                        video_writer.write(annotated)
                    if jsonl_writer is not None:
                        jsonl_writer.write(
                            {
                                "record_type": "frame",
                                "processed_index": frames_processed,
                                **result.to_dict(),
                            }
                        )
                    if first_timestamp is None:
                        first_timestamp = result.timestamp_ms
                    last_timestamp = result.timestamp_ms
                    frames_processed += 1
                    detection_counts.update(
                        detection.class_name for detection in result.detections
                    )
            except KeyboardInterrupt:
                status = "interrupted"
                interruption_reason = "keyboard_interrupt"
            except Exception as exc:
                status = "failed"
                failure = exc
        finally:
            if jsonl_writer is not None:
                jsonl_writer.close()
            if video_writer is not None:
                video_writer.close()

        frames_written = video_writer.frames_written if video_writer is not None else 0
        summary = VideoRunSummary(
            status=status,
            interruption_reason=interruption_reason,
            error=str(failure) if failure is not None else None,
            source=str(source_path),
            annotated_media=str(annotated_path) if annotated_path is not None else None,
            detections_file=str(detections_path) if detections_path is not None else None,
            summary_file=str(summary_path),
            checkpoint_sha256=engine.checkpoint.sha256,
            device=engine.device.value,
            codec=codec if save_media else None,
            source_width=metadata.width,
            source_height=metadata.height,
            source_fps=metadata.fps,
            output_fps=output_fps,
            declared_source_frames=metadata.declared_frame_count,
            frames_read=reader.frames_read,
            frames_processed=frames_processed,
            frames_written=frames_written,
            frame_skip=frame_skip,
            first_timestamp_ms=first_timestamp,
            last_timestamp_ms=last_timestamp,
            detections_total=sum(detection_counts.values()),
            detections_per_class={
                name: detection_counts.get(name, 0) for name in engine.class_names
            },
            started_at_utc=started_at,
            completed_at_utc=_utc_now(),
        )
        write_json_atomic(summary_path, summary.to_dict())
        if failure is not None:
            raise failure
        return VideoInferenceOutput(summary=summary)
