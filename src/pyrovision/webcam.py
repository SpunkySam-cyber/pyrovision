"""Real-time webcam inference with interruption-safe local outputs."""

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
from .outputs import (
    AnnotatedVideoWriter,
    JsonlWriter,
    LiveDisplay,
    write_json_atomic,
)
from .sources import WebcamReader


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id(index: int, started_at: str) -> str:
    compact_time = started_at.replace("-", "").replace(":", "")
    compact_time = compact_time.replace("+0000", "Z").replace("+00:00", "Z")
    return f"webcam_{index}_{compact_time.replace('.', '_')}"


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
    if frame_skip < 0:
        raise ValueError("frame_skip cannot be negative")
    if max_frames is not None and (
        isinstance(max_frames, bool)
        or not isinstance(max_frames, int)
        or max_frames <= 0
    ):
        raise ValueError("max_frames must be a positive integer")

    output_root = output_directory.resolve()
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputMediaError(
            f"Cannot create output directory {output_root}: {exc}"
        ) from exc

    started_at = _utc_now()
    stem = run_name or _run_id(webcam_index, started_at)
    annotated_path = (
        output_root / f"{stem}_annotated{video_extension}" if record else None
    )
    detections_path = (
        output_root / f"{stem}_detections.jsonl" if save_detections else None
    )
    summary_path = output_root / f"{stem}_summary.json"
    status = "complete"
    termination_reason = "unknown"
    failure: Exception | None = None
    frames_processed = 0
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    detection_counts: Counter[str] = Counter()
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

            try:
                for source_frame in reader.frames(frame_skip=frame_skip):
                    if stop_requested is not None and stop_requested():
                        status = "interrupted"
                        termination_reason = "stop_requested"
                        break
                    result = engine.predict_frame(
                        source_frame.image,
                        source=f"webcam:{webcam_index}",
                        frame_index=source_frame.frame_index,
                        timestamp_ms=source_frame.timestamp_ms,
                    )
                    annotated = None
                    if video_writer is not None or live_display is not None:
                        annotated = annotate_frame(
                            source_frame.image, result, style=annotation_style
                        )
                    if video_writer is not None:
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
        finally:
            if live_display is not None:
                live_display.close()
            if jsonl_writer is not None:
                jsonl_writer.close()
            if video_writer is not None:
                video_writer.close()

        frames_written = video_writer.frames_written if video_writer is not None else 0
        summary = WebcamRunSummary(
            status=status,
            termination_reason=termination_reason,
            error=str(failure) if failure is not None else None,
            webcam_index=metadata.index,
            annotated_media=str(annotated_path) if annotated_path is not None else None,
            detections_file=str(detections_path) if detections_path is not None else None,
            summary_file=str(summary_path),
            checkpoint_sha256=engine.checkpoint.sha256,
            device=engine.device.value,
            display_enabled=display,
            recording_enabled=record,
            codec=codec if record else None,
            width=metadata.width,
            height=metadata.height,
            capture_fps=metadata.fps,
            capture_fps_source=metadata.fps_source,
            recording_fps=recording_fps,
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
        return WebcamInferenceOutput(summary=summary)
