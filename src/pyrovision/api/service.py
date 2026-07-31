"""Application service translating HTTP uploads into existing inference calls."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from time import perf_counter
from urllib.parse import quote

from ..errors import OutputMediaError
from ..images import infer_image
from ..model import DetectorEngine
from ..video import infer_video
from .config import BackendConfig
from .schemas import (
    DetectionResponse,
    ImagePredictionResponse,
    OutputReference,
    ProcessingMetadata,
    VideoFrameResponse,
    VideoOutputReferences,
    VideoPredictionResponse,
    VideoSummaryResponse,
)
from .uploads import StoredUpload


def _detections(values: list[dict[str, object]]) -> list[DetectionResponse]:
    return [DetectionResponse.model_validate(value) for value in values]


class InferenceService:
    """Own API orchestration while delegating all inference to project pipelines."""

    def __init__(self, engine: DetectorEngine, config: BackendConfig) -> None:
        self.engine = engine
        self.config = config

    def _reference(self, path: Path, content_type: str | None = None) -> OutputReference:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self.config.output_directory)
        except ValueError as exc:
            raise OutputMediaError(
                f"Generated output is outside the configured API directory: {resolved}"
            ) from exc
        media_type = content_type or mimetypes.guess_type(resolved.name)[0]
        return OutputReference(
            url=f"/outputs/{quote(relative.as_posix())}",
            filename=resolved.name,
            content_type=media_type or "application/octet-stream",
        )

    def _metadata(
        self,
        upload: StoredUpload,
        media_type: str,
        started_at: float,
    ) -> ProcessingMetadata:
        return ProcessingMetadata(
            request_id=upload.request_id,
            media_type=media_type,
            duration_ms=round((perf_counter() - started_at) * 1000.0, 3),
            device=self.engine.device.value,
            checkpoint_sha256=self.engine.checkpoint.sha256,
        )

    def predict_image(self, upload: StoredUpload) -> ImagePredictionResponse:
        started_at = perf_counter()
        output = infer_image(
            self.engine,
            upload.path,
            output_directory=self.config.output_directory / "images",
            save_media=True,
            save_detections=True,
        )
        if output.annotated_media is None or output.detections_file is None:
            raise OutputMediaError("Image pipeline did not publish required API outputs")
        result = output.result.to_dict()
        return ImagePredictionResponse(
            original_filename=upload.original_filename,
            width=output.result.width,
            height=output.result.height,
            detections=_detections(result["detections"]),
            processing=self._metadata(upload, "image", started_at),
            annotated_output=self._reference(output.annotated_media),
            detections_output=self._reference(
                output.detections_file,
                "application/json",
            ),
        )

    def _read_video_frames(self, path: Path) -> list[VideoFrameResponse]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            records = [json.loads(line) for line in lines]
        except (OSError, json.JSONDecodeError) as exc:
            raise OutputMediaError(f"Cannot read video detection output {path}: {exc}") from exc
        return [
            VideoFrameResponse(
                processed_index=record["processed_index"],
                frame_index=record["frame_index"],
                timestamp_ms=record["timestamp_ms"],
                width=record["width"],
                height=record["height"],
                detections=_detections(record["detections"]),
            )
            for record in records
        ]

    def predict_video(self, upload: StoredUpload) -> VideoPredictionResponse:
        started_at = perf_counter()
        inference = self.config.inference
        output = infer_video(
            self.engine,
            upload.path,
            output_directory=self.config.output_directory / "videos",
            frame_skip=inference.input.frame_skip,
            save_media=True,
            save_detections=True,
            codec=inference.output.video_codec,
            video_extension=inference.output.video_extension,
        )
        summary = output.summary
        if summary.annotated_media is None or summary.detections_file is None:
            raise OutputMediaError("Video pipeline did not publish required API outputs")
        frames = self._read_video_frames(Path(summary.detections_file))
        return VideoPredictionResponse(
            original_filename=upload.original_filename,
            processed_frames=summary.frames_processed,
            detections=frames,
            processing=self._metadata(upload, "video", started_at),
            output=VideoOutputReferences(
                annotated_video=self._reference(Path(summary.annotated_media)),
                detections=self._reference(
                    Path(summary.detections_file),
                    "application/x-ndjson",
                ),
                summary=self._reference(
                    Path(summary.summary_file),
                    "application/json",
                ),
            ),
            summary=VideoSummaryResponse(
                status=summary.status,
                frames_read=summary.frames_read,
                frames_processed=summary.frames_processed,
                frames_written=summary.frames_written,
                detections_total=summary.detections_total,
                detections_per_class=summary.detections_per_class,
                source_fps=summary.source_fps,
                output_fps=summary.output_fps,
                frame_skip=summary.frame_skip,
            ),
        )
