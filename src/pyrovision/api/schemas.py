"""Public, framework-independent API response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DetectionResponse(StrictModel):
    class_id: int = Field(ge=0, examples=[1])
    class_name: str = Field(alias="class", examples=["fire"])
    confidence: float = Field(ge=0.0, le=1.0, examples=[0.91])
    bbox: tuple[float, float, float, float] = Field(
        description="Pixel-space [x_min, y_min, x_max, y_max] coordinates",
        examples=[(104.2, 82.5, 310.8, 287.1)],
    )


class OutputReference(StrictModel):
    url: str = Field(examples=["/outputs/images/request_annotated.jpg"])
    filename: str = Field(examples=["request_annotated.jpg"])
    content_type: str = Field(examples=["image/jpeg"])


class ProcessingMetadata(StrictModel):
    request_id: str
    media_type: Literal["image", "video"]
    duration_ms: float = Field(ge=0.0)
    device: str
    checkpoint_sha256: str = Field(min_length=64, max_length=64)


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    version: str
    model_loaded: bool
    checkpoint_sha256: str
    checkpoint_identifier: str
    device: str
    uptime_seconds: float = Field(ge=0.0)


class ImagePredictionResponse(StrictModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "success": True,
                "original_filename": "warehouse.jpg",
                "width": 1280,
                "height": 720,
                "detections": [
                    {
                        "class_id": 1,
                        "class": "fire",
                        "confidence": 0.91,
                        "bbox": [104.2, 82.5, 310.8, 287.1],
                    }
                ],
                "processing": {
                    "request_id": "8f6c7c0f",
                    "media_type": "image",
                    "duration_ms": 42.1,
                    "device": "cuda:0",
                    "checkpoint_sha256": "a" * 64,
                },
                "annotated_output": {
                    "url": "/outputs/images/8f6c7c0f_annotated.jpg",
                    "filename": "8f6c7c0f_annotated.jpg",
                    "content_type": "image/jpeg",
                },
                "detections_output": {
                    "url": "/outputs/images/8f6c7c0f_detections.json",
                    "filename": "8f6c7c0f_detections.json",
                    "content_type": "application/json",
                },
            }
        },
    )
    success: Literal[True] = True
    original_filename: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    detections: list[DetectionResponse]
    processing: ProcessingMetadata
    annotated_output: OutputReference
    detections_output: OutputReference


class VideoFrameResponse(StrictModel):
    processed_index: int = Field(ge=0)
    frame_index: int = Field(ge=0)
    timestamp_ms: float | None = Field(default=None, ge=0.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    detections: list[DetectionResponse]


class VideoSummaryResponse(StrictModel):
    status: Literal["complete", "interrupted"]
    frames_read: int = Field(ge=0)
    frames_processed: int = Field(ge=0)
    frames_written: int = Field(ge=0)
    detections_total: int = Field(ge=0)
    detections_per_class: dict[str, int]
    source_fps: float = Field(gt=0.0)
    output_fps: float | None = Field(default=None, gt=0.0)
    frame_skip: int = Field(ge=0)


class VideoOutputReferences(StrictModel):
    annotated_video: OutputReference
    detections: OutputReference
    summary: OutputReference


class VideoPredictionResponse(StrictModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "success": True,
                "original_filename": "camera.mp4",
                "processed_frames": 240,
                "detections": [],
                "processing": {
                    "request_id": "12da4d09",
                    "media_type": "video",
                    "duration_ms": 2150.4,
                    "device": "cuda:0",
                    "checkpoint_sha256": "a" * 64,
                },
                "output": {
                    "annotated_video": {
                        "url": "/outputs/videos/12da4d09_annotated.mp4",
                        "filename": "12da4d09_annotated.mp4",
                        "content_type": "video/mp4",
                    },
                    "detections": {
                        "url": "/outputs/videos/12da4d09_detections.jsonl",
                        "filename": "12da4d09_detections.jsonl",
                        "content_type": "application/x-ndjson",
                    },
                    "summary": {
                        "url": "/outputs/videos/12da4d09_summary.json",
                        "filename": "12da4d09_summary.json",
                        "content_type": "application/json",
                    },
                },
                "summary": {
                    "status": "complete",
                    "frames_read": 240,
                    "frames_processed": 240,
                    "frames_written": 240,
                    "detections_total": 18,
                    "detections_per_class": {"smoke": 7, "fire": 11},
                    "source_fps": 30.0,
                    "output_fps": 30.0,
                    "frame_skip": 0,
                },
            }
        },
    )
    success: Literal[True] = True
    original_filename: str
    processed_frames: int = Field(ge=0)
    detections: list[VideoFrameResponse]
    processing: ProcessingMetadata
    output: VideoOutputReferences
    summary: VideoSummaryResponse


class ErrorDetail(StrictModel):
    code: str
    message: str
    details: dict[str, object] | None = None


class ErrorResponse(StrictModel):
    success: Literal[False] = False
    error: ErrorDetail
