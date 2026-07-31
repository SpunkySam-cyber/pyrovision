"""Bounded, temporary persistence for multipart media uploads."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Literal
from uuid import uuid4

from fastapi import UploadFile

from ..sources import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS
from .errors import ApiError


CHUNK_SIZE = 1024 * 1024
GENERIC_BINARY_TYPES = {"application/octet-stream", "binary/octet-stream"}
IMAGE_CONTENT_TYPES = {
    ".bmp": {"image/bmp", "image/x-ms-bmp"},
    ".jpeg": {"image/jpeg"},
    ".jpg": {"image/jpeg"},
    ".png": {"image/png"},
    ".tif": {"image/tiff"},
    ".tiff": {"image/tiff"},
    ".webp": {"image/webp"},
}
VIDEO_CONTENT_TYPES = {
    ".avi": {"video/x-msvideo", "video/avi"},
    ".m4v": {"video/x-m4v", "video/mp4"},
    ".mkv": {"video/x-matroska", "video/matroska"},
    ".mov": {"video/quicktime"},
    ".mp4": {"video/mp4"},
    ".webm": {"video/webm"},
}


@dataclass(frozen=True)
class StoredUpload:
    request_id: str
    original_filename: str
    content_type: str
    size_bytes: int
    path: Path


def _validate_upload_type(
    upload: UploadFile,
    media_kind: Literal["image", "video"],
) -> tuple[str, str]:
    filename = upload.filename or ""
    if not filename.strip():
        raise ApiError(422, "missing_filename", "Upload filename is required")
    suffix = Path(filename).suffix.lower()
    supported_extensions = (
        SUPPORTED_IMAGE_EXTENSIONS
        if media_kind == "image"
        else SUPPORTED_VIDEO_EXTENSIONS
    )
    content_types = IMAGE_CONTENT_TYPES if media_kind == "image" else VIDEO_CONTENT_TYPES
    if suffix not in supported_extensions or suffix not in content_types:
        raise ApiError(
            415,
            "unsupported_media_type",
            f"Unsupported {media_kind} file extension: {suffix or '<none>'}",
        )
    content_type = (upload.content_type or "application/octet-stream").split(";", 1)[0]
    content_type = content_type.strip().lower()
    allowed = content_types[suffix] | GENERIC_BINARY_TYPES
    if content_type not in allowed:
        raise ApiError(
            415,
            "unsupported_media_type",
            f"Content type {content_type!r} does not match {suffix}",
        )
    return suffix, content_type


@asynccontextmanager
async def store_upload(
    upload: UploadFile,
    *,
    media_kind: Literal["image", "video"],
    temporary_directory: Path,
    max_size_bytes: int,
) -> AsyncIterator[StoredUpload]:
    """Write one upload with a hard byte limit and always remove it afterward."""
    temporary_path: Path | None = None
    try:
        suffix, content_type = _validate_upload_type(upload, media_kind)
        request_id = uuid4().hex
        temporary_path = temporary_directory / f"{request_id}{suffix}"
        size = 0
        try:
            temporary_directory.mkdir(parents=True, exist_ok=True)
            with temporary_path.open("xb") as destination:
                while chunk := await upload.read(CHUNK_SIZE):
                    size += len(chunk)
                    if size > max_size_bytes:
                        raise ApiError(
                            413,
                            "upload_too_large",
                            f"Upload exceeds the {max_size_bytes}-byte limit",
                            {"max_upload_size_bytes": max_size_bytes},
                        )
                    destination.write(chunk)
        except ApiError:
            raise
        except OSError as exc:
            raise ApiError(
                500,
                "temporary_storage_failed",
                "Could not persist the uploaded file",
            ) from exc
        if size == 0:
            raise ApiError(422, "empty_upload", "Uploaded file is empty")
        yield StoredUpload(
            request_id=request_id,
            original_filename=Path(upload.filename or "upload").name,
            content_type=content_type,
            size_bytes=size,
            path=temporary_path,
        )
    finally:
        try:
            await upload.close()
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
