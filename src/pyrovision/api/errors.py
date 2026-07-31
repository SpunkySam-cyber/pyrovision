"""Structured HTTP error policy for the FastAPI adapter."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..errors import (
    CheckpointError,
    ConfigurationError,
    DeviceResolutionError,
    InferenceError,
    InputMediaError,
    OutputMediaError,
    PyroVisionError,
)


LOGGER = logging.getLogger(__name__)


class ApiError(Exception):
    """Expected request failure with a stable public error code."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def error_body(
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "success": False,
        "error": {"code": code, "message": message, "details": details},
    }


def _project_error(error: PyroVisionError) -> tuple[int, str]:
    if isinstance(error, InputMediaError):
        return 422, "invalid_media"
    if isinstance(error, OutputMediaError):
        return 500, "output_generation_failed"
    if isinstance(error, InferenceError):
        return 500, "inference_failed"
    if isinstance(
        error,
        (CheckpointError, ConfigurationError, DeviceResolutionError),
    ):
        return 503, "model_unavailable"
    return 500, "pyrovision_error"


def install_exception_handlers(app: FastAPI) -> None:
    """Install deterministic JSON handlers without leaking internal tracebacks."""

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, error: ApiError) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=error.status_code,
            content=error_body(error.code, error.message, error.details),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        del request
        details = {
            "errors": [
                {
                    "location": [str(item) for item in issue["loc"]],
                    "message": issue["msg"],
                    "type": issue["type"],
                }
                for issue in error.errors()
            ]
        }
        return JSONResponse(
            status_code=422,
            content=error_body(
                "validation_error",
                "Request validation failed",
                details,
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        request: Request,
        error: StarletteHTTPException,
    ) -> JSONResponse:
        del request
        codes = {404: "not_found", 405: "method_not_allowed"}
        return JSONResponse(
            status_code=error.status_code,
            content=error_body(
                codes.get(error.status_code, "http_error"),
                str(error.detail),
            ),
        )

    @app.exception_handler(PyroVisionError)
    async def handle_project_error(
        request: Request,
        error: PyroVisionError,
    ) -> JSONResponse:
        del request
        status_code, code = _project_error(error)
        LOGGER.warning("PyroVision request failed: %s", error)
        return JSONResponse(
            status_code=status_code,
            content=error_body(code, str(error)),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request,
        error: Exception,
    ) -> JSONResponse:
        LOGGER.exception("Unexpected API error for %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content=error_body(
                "internal_error",
                "An unexpected server error occurred",
            ),
        )
