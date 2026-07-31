"""FastAPI application factory and thin HTTP routes."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from time import monotonic
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, FastAPI, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import InferenceConfig
from ..model import DetectorEngine
from .config import BackendConfig, load_backend_config
from .errors import install_exception_handlers
from .schemas import (
    ErrorResponse,
    HealthResponse,
    ImagePredictionResponse,
    VideoPredictionResponse,
)
from .service import InferenceService
from .uploads import store_upload


LOGGER = logging.getLogger(__name__)
EngineFactory = Callable[[InferenceConfig], DetectorEngine]


def create_app(
    config: BackendConfig | None = None,
    *,
    engine_factory: EngineFactory | None = None,
) -> FastAPI:
    """Create an independently testable API with one lifespan-owned model."""
    settings = config or load_backend_config()
    factory = engine_factory or DetectorEngine.from_config

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:
            settings.output_directory.mkdir(parents=True, exist_ok=True)
            settings.temporary_directory.mkdir(parents=True, exist_ok=True)
            engine = factory(settings.inference)
        except Exception as exc:
            LOGGER.exception("PyroVision API startup failed")
            raise RuntimeError(f"PyroVision model startup failed: {exc}") from exc
        application.state.engine = engine
        application.state.inference_service = InferenceService(engine, settings)
        application.state.started_monotonic = monotonic()
        application.state.model_load_count = 1
        LOGGER.info(
            "PyroVision model loaded on %s with checkpoint %s",
            engine.device.value,
            engine.checkpoint.sha256[:12],
        )
        yield

    application = FastAPI(
        title="PyroVision AI API",
        summary="Fire and smoke detection for uploaded images and videos",
        description=(
            "A thin HTTP adapter over the verified PyroVision DetectorEngine. "
            "The model is loaded once during application startup."
        ),
        version=__version__,
        lifespan=lifespan,
        contact={"name": "PyroVision AI"},
        license_info={"name": "MIT", "identifier": "MIT"},
    )
    application.state.backend_config = settings
    install_exception_handlers(application)
    if settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials="*" not in settings.cors_origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

    router = APIRouter()

    @router.get(
        "/health",
        response_model=HealthResponse,
        summary="Check API and model readiness",
        description="Returns success only after the verified model has loaded.",
    )
    async def health(request: Request) -> HealthResponse:
        engine = request.app.state.engine
        checkpoint_sha256 = engine.checkpoint.sha256
        return HealthResponse(
            version=__version__,
            model_loaded=True,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_identifier=checkpoint_sha256[:12],
            device=engine.device.value,
            uptime_seconds=round(
                monotonic() - request.app.state.started_monotonic,
                3,
            ),
        )

    @router.post(
        "/predict/image",
        response_model=ImagePredictionResponse,
        responses={
            413: {"model": ErrorResponse, "description": "Upload too large"},
            415: {"model": ErrorResponse, "description": "Unsupported image"},
            422: {"model": ErrorResponse, "description": "Invalid image"},
            500: {"model": ErrorResponse, "description": "Inference/output failure"},
            503: {"model": ErrorResponse, "description": "Model unavailable"},
        },
        summary="Detect fire and smoke in an image",
        description=(
            "Accepts one multipart image, validates and decodes it through the "
            "existing image pipeline, and publishes an annotated image."
        ),
    )
    async def predict_image(
        request: Request,
        file: Annotated[
            UploadFile,
            File(description="JPEG, PNG, BMP, TIFF, or WebP image"),
        ],
    ) -> ImagePredictionResponse:
        async with store_upload(
            file,
            media_kind="image",
            temporary_directory=settings.temporary_directory,
            max_size_bytes=settings.max_upload_size_bytes,
        ) as upload:
            return await run_in_threadpool(
                request.app.state.inference_service.predict_image,
                upload,
            )

    @router.post(
        "/predict/video",
        response_model=VideoPredictionResponse,
        responses={
            413: {"model": ErrorResponse, "description": "Upload too large"},
            415: {"model": ErrorResponse, "description": "Unsupported video"},
            422: {"model": ErrorResponse, "description": "Invalid video"},
            500: {"model": ErrorResponse, "description": "Inference/output failure"},
            503: {"model": ErrorResponse, "description": "Model unavailable"},
        },
        summary="Detect fire and smoke in a video",
        description=(
            "Processes a complete uploaded video through the existing ordered "
            "video pipeline. Live streaming and webcam capture are not included."
        ),
    )
    async def predict_video(
        request: Request,
        file: Annotated[
            UploadFile,
            File(description="AVI, M4V, MKV, MOV, MP4, or WebM video"),
        ],
    ) -> VideoPredictionResponse:
        async with store_upload(
            file,
            media_kind="video",
            temporary_directory=settings.temporary_directory,
            max_size_bytes=settings.max_upload_size_bytes,
        ) as upload:
            return await run_in_threadpool(
                request.app.state.inference_service.predict_video,
                upload,
            )

    application.include_router(router)
    application.mount(
        "/outputs",
        StaticFiles(directory=settings.output_directory, check_dir=False),
        name="outputs",
    )
    return application


app = create_app()
