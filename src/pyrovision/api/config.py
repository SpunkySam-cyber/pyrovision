"""Typed configuration for the PyroVision HTTP service."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ..config import InferenceConfig, load_inference_config
from ..errors import ConfigurationError


ENV_PREFIX = "PYROVISION_API_"


def _project_path(value: str, project_root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty path")
    path = Path(value.strip())
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _integer(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    return parsed


@dataclass(frozen=True)
class BackendConfig:
    """Validated API settings plus the reusable inference configuration."""

    inference: InferenceConfig
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)
    max_upload_size_bytes: int = 250 * 1024 * 1024
    output_directory: Path = Path("outputs/api")
    temporary_directory: Path = Path("outputs/api-temp")

    def __post_init__(self) -> None:
        if not isinstance(self.inference, InferenceConfig):
            raise ConfigurationError("backend inference config is invalid")
        if not isinstance(self.host, str) or not self.host.strip():
            raise ConfigurationError("API host must be a non-empty string")
        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise ConfigurationError("API port must be an integer")
        if not 1 <= self.port <= 65535:
            raise ConfigurationError("API port must be between 1 and 65535")
        if (
            isinstance(self.max_upload_size_bytes, bool)
            or not isinstance(self.max_upload_size_bytes, int)
            or self.max_upload_size_bytes <= 0
        ):
            raise ConfigurationError("API maximum upload size must be positive")
        if not isinstance(self.cors_origins, (tuple, list)) or not all(
            isinstance(origin, str) and origin.strip()
            for origin in self.cors_origins
        ):
            raise ConfigurationError("API CORS origins must be non-empty strings")
        if not isinstance(self.output_directory, Path):
            raise ConfigurationError("API output directory must be a path")
        if not isinstance(self.temporary_directory, Path):
            raise ConfigurationError("API temporary directory must be a path")

        output = self.output_directory.resolve()
        temporary = self.temporary_directory.resolve()
        if output == temporary:
            raise ConfigurationError(
                "API output and temporary directories must be different"
            )
        object.__setattr__(self, "host", self.host.strip())
        object.__setattr__(
            self,
            "cors_origins",
            tuple(origin.strip() for origin in self.cors_origins),
        )
        object.__setattr__(self, "output_directory", output)
        object.__setattr__(self, "temporary_directory", temporary)


def load_backend_config(
    environ: Mapping[str, str] | None = None,
    *,
    project_root: Path | None = None,
) -> BackendConfig:
    """Load backend settings from a small, explicit environment contract."""
    values = os.environ if environ is None else environ
    root = (
        project_root.resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    inference_path = _project_path(
        values.get(f"{ENV_PREFIX}INFERENCE_CONFIG", "configs/inference.yaml"),
        root,
        "API inference configuration",
    )
    inference = load_inference_config(inference_path, project_root=root)
    cors_value = values.get(
        f"{ENV_PREFIX}CORS_ORIGINS", "http://localhost:3000"
    )
    cors_origins = tuple(
        origin.strip() for origin in cors_value.split(",") if origin.strip()
    )
    return BackendConfig(
        inference=inference,
        host=values.get(f"{ENV_PREFIX}HOST", "127.0.0.1"),
        port=_integer(values.get(f"{ENV_PREFIX}PORT", "8000"), "API port"),
        cors_origins=cors_origins,
        max_upload_size_bytes=_integer(
            values.get(f"{ENV_PREFIX}MAX_UPLOAD_SIZE_BYTES", str(250 * 1024 * 1024)),
            "API maximum upload size",
        ),
        output_directory=_project_path(
            values.get(f"{ENV_PREFIX}OUTPUT_DIR", "outputs/api"),
            root,
            "API output directory",
        ),
        temporary_directory=_project_path(
            values.get(f"{ENV_PREFIX}TEMP_DIR", "outputs/api-temp"),
            root,
            "API temporary directory",
        ),
    )
