"""Deterministic CPU/CUDA device resolution for inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import DeviceResolutionError


@dataclass(frozen=True)
class ResolvedDevice:
    """Concrete runtime device suitable for PyTorch and Ultralytics."""

    requested: str
    value: str
    is_cuda: bool
    index: int | None
    name: str

    @property
    def use_half(self) -> bool:
        """FP16 inference is eligible only on CUDA."""
        return self.is_cuda


def _import_torch() -> Any | None:
    try:
        import torch
    except (ImportError, OSError):
        return None
    return torch


def resolve_device(requested: str, torch_module: Any | None = None) -> ResolvedDevice:
    """Resolve auto/cpu/cuda/cuda:N with explicit availability checks."""
    if not isinstance(requested, str) or not requested.strip():
        raise DeviceResolutionError("Device must be auto, cpu, cuda, or cuda:N")
    normalized = requested.strip().lower()
    if normalized == "cpu":
        return ResolvedDevice(
            requested=normalized,
            value="cpu",
            is_cuda=False,
            index=None,
            name="CPU",
        )
    if (
        normalized != "auto"
        and normalized != "cuda"
        and not normalized.startswith("cuda:")
    ):
        raise DeviceResolutionError(f"Unsupported device '{requested}'")

    torch_api = torch_module if torch_module is not None else _import_torch()
    try:
        cuda_available = bool(
            torch_api is not None
            and getattr(torch_api, "cuda", None) is not None
            and torch_api.cuda.is_available()
        )
    except Exception as exc:
        raise DeviceResolutionError(
            f"Could not query CUDA availability: {exc}"
        ) from exc
    if normalized == "auto" and not cuda_available:
        return ResolvedDevice(
            requested=normalized,
            value="cpu",
            is_cuda=False,
            index=None,
            name="CPU",
        )
    if not cuda_available:
        raise DeviceResolutionError(
            f"Device '{requested}' requires CUDA, but CUDA is unavailable"
        )

    if normalized in {"auto", "cuda"}:
        index = 0
    else:
        index_text = normalized.partition(":")[2]
        try:
            index = int(index_text)
        except ValueError as exc:
            raise DeviceResolutionError(f"Invalid CUDA device '{requested}'") from exc
        if index < 0:
            raise DeviceResolutionError(f"Invalid CUDA device '{requested}'")

    try:
        device_count = int(torch_api.cuda.device_count())
    except Exception as exc:
        raise DeviceResolutionError(f"Could not query CUDA devices: {exc}") from exc
    if index >= device_count:
        raise DeviceResolutionError(
            f"CUDA device index {index} is unavailable; detected {device_count} device(s)"
        )
    try:
        device_name = str(torch_api.cuda.get_device_name(index))
    except Exception as exc:
        raise DeviceResolutionError(
            f"Could not query CUDA device {index}: {exc}"
        ) from exc
    return ResolvedDevice(
        requested=normalized,
        value=f"cuda:{index}",
        is_cuda=True,
        index=index,
        name=device_name,
    )
