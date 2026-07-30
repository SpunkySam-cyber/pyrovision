"""Reusable PyroVision inference components."""

from .config import InferenceConfig, load_inference_config
from .types import BoundingBox, Detection, FrameResult

__all__ = [
    "BoundingBox",
    "Detection",
    "FrameResult",
    "InferenceConfig",
    "load_inference_config",
]

__version__ = "0.1.0"
