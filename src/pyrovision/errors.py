"""Project-owned exception hierarchy for local inference."""


class PyroVisionError(Exception):
    """Base exception for expected PyroVision failures."""


class ConfigurationError(PyroVisionError, ValueError):
    """Raised when inference configuration is missing or invalid."""


class CheckpointError(PyroVisionError):
    """Raised when a model checkpoint cannot be resolved or inspected."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when checkpoint integrity verification fails."""


class ClassNameMismatchError(CheckpointError):
    """Raised when a checkpoint's class names do not match the contract."""


class DeviceResolutionError(PyroVisionError):
    """Raised when the requested inference device is invalid or unavailable."""


class InferenceError(PyroVisionError):
    """Raised when model execution returns invalid or unusable results."""


class InputMediaError(PyroVisionError):
    """Raised when an input image or media source cannot be read."""


class OutputMediaError(PyroVisionError):
    """Raised when an annotated image or detection record cannot be written."""
