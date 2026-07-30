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
