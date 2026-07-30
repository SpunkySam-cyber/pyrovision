"""Checkpoint resolution, integrity verification, and class validation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import CheckpointConfig
from .errors import CheckpointError, CheckpointIntegrityError, ClassNameMismatchError
from .hashing import sha256_file


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """Verified checkpoint selected for inference."""

    path: Path
    sha256: str
    expected_sha256: str | None
    epoch: int | None
    source: str


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CheckpointError(f"Metrics record does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"Cannot read metrics record {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"Metrics record must contain a JSON object: {path}")
    return value


def _auto_candidates(
    selected_path: str | None, experiment_id: str | None, project_root: Path
) -> list[Path]:
    candidates: list[Path] = []
    if selected_path:
        stored = Path(selected_path)
        candidates.append(stored if stored.is_absolute() else project_root / stored)
    if experiment_id:
        candidates.append(
            project_root
            / "runs"
            / "pyrovision"
            / f"{experiment_id}_train"
            / "weights"
            / "best.pt"
        )
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def resolve_checkpoint(
    config: CheckpointConfig, project_root: Path
) -> ResolvedCheckpoint:
    """Resolve a checkpoint and verify it against its expected SHA-256 digest."""
    root = project_root.resolve()
    expected_sha256 = config.sha256
    epoch: int | None = None
    source = "explicit"

    if config.path == "auto":
        source = "metrics"
        metrics_path = (
            config.metrics_file.resolve()
            if config.metrics_file.is_absolute()
            else (root / config.metrics_file).resolve()
        )
        metrics = _load_metrics(metrics_path)
        training = metrics.get("training") or {}
        if not isinstance(training, dict):
            raise CheckpointError("training must be a JSON object")
        selected = training.get("selected_checkpoint") or {}
        if not isinstance(selected, dict):
            raise CheckpointError("training.selected_checkpoint must be a JSON object")
        expected_sha256 = expected_sha256 or selected.get("sha256")
        selected_epoch = selected.get("epoch")
        epoch = int(selected_epoch) if selected_epoch is not None else None
        candidates = _auto_candidates(
            selected.get("path"), metrics.get("experiment_id"), root
        )
    else:
        checkpoint_path = Path(config.path)
        candidates = [
            checkpoint_path.resolve()
            if checkpoint_path.is_absolute()
            else (root / checkpoint_path).resolve()
        ]

    checkpoint = next((candidate for candidate in candidates if candidate.is_file()), None)
    if checkpoint is None:
        rendered = ", ".join(str(candidate) for candidate in candidates) or "none"
        raise CheckpointError(f"Checkpoint was not found; checked: {rendered}")

    actual_sha256 = sha256_file(checkpoint)
    if config.verify_sha256:
        if not expected_sha256:
            raise CheckpointIntegrityError(
                "Checkpoint SHA-256 verification is enabled but no expected digest is available"
            )
        if actual_sha256.lower() != str(expected_sha256).lower():
            raise CheckpointIntegrityError(
                f"Checkpoint SHA-256 mismatch for {checkpoint}: expected "
                f"{str(expected_sha256).lower()}, got {actual_sha256}"
            )

    return ResolvedCheckpoint(
        path=checkpoint,
        sha256=actual_sha256,
        expected_sha256=(str(expected_sha256).lower() if expected_sha256 else None),
        epoch=epoch,
        source=source,
    )


def normalize_class_names(
    names: Mapping[int | str, str] | Sequence[str],
) -> tuple[str, ...]:
    """Normalize YOLO list/dict names and require contiguous IDs from zero."""
    if isinstance(names, Mapping):
        try:
            normalized = {int(class_id): str(name) for class_id, name in names.items()}
        except (TypeError, ValueError) as exc:
            raise ClassNameMismatchError("Checkpoint class IDs must be integers") from exc
        if len(normalized) != len(names):
            raise ClassNameMismatchError("Checkpoint contains duplicate normalized class IDs")
        expected_ids = list(range(len(normalized)))
        if sorted(normalized) != expected_ids:
            raise ClassNameMismatchError(
                f"Checkpoint class IDs must be contiguous from zero; got {sorted(normalized)}"
            )
        ordered = tuple(normalized[index].strip() for index in expected_ids)
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        ordered = tuple(str(name).strip() for name in names)
    else:
        raise ClassNameMismatchError("Checkpoint class names must be a mapping or sequence")
    if not ordered or any(not name for name in ordered):
        raise ClassNameMismatchError("Checkpoint class names cannot be empty")
    return ordered


def validate_class_names(
    actual: Mapping[int | str, str] | Sequence[str], expected: Sequence[str]
) -> tuple[str, ...]:
    """Require exact class count, order, spelling, and casing."""
    actual_names = normalize_class_names(actual)
    expected_names = tuple(str(name).strip() for name in expected)
    if actual_names != expected_names:
        raise ClassNameMismatchError(
            f"Checkpoint classes {actual_names} do not match expected classes {expected_names}"
        )
    return actual_names
