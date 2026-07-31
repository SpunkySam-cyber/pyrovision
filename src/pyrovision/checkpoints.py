"""Checkpoint resolution, integrity verification, and class validation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import CheckpointConfig
from .errors import CheckpointError, CheckpointIntegrityError, ClassNameMismatchError
from .hashing import sha256_file


_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


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
        selected_digest = selected.get("sha256")
        if selected_digest is not None and (
            not isinstance(selected_digest, str)
            or not _SHA256_PATTERN.fullmatch(selected_digest)
        ):
            raise CheckpointError(
                "training.selected_checkpoint.sha256 must contain 64 hexadecimal characters"
            )
        expected_sha256 = expected_sha256 or selected_digest
        selected_epoch = selected.get("epoch")
        if selected_epoch is not None and (
            isinstance(selected_epoch, bool)
            or not isinstance(selected_epoch, int)
            or selected_epoch < 0
        ):
            raise CheckpointError(
                "training.selected_checkpoint.epoch must be a non-negative integer"
            )
        epoch = selected_epoch
        selected_path = selected.get("path")
        if selected_path is not None and (
            not isinstance(selected_path, str) or not selected_path.strip()
        ):
            raise CheckpointError(
                "training.selected_checkpoint.path must be a non-empty string"
            )
        experiment_id = metrics.get("experiment_id")
        if experiment_id is not None and (
            not isinstance(experiment_id, str) or not experiment_id.strip()
        ):
            raise CheckpointError("experiment_id must be a non-empty string")
        candidates = _auto_candidates(
            selected_path,
            experiment_id,
            root,
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

    try:
        actual_sha256 = sha256_file(checkpoint)
    except OSError as exc:
        raise CheckpointError(f"Cannot read checkpoint {checkpoint}: {exc}") from exc
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
        normalized: dict[int, str] = {}
        for class_id, name in names.items():
            if isinstance(class_id, bool):
                raise ClassNameMismatchError("Checkpoint class IDs must be integers")
            if isinstance(class_id, int):
                normalized_id = class_id
            elif isinstance(class_id, str) and class_id.isdecimal():
                normalized_id = int(class_id)
            else:
                raise ClassNameMismatchError("Checkpoint class IDs must be integers")
            if not isinstance(name, str):
                raise ClassNameMismatchError(
                    "Checkpoint class names must contain only strings"
                )
            if normalized_id in normalized:
                raise ClassNameMismatchError(
                    "Checkpoint contains duplicate normalized class IDs"
                )
            normalized[normalized_id] = name
        if len(normalized) != len(names):
            raise ClassNameMismatchError("Checkpoint contains duplicate normalized class IDs")
        expected_ids = list(range(len(normalized)))
        if sorted(normalized) != expected_ids:
            raise ClassNameMismatchError(
                f"Checkpoint class IDs must be contiguous from zero; got {sorted(normalized)}"
            )
        ordered = tuple(normalized[index].strip() for index in expected_ids)
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        if not all(isinstance(name, str) for name in names):
            raise ClassNameMismatchError(
                "Checkpoint class names must contain only strings"
            )
        ordered = tuple(name.strip() for name in names)
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
