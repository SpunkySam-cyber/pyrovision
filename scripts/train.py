"""Run documented YOLO11 baseline validation, smoke tests, and full training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CACHE_ROOT = PROJECT_ROOT / "artifacts" / "cache"
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_CACHE_ROOT / "ultralytics"))
(LOCAL_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(LOCAL_CACHE_ROOT / "ultralytics").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from check_environment import collect_environment  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolved_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"experiment_id", "model", "data", "metrics_file", "project", "validation", "training"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Training config is missing keys: {sorted(missing)}")
    return config


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Metrics record does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def dataset_class_names(data_config: str | Path) -> dict[int, str]:
    data = yaml.safe_load(resolved_path(data_config).read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, list):
        return {index: name for index, name in enumerate(names)}
    return {int(class_id): name for class_id, name in names.items()}


def validation_metrics(
    result: Any, target_class_names: dict[int, str]
) -> dict[str, Any]:
    box = result.box
    per_class: dict[str, dict[str, float | None]] = {}
    ap_class_index = getattr(box, "ap_class_index", [])
    for position, class_id in enumerate(ap_class_index):
        class_number = int(class_id)
        name = target_class_names.get(class_number, str(class_number))
        per_class[name] = {
            "precision": finite_float(box.p[position]),
            "recall": finite_float(box.r[position]),
            "map50": finite_float(box.ap50[position]),
            "map50_95": finite_float(box.ap[position]),
        }
    return {
        "precision": finite_float(box.mp),
        "recall": finite_float(box.mr),
        "map50": finite_float(box.map50),
        "map50_95": finite_float(box.map),
        "per_class": per_class,
        "speed_ms_per_image": {
            key: finite_float(value) for key, value in result.speed.items()
        },
        "save_dir": str(result.save_dir),
    }


def read_epoch_metrics(results_csv: Path) -> dict[str, Any]:
    with results_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise ValueError(f"No epoch rows found in {results_csv}")

    map_key = next(
        (key for key in rows[0] if "mAP50-95" in key and "metrics/" in key), None
    )
    if map_key is None:
        raise ValueError(f"Could not find mAP50-95 column in {results_csv}")
    best_row = max(rows, key=lambda row: finite_float(row.get(map_key)) or float("-inf"))

    tracked_suffixes = (
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "lr/pg0",
        "lr/pg1",
        "lr/pg2",
    )

    def selected(row: dict[str, str]) -> dict[str, float | int | None]:
        selected_values: dict[str, float | int | None] = {
            "epoch": int(float(row["epoch"]))
        }
        for key in tracked_suffixes:
            if key in row:
                selected_values[key] = finite_float(row[key])
        if "time" in row:
            selected_values["time_seconds"] = finite_float(row["time"])
        return selected_values

    return {
        "epochs_completed": len(rows),
        "best_epoch": int(float(best_row["epoch"])),
        "best_epoch_metrics": selected(best_row),
        "last_epoch_metrics": selected(rows[-1]),
        "per_epoch": [selected(row) for row in rows],
        "results_csv": str(results_csv),
    }


def stratified_sample_rows(
    rows: list[dict[str, str]], size: int, seed: int
) -> list[dict[str, str]]:
    if not 0 < size <= len(rows):
        raise ValueError(f"Subset size must be between 1 and {len(rows)}, got {size}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)

    exact = {category: size * len(items) / len(rows) for category, items in grouped.items()}
    counts = {category: math.floor(value) for category, value in exact.items()}
    remainder = size - sum(counts.values())
    order = sorted(
        grouped,
        key=lambda category: (exact[category] - counts[category], category),
        reverse=True,
    )
    for category in order[:remainder]:
        counts[category] += 1

    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for category in sorted(grouped):
        selected.extend(rng.sample(grouped[category], counts[category]))
    rng.shuffle(selected)
    return selected


def create_smoke_test_dataset(config: dict[str, Any]) -> dict[str, Any]:
    smoke = config["smoke_test"]
    manifest_path = resolved_path(smoke["manifest"])
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    selected = stratified_sample_rows(
        rows, size=int(smoke["subset_size"]), seed=int(smoke["subset_seed"])
    )

    base_data = yaml.safe_load(resolved_path(config["data"]).read_text(encoding="utf-8"))
    dataset_root = resolved_path(base_data["path"]).resolve()
    image_list = resolved_path(smoke["image_list"])
    image_list.parent.mkdir(parents=True, exist_ok=True)
    image_paths = [(dataset_root / row["image"]).resolve() for row in selected]
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Smoke subset contains missing image: {missing[0]}")
    image_list.write_text(
        "\n".join(path.as_posix() for path in image_paths) + "\n", encoding="utf-8"
    )

    smoke_data_config = resolved_path(smoke["data_config"])
    smoke_data = {
        "path": dataset_root.as_posix(),
        "train": image_list.resolve().as_posix(),
        "val": (dataset_root / "images" / "val").as_posix(),
        "test": (dataset_root / "images" / "test").as_posix(),
        "names": base_data["names"],
    }
    smoke_data_config.write_text(
        yaml.safe_dump(smoke_data, sort_keys=False), encoding="utf-8"
    )

    category_counts: dict[str, int] = defaultdict(int)
    for row in selected:
        category_counts[row["category"]] += 1
    return {
        "size": len(selected),
        "seed": int(smoke["subset_seed"]),
        "categories": dict(sorted(category_counts.items())),
        "image_list": str(image_list),
        "data_config": str(smoke_data_config),
    }


def ensure_cuda(allow_cpu: bool) -> None:
    import torch

    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError(
            "CUDA is unavailable in this Python environment. Run the CUDA environment "
            "setup and scripts/check_environment.py --require-cuda first."
        )


def validate_model(
    checkpoint: str | Path,
    config: dict[str, Any],
    run_name: str,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    target_names = dataset_class_names(config["data"])
    checkpoint_names = {
        class_id: model.names.get(class_id, str(class_id)) for class_id in target_names
    }
    arguments = dict(config["validation"])
    arguments.update(
        {
            "data": str(resolved_path(config["data"])),
            "project": str(resolved_path(config["project"])),
            "name": run_name,
            "exist_ok": True,
        }
    )
    result = model.val(**arguments)
    output = validation_metrics(result, target_names)
    output["checkpoint"] = str(checkpoint)
    output["class_name_alignment"] = {
        "target_names": target_names,
        "checkpoint_names_for_target_ids": checkpoint_names,
    }
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_file():
        output["checkpoint_sha256"] = sha256_file(checkpoint_path)
    return output


def run_baseline(config: dict[str, Any], metrics_path: Path) -> None:
    metrics = load_metrics(metrics_path)
    started_at = utc_now()
    result = validate_model(
        config["model"], config, f"{config['experiment_id']}_pretrain_validation"
    )
    result["started_at_utc"] = started_at
    result["completed_at_utc"] = utc_now()
    result["environment"] = collect_environment()
    metrics["pretrained_validation"] = result
    metrics["status"] = "pretrained_baseline_complete"
    write_metrics(metrics_path, metrics)


def checkpoint_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def record_training_progress(
    metrics_path: Path,
    stage: str,
    trainer: Any,
) -> None:
    """Persist the live results.csv history after every completed epoch."""
    results_csv = Path(trainer.save_dir) / "results.csv"
    if not results_csv.is_file():
        return
    metrics = load_metrics(metrics_path)
    record = metrics.get(stage) or {}
    record["run_dir"] = str(trainer.save_dir)
    record["epoch_metrics"] = read_epoch_metrics(results_csv)
    record["last_progress_at_utc"] = utc_now()
    metrics[stage] = record
    write_metrics(metrics_path, metrics)


def run_training(
    config: dict[str, Any], metrics_path: Path, smoke_test: bool
) -> None:
    import torch
    from ultralytics import YOLO

    metrics = load_metrics(metrics_path)
    stage = "smoke_test" if smoke_test else "training"
    run_name = f"{config['experiment_id']}_{'smoke_test_stratified' if smoke_test else 'train'}"
    arguments = dict(config["training"])
    data_config = resolved_path(config["data"])
    subset_record: dict[str, Any] | None = None
    if smoke_test:
        previous = metrics.get("smoke_test")
        if previous:
            previous["accepted"] = False
            previous["rejection_reason"] = (
                "Ultralytics fraction sampling selected 300 negatives and 1 positive "
                "from filename-sorted data; the run did not exercise box regression."
            )
            metrics.setdefault("smoke_test_attempts", []).append(previous)
            metrics["smoke_test"] = None
            metrics["status"] = "smoke_test_retry_in_progress"
            write_metrics(metrics_path, metrics)
        subset_record = create_smoke_test_dataset(config)
        data_config = Path(subset_record["data_config"])
        arguments.update(config["smoke_test"]["training_overrides"])
    arguments.update(
        {
            "data": str(data_config),
            "project": str(resolved_path(config["project"])),
            "name": run_name,
            "exist_ok": False,
        }
    )

    started_at = utc_now()
    initial_record: dict[str, Any] = {
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "arguments": arguments,
        "environment_at_start": collect_environment(),
        "metric_logging": (
            "Ultralytics results.csv and this JSON record are updated after every epoch."
        ),
    }
    if subset_record is not None:
        initial_record["subset"] = subset_record
    metrics[stage] = initial_record
    metrics["status"] = f"{stage}_in_progress"
    write_metrics(metrics_path, metrics)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    model = YOLO(config["model"])
    model.add_callback(
        "on_fit_epoch_end",
        lambda trainer: record_training_progress(metrics_path, stage, trainer),
    )
    try:
        model.train(**arguments)
    except BaseException as exc:
        failed_metrics = load_metrics(metrics_path)
        failed_record = failed_metrics.get(stage) or initial_record
        failed_record["failed_at_utc"] = utc_now()
        failed_record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        if getattr(model, "trainer", None) is not None:
            failed_record["run_dir"] = str(model.trainer.save_dir)
        failed_metrics[stage] = failed_record
        failed_metrics["status"] = f"{stage}_failed"
        write_metrics(metrics_path, failed_metrics)
        raise
    completed_at = utc_now()
    save_dir = Path(model.trainer.save_dir)
    results_csv = save_dir / "results.csv"
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"
    record: dict[str, Any] = {
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "run_dir": str(save_dir),
        "arguments": arguments,
        "epoch_metrics": read_epoch_metrics(results_csv),
        "best_checkpoint": checkpoint_record(best_path),
        "last_checkpoint": checkpoint_record(last_path),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "environment_at_end": collect_environment(),
    }
    if subset_record is not None:
        record["subset"] = subset_record
        record["accepted"] = True

    if best_path.is_file():
        record["best_validation"] = validate_model(
            best_path, config, f"{run_name}_best_validation"
        )
    if not smoke_test and last_path.is_file():
        record["last_validation"] = validate_model(
            last_path, config, f"{run_name}_last_validation"
        )

    metrics[stage] = record
    if smoke_test:
        metrics["status"] = "smoke_test_complete"
    else:
        metrics["posttrained_validation"] = {
            "best": record.get("best_validation"),
            "last": record.get("last_validation"),
        }
        metrics["status"] = "training_complete"
    write_metrics(metrics_path, metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("baseline", "smoke-test", "train"), help="Experiment stage"
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "yolo11s_baseline.yaml"
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow execution without CUDA (not recommended for this project)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(PROJECT_ROOT)
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = load_config(config_path)
    metrics_path = resolved_path(config["metrics_file"])
    ensure_cuda(args.allow_cpu)
    if args.mode == "baseline":
        run_baseline(config, metrics_path)
    elif args.mode == "smoke-test":
        run_training(config, metrics_path, smoke_test=True)
    else:
        run_training(config, metrics_path, smoke_test=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
