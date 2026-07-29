"""Evaluate the selected PyroVision checkpoint once on the held-out test set."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from check_environment import collect_environment  # noqa: E402
from train import (  # noqa: E402
    checkpoint_record,
    dataset_class_names,
    ensure_cuda,
    load_config,
    load_metrics,
    resolved_path,
    validation_metrics,
    write_metrics,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def select_sanity_samples(
    manifest_path: Path,
    dataset_root: Path,
    seed: int,
) -> list[dict[str, str]]:
    """Select one deterministic test image from each D-Fire category."""
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["category"], []).append(row)
    required = ("negative", "smoke", "fire", "smoke+fire")
    missing = [category for category in required if category not in grouped]
    if missing:
        raise ValueError(f"Test manifest is missing categories: {missing}")
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for category in required:
        row = rng.choice(sorted(grouped[category], key=lambda item: item["image"]))
        image = (dataset_root / row["image"]).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"Sanity-check image does not exist: {image}")
        selected.append({"category": category, "image": str(image)})
    return selected


def prediction_record(
    result: Any,
    expected_category: str,
    source_image: str,
) -> dict[str, Any]:
    detections: list[dict[str, Any]] = []
    boxes = result.boxes
    if boxes is not None:
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist()
        for coordinates, confidence, class_id in zip(xyxy, confidences, classes):
            class_number = int(class_id)
            detections.append(
                {
                    "class_id": class_number,
                    "class": result.names[class_number],
                    "confidence": float(confidence),
                    "bbox": [float(value) for value in coordinates],
                }
            )
    return {
        "expected_category": expected_category,
        "source": source_image,
        "annotated_output": str(Path(result.save_dir) / Path(result.path).name),
        "detections": detections,
    }


def run_evaluation(config_path: Path, allow_cpu: bool) -> None:
    from ultralytics import YOLO

    config = load_config(config_path)
    metrics_path = resolved_path(config["metrics_file"])
    metrics = load_metrics(metrics_path)
    existing = metrics.get("test_evaluation")
    if isinstance(existing, dict) and existing.get("status") == "complete":
        raise RuntimeError("The held-out test split has already been evaluated")
    if metrics.get("status") != "training_complete":
        raise RuntimeError("Step 2 must be finalized before test evaluation")

    ensure_cuda(allow_cpu)
    selected = (metrics.get("training") or {}).get("selected_checkpoint") or {}
    checkpoint = Path(selected.get("path", "")).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint does not exist: {checkpoint}")

    started_at = utc_now()
    metrics["status"] = "test_evaluation_in_progress"
    metrics["test_evaluation"] = {
        "status": "in_progress",
        "started_at_utc": started_at,
        "checkpoint": checkpoint_record(checkpoint),
        "split": "test",
    }
    write_metrics(metrics_path, metrics)

    try:
        model = YOLO(str(checkpoint))
        target_names = dataset_class_names(config["data"])
        arguments = dict(config["validation"])
        arguments.update(
            {
                "data": str(resolved_path(config["data"])),
                "split": "test",
                "project": str(resolved_path(config["project"])),
                "name": f"{config['experiment_id']}_test_evaluation",
                "exist_ok": False,
                "plots": True,
            }
        )
        result = model.val(**arguments)
        evaluation = validation_metrics(result, target_names)
        evaluation["arguments"] = arguments
        evaluation["checkpoint"] = checkpoint_record(checkpoint)
        matrix = getattr(getattr(result, "confusion_matrix", None), "matrix", None)
        evaluation["confusion_matrix"] = (
            matrix.tolist() if matrix is not None else None
        )

        data = yaml.safe_load(resolved_path(config["data"]).read_text(encoding="utf-8"))
        dataset_root = resolved_path(data["path"]).resolve()
        manifest_path = dataset_root / "manifest.csv"
        with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            test_rows = [
                row for row in csv.DictReader(handle) if row["split"] == "test"
            ]
        instance_counts = getattr(result.box, "nt_per_class", [])
        evaluation["test_dataset"] = {
            "images": len(test_rows),
            "categories": dict(
                sorted(Counter(row["category"] for row in test_rows).items())
            ),
            "instances_per_class": {
                target_names[index]: int(value)
                for index, value in enumerate(instance_counts)
            },
        }
        samples = select_sanity_samples(
            manifest_path, dataset_root, seed=42
        )
        prediction_results = model.predict(
            source=[sample["image"] for sample in samples],
            imgsz=int(config["validation"]["imgsz"]),
            conf=0.25,
            iou=float(config["validation"]["iou"]),
            device=config["validation"]["device"],
            save=True,
            project=str(resolved_path(config["project"])),
            name=f"{config['experiment_id']}_test_sanity",
            exist_ok=False,
            verbose=False,
        )
        evaluation["sanity_samples"] = [
            prediction_record(result_item, sample["category"], sample["image"])
            for result_item, sample in zip(prediction_results, samples)
        ]
        save_dir = Path(evaluation["save_dir"])
        evaluation["artifacts"] = [
            str(path)
            for path in sorted(save_dir.iterdir())
            if path.is_file()
        ]
        evaluation["status"] = "complete"
        evaluation["started_at_utc"] = started_at
        evaluation["completed_at_utc"] = utc_now()
        evaluation["environment"] = collect_environment()
    except BaseException as exc:
        failed = load_metrics(metrics_path)
        failed["status"] = "test_evaluation_failed"
        failed["test_evaluation"] = {
            **(failed.get("test_evaluation") or {}),
            "status": "failed",
            "failed_at_utc": utc_now(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        write_metrics(metrics_path, failed)
        raise

    metrics = load_metrics(metrics_path)
    metrics["test_evaluation"] = evaluation
    metrics["status"] = "evaluation_complete"
    write_metrics(metrics_path, metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "yolo11s_baseline.yaml",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    run_evaluation(config_path, args.allow_cpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
