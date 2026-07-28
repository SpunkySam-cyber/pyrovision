from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train import (  # noqa: E402
    read_epoch_metrics,
    record_training_progress,
    stratified_sample_rows,
)


class TrainingToolsTest(unittest.TestCase):
    def test_stratified_subset_is_deterministic_and_proportional(self) -> None:
        rows = []
        for category, count in (("negative", 50), ("smoke", 30), ("smoke+fire", 15), ("fire", 5)):
            rows.extend(
                {"category": category, "image": f"{category}_{index}.jpg"}
                for index in range(count)
            )

        first = stratified_sample_rows(rows, size=20, seed=42)
        second = stratified_sample_rows(rows, size=20, seed=42)

        self.assertEqual(first, second)
        counts = {
            category: sum(row["category"] == category for row in first)
            for category in ("negative", "smoke", "smoke+fire", "fire")
        }
        self.assertEqual(
            counts, {"negative": 10, "smoke": 6, "smoke+fire": 3, "fire": 1}
        )

    def test_epoch_metrics_preserve_history_and_best_epoch(self) -> None:
        fieldnames = (
            "epoch",
            "time",
            "train/box_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
        )
        rows = (
            ("1", "10.0", "1.2", "2.0", "1.0", "0.3", "0.2", "0.25", "0.1"),
            ("2", "20.0", "1.0", "1.5", "0.9", "0.5", "0.4", "0.45", "0.3"),
            ("3", "30.0", "0.9", "1.3", "0.8", "0.4", "0.5", "0.44", "0.28"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            results_csv = Path(temp_dir) / "results.csv"
            with results_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fieldnames)
                writer.writerows(rows)

            summary = read_epoch_metrics(results_csv)

        self.assertEqual(summary["epochs_completed"], 3)
        self.assertEqual(summary["best_epoch"], 2)
        self.assertEqual(len(summary["per_epoch"]), 3)
        self.assertEqual(summary["last_epoch_metrics"]["epoch"], 3)
        self.assertEqual(
            summary["best_epoch_metrics"]["metrics/mAP50-95(B)"], 0.3
        )

    def test_training_progress_is_persisted_after_an_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metrics_path = root / "metrics.json"
            metrics_path.write_text(
                '{"status": "training_in_progress", "training": {}}\n',
                encoding="utf-8",
            )
            results_csv = root / "results.csv"
            results_csv.write_text(
                "epoch,metrics/mAP50-95(B),train/box_loss\n"
                "1,0.25,1.5\n",
                encoding="utf-8",
            )

            record_training_progress(
                metrics_path,
                "training",
                SimpleNamespace(save_dir=root),
            )

            persisted = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(
                persisted["training"]["epoch_metrics"]["epochs_completed"], 1
            )
            self.assertEqual(
                persisted["training"]["epoch_metrics"]["best_epoch"], 1
            )


if __name__ == "__main__":
    unittest.main()
