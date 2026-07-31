from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pyrovision.checkpoints import (  # noqa: E402
    resolve_checkpoint,
    validate_class_names,
)
from pyrovision.config import (  # noqa: E402
    CheckpointConfig,
    ConfigurationError,
    InputConfig,
    ModelConfig,
    OutputConfig,
    load_inference_config,
)
from pyrovision.device import resolve_device  # noqa: E402
from pyrovision.errors import (  # noqa: E402
    CheckpointError,
    CheckpointIntegrityError,
    ClassNameMismatchError,
    DeviceResolutionError,
)
from pyrovision.types import BoundingBox, Detection, FrameResult  # noqa: E402


class FakeCuda:
    def __init__(self, available: bool, count: int = 0) -> None:
        self.available = available
        self.count = count

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count

    def get_device_name(self, index: int) -> str:
        return f"Fake GPU {index}"


class InferenceFoundationTest(unittest.TestCase):
    def test_repository_inference_config_loads_with_resolved_paths(self) -> None:
        config = load_inference_config(
            PROJECT_ROOT / "configs" / "inference.yaml", project_root=PROJECT_ROOT
        )

        self.assertEqual(config.schema_version, 1)
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.checkpoint.expected_classes, ("smoke", "fire"))
        self.assertEqual(config.model.confidence_threshold, 0.35)
        self.assertEqual(config.model.class_thresholds["fire"], 0.35)
        self.assertEqual(
            config.checkpoint.metrics_file,
            (PROJECT_ROOT / "metrics" / "yolo11s_baseline.json").resolve(),
        )
        self.assertEqual(
            config.output.directory, (PROJECT_ROOT / "outputs" / "inference").resolve()
        )

    def test_config_rejects_unknown_keys_and_unknown_threshold_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bad_key = root / "bad_key.yaml"
            bad_key.write_text("schema_version: 1\nunknown: true\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_inference_config(bad_key, project_root=root)

            bad_class = root / "bad_class.yaml"
            bad_class.write_text(
                "schema_version: 1\n"
                "checkpoint:\n  expected_classes: [smoke, fire]\n"
                "model:\n  class_thresholds:\n    steam: 0.5\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigurationError):
                load_inference_config(bad_class, project_root=root)

    def test_config_rejects_duplicates_missing_schema_and_invalid_types(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            duplicate = root / "duplicate.yaml"
            duplicate.write_text(
                "schema_version: 1\ndevice: cpu\ndevice: auto\n",
                encoding="utf-8",
            )
            missing_schema = root / "missing_schema.yaml"
            missing_schema.write_text("device: cpu\n", encoding="utf-8")

            with self.assertRaisesRegex(ConfigurationError, "Duplicate YAML key"):
                load_inference_config(duplicate, project_root=root)
            with self.assertRaisesRegex(ConfigurationError, "schema_version is required"):
                load_inference_config(missing_schema, project_root=root)
            with self.assertRaises(ConfigurationError):
                ModelConfig(confidence_threshold=float("nan"))
            with self.assertRaises(ConfigurationError):
                ModelConfig(class_thresholds={1: 0.5})
            with self.assertRaises(ConfigurationError):
                OutputConfig(video_codec="BAD!")
            with self.assertRaises(ConfigurationError):
                CheckpointConfig(expected_classes=("smoke", 1))

        self.assertEqual(InputConfig(source="  image.jpg  ").source, "image.jpg")

    def test_auto_checkpoint_uses_metrics_and_verifies_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "runs" / "pyrovision" / "demo_train" / "weights" / "best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"verified checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            metrics = root / "metrics.json"
            metrics.write_text(
                json.dumps(
                    {
                        "experiment_id": "demo",
                        "training": {
                            "selected_checkpoint": {
                                "path": "X:/missing/old/location/best.pt",
                                "epoch": 12,
                                "sha256": digest,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            resolved = resolve_checkpoint(
                CheckpointConfig(path="auto", metrics_file=metrics), root
            )

        self.assertEqual(resolved.path, checkpoint.resolve())
        self.assertEqual(resolved.sha256, digest)
        self.assertEqual(resolved.epoch, 12)
        self.assertEqual(resolved.source, "metrics")

    def test_checkpoint_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"unexpected bytes")
            config = CheckpointConfig(
                path=checkpoint,
                verify_sha256=True,
                sha256="0" * 64,
            )
            with self.assertRaises(CheckpointIntegrityError):
                resolve_checkpoint(config, root)

    def test_auto_checkpoint_rejects_malformed_metrics_contract(self) -> None:
        malformed_values = (
            {"training": {"selected_checkpoint": {"sha256": "bad"}}},
            {"training": {"selected_checkpoint": {"epoch": -1}}},
            {"training": {"selected_checkpoint": {"path": 123}}},
            {"experiment_id": [], "training": {}},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metrics = root / "metrics.json"
            for value in malformed_values:
                with self.subTest(value=value):
                    metrics.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(CheckpointError):
                        resolve_checkpoint(
                            CheckpointConfig(path="auto", metrics_file=metrics),
                            root,
                        )

    def test_class_names_require_exact_order_and_spelling(self) -> None:
        self.assertEqual(
            validate_class_names({0: "smoke", 1: "fire"}, ("smoke", "fire")),
            ("smoke", "fire"),
        )
        with self.assertRaises(ClassNameMismatchError):
            validate_class_names({0: "fire", 1: "smoke"}, ("smoke", "fire"))
        with self.assertRaises(ClassNameMismatchError):
            validate_class_names({0: "smoke", 2: "fire"}, ("smoke", "fire"))
        with self.assertRaises(ClassNameMismatchError):
            validate_class_names({0.5: "smoke", 1: "fire"}, ("smoke", "fire"))
        with self.assertRaises(ClassNameMismatchError):
            validate_class_names({False: "smoke", True: "fire"}, ("smoke", "fire"))
        with self.assertRaises(ClassNameMismatchError):
            validate_class_names(["smoke", 1], ("smoke", "fire"))

    def test_device_resolution_supports_cpu_auto_and_cuda(self) -> None:
        no_cuda = SimpleNamespace(cuda=FakeCuda(False))
        two_gpus = SimpleNamespace(cuda=FakeCuda(True, count=2))

        self.assertEqual(resolve_device("cpu", no_cuda).value, "cpu")
        self.assertEqual(resolve_device("auto", no_cuda).value, "cpu")
        resolved = resolve_device("cuda:1", two_gpus)
        self.assertEqual(resolved.value, "cuda:1")
        self.assertEqual(resolved.name, "Fake GPU 1")
        self.assertTrue(resolved.use_half)

        with self.assertRaises(DeviceResolutionError):
            resolve_device("cuda", no_cuda)
        with self.assertRaises(DeviceResolutionError):
            resolve_device("cuda:2", two_gpus)
        with self.assertRaises(DeviceResolutionError):
            resolve_device("gpu", two_gpus)

    def test_project_types_serialize_deterministically(self) -> None:
        detection = Detection(
            class_id=1,
            class_name="fire",
            confidence=0.912345678,
            bbox=BoundingBox(10.123456, 20.0, 100.987654, 200.0),
        )
        result = FrameResult(
            source="sample.jpg",
            frame_index=0,
            timestamp_ms=0.0,
            width=640,
            height=480,
            detections=[detection],
        )

        self.assertEqual(
            result.to_dict(),
            {
                "source": "sample.jpg",
                "frame_index": 0,
                "timestamp_ms": 0.0,
                "width": 640,
                "height": 480,
                "detections": [
                    {
                        "class_id": 1,
                        "class": "fire",
                        "confidence": 0.912346,
                        "bbox": [10.1235, 20.0, 100.9877, 200.0],
                    }
                ],
            },
        )
        with self.assertRaises(ValueError):
            BoundingBox(-1.0, 0.0, 10.0, 10.0)


if __name__ == "__main__":
    unittest.main()
