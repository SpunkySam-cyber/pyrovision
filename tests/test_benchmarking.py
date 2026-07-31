from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pyrovision.benchmarking import (  # noqa: E402
    load_benchmark_config,
    merge_device_reports,
    summarize_samples,
)
from pyrovision.checkpoints import ResolvedCheckpoint  # noqa: E402
from pyrovision.config import (  # noqa: E402
    CheckpointConfig,
    InferenceConfig,
    InputConfig,
    ModelConfig,
    OutputConfig,
)
from pyrovision.device import ResolvedDevice  # noqa: E402
from pyrovision.errors import ConfigurationError  # noqa: E402
from pyrovision.model import DetectorEngine  # noqa: E402


class FakeTimedResult:
    boxes = None
    speed = {"preprocess": 1.25, "inference": 5.5, "postprocess": 0.75}


class FakeTimedModel:
    names = {0: "smoke", 1: "fire"}

    def predict(self, **kwargs: object) -> list[FakeTimedResult]:
        del kwargs
        return [FakeTimedResult()]


def make_engine(root: Path) -> DetectorEngine:
    config = InferenceConfig(
        schema_version=1,
        checkpoint=CheckpointConfig(verify_sha256=False),
        device="cpu",
        model=ModelConfig(half=False),
        input=InputConfig(),
        output=OutputConfig(),
        project_root=root,
    )
    checkpoint = ResolvedCheckpoint(
        path=root / "best.pt",
        sha256="a" * 64,
        expected_sha256=None,
        epoch=54,
        source="test",
    )
    device = ResolvedDevice(
        requested="cpu",
        value="cpu",
        is_cuda=False,
        index=None,
        name="CPU",
    )
    return DetectorEngine(FakeTimedModel(), config, checkpoint, device)


class BenchmarkingTest(unittest.TestCase):
    def test_timed_prediction_preserves_result_and_framework_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = make_engine(Path(temp_dir))
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            normal = engine.predict_frame(frame, source="frame.jpg")
            timed = engine.predict_frame_timed(frame, source="frame.jpg")

        self.assertEqual(timed.result.to_dict(), normal.to_dict())
        self.assertEqual(timed.timing.preprocessing_ms, 1.25)
        self.assertEqual(timed.timing.inference_ms, 5.5)
        self.assertEqual(timed.timing.framework_postprocessing_ms, 0.75)
        self.assertGreaterEqual(timed.timing.project_postprocessing_ms, 0.0)
        self.assertGreaterEqual(timed.timing.engine_total_ms, 0.0)
        self.assertAlmostEqual(timed.timing.model_only_fps, 1000.0 / 5.5)

    def test_sample_summary_is_deterministic_and_rejects_invalid_values(self) -> None:
        summary = summarize_samples([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean"], 2.5)
        self.assertEqual(summary["median"], 2.5)
        self.assertEqual(summary["p95"], 3.85)
        with self.assertRaises(ValueError):
            summarize_samples([])
        with self.assertRaises(ValueError):
            summarize_samples([float("nan")])

    def test_benchmark_configuration_resolves_paths_and_rejects_typos(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "benchmark.yaml"
            config_path.write_text(
                """schema_version: 1
inference_config: configs/inference.yaml
image: media/image.jpg
video: media/video.mp4
output_directory: outputs/benchmark
report_path: metrics/benchmark.json
devices: [cpu, 'cuda:0']
warmup_iterations: 2
""",
                encoding="utf-8",
            )
            config = load_benchmark_config(config_path, root)
            config_path.write_text(
                """schema_version: 1
image: image.jpg
video: video.mp4
unknown: true
""",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigurationError):
                load_benchmark_config(config_path, root)
            config_path.write_text(
                """schema_version: 1
image: image.jpg
image: duplicate.jpg
video: video.mp4
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "Duplicate YAML key"):
                load_benchmark_config(config_path, root)

        self.assertEqual(config.devices, ("cpu", "cuda:0"))
        self.assertEqual(config.warmup_iterations, 2)
        self.assertEqual(config.image, (root / "media/image.jpg").resolve())

    def test_device_report_merge_preserves_every_isolated_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_benchmark_config(
                PROJECT_ROOT / "configs" / "benchmark.yaml",
                PROJECT_ROOT,
            )
            config = type(config)(
                **{
                    **config.__dict__,
                    "report_path": root / "report.json",
                }
            )
            common = {
                "configuration": {},
                "hardware": {"cuda_device_0": None},
                "methodology": {},
            }
            cpu = {**common, "devices": {"cpu": {"value": 1}}}
            cuda = {
                **common,
                "hardware": {"cuda_device_0": {"name": "GPU"}},
                "devices": {"cuda:0": {"value": 2}},
            }
            merged = merge_device_reports([cpu, cuda], config, root)

        self.assertEqual(set(merged["devices"]), {"cpu", "cuda:0"})
        self.assertEqual(merged["devices"]["cpu"]["value"], 1)
        self.assertEqual(merged["hardware"]["cuda_device_0"]["name"], "GPU")
        self.assertEqual(cpu["devices"], {"cpu": {"value": 1}})


if __name__ == "__main__":
    unittest.main()
