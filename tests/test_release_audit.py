from __future__ import annotations

import json
import re
import sys
import tomllib
import unittest
from pathlib import Path
from urllib.parse import unquote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pyrovision import __version__  # noqa: E402
from pyrovision.benchmarking import load_benchmark_config  # noqa: E402
from pyrovision.config import load_inference_config  # noqa: E402


class ReleaseAuditTest(unittest.TestCase):
    def test_version_and_package_metadata_are_release_consistent(self) -> None:
        manifest = tomllib.loads(
            (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        metadata = manifest["project"]

        self.assertEqual(metadata["version"], "1.0.0")
        self.assertEqual(__version__, metadata["version"])
        self.assertEqual(metadata["license"]["file"], "LICENSE")
        self.assertEqual(metadata["readme"], "README.md")
        self.assertIn("wheel>=0.45", manifest["build-system"]["requires"])
        self.assertEqual(
            metadata["scripts"]["pyrovision-api"],
            "pyrovision.api.__main__:main",
        )

    def test_release_files_and_documented_project_paths_exist(self) -> None:
        required = (
            "CHANGELOG.md",
            "LICENSE",
            "README.md",
            "RELEASE_NOTES.md",
            "configs/benchmark.yaml",
            "configs/inference.yaml",
            "docs/api.md",
            "docs/benchmark.md",
            "docs/dataset.md",
            "docs/deployment.md",
            "docs/evaluation.md",
            "docs/inference.md",
            "docs/project_log.md",
            "docs/training/yolo11s_baseline.md",
            "metrics/step6_benchmark.json",
            "metrics/step6_release.json",
            "requirements-backend.txt",
            "requirements-cuda129.txt",
            "requirements-test.txt",
            "requirements-training.txt",
            "scripts/benchmark.py",
            "scripts/evaluate.py",
            "scripts/infer.py",
            "scripts/prepare_dataset.py",
            "scripts/train.py",
            "scripts/verify_dataset.py",
        )
        missing = [path for path in required if not (PROJECT_ROOT / path).exists()]
        self.assertEqual(missing, [])

    def test_relative_markdown_links_resolve(self) -> None:
        markdown_files = [PROJECT_ROOT / "README.md", PROJECT_ROOT / "CHANGELOG.md"]
        markdown_files.append(PROJECT_ROOT / "RELEASE_NOTES.md")
        markdown_files.extend(sorted((PROJECT_ROOT / "docs").rglob("*.md")))
        pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
        broken: list[str] = []
        for document in markdown_files:
            content = document.read_text(encoding="utf-8")
            for raw_target in pattern.findall(content):
                target = raw_target.strip().split("#", 1)[0]
                if not target or re.match(r"^(?:https?://|mailto:|/)", target):
                    continue
                resolved = (document.parent / unquote(target)).resolve()
                if not resolved.exists():
                    broken.append(f"{document.relative_to(PROJECT_ROOT)} -> {raw_target}")
        self.assertEqual(broken, [])

    def test_versioned_configs_match_current_contract(self) -> None:
        inference = load_inference_config(
            PROJECT_ROOT / "configs" / "inference.yaml",
            project_root=PROJECT_ROOT,
        )
        benchmark = load_benchmark_config(
            PROJECT_ROOT / "configs" / "benchmark.yaml",
            PROJECT_ROOT,
        )

        self.assertEqual(inference.checkpoint.expected_classes, ("smoke", "fire"))
        self.assertEqual(benchmark.devices, ("cpu", "cuda:0"))
        self.assertEqual(
            benchmark.report_path,
            PROJECT_ROOT / "metrics" / "step6_benchmark.json",
        )

    def test_benchmark_report_has_required_devices_stages_and_hashes(self) -> None:
        report = json.loads(
            (PROJECT_ROOT / "metrics" / "step6_benchmark.json").read_text(
                encoding="utf-8"
            )
        )
        release = json.loads(
            (PROJECT_ROOT / "metrics" / "step6_release.json").read_text(
                encoding="utf-8"
            )
        )
        required_component_stages = {
            "preprocessing_ms",
            "model_inference_ms",
            "combined_postprocessing_ms",
            "project_postprocessing_ms",
            "model_call_ms",
            "engine_total_ms",
            "model_only_fps",
        }

        self.assertEqual(set(report["devices"]), {"cpu", "cuda:0"})
        self.assertEqual(len(report["checkpoint"]["sha256"]), 64)
        self.assertFalse(report["methodology"]["test_split_used"])
        self.assertTrue(report["methodology"]["device_process_isolation"])
        for value in report["inputs"].values():
            self.assertEqual(len(value["sha256"]), 64)
        for device in report["devices"].values():
            steady = device["image_components"]["steady_state"]
            self.assertTrue(required_component_stages <= set(steady))
            self.assertGreater(device["image_pipeline"]["end_to_end_fps"], 0)
            self.assertGreater(device["video_pipeline"]["end_to_end_fps"], 0)
            self.assertTrue(device["api"]["model_reused"])
        self.assertEqual(release["package"]["version"], "1.0.0")
        self.assertEqual(release["automated_tests"]["passed"], 69)
        self.assertEqual(release["release"]["recommended_tag"], "v1.0.0")


if __name__ == "__main__":
    unittest.main()
