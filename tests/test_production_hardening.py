from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pyrovision.errors import OutputMediaError  # noqa: E402
from pyrovision.outputs import (  # noqa: E402
    AnnotatedVideoWriter,
    JsonlWriter,
    allocate_output_stem,
    write_json_atomic,
)
from pyrovision.streaming import close_resources  # noqa: E402


class FailingResource:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def close(self) -> None:
        self.calls.append(self.name)
        raise RuntimeError(f"{self.name} close failed")


class SuccessfulResource:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def close(self) -> None:
        self.calls.append(self.name)


class FakeOpenWriter:
    def __init__(self) -> None:
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        pass

    def release(self) -> None:
        self.released = True


class ProductionHardeningTest(unittest.TestCase):
    def test_atomic_json_is_sorted_strict_and_leaves_no_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "result.json"
            write_json_atomic(target, {"z": 1, "a": {"d": 2, "b": 1}})
            serialized = target.read_text(encoding="utf-8")
            parsed = json.loads(serialized)
            temporary_files = list(root.glob("*.tmp"))

            with self.assertRaisesRegex(OutputMediaError, "Cannot serialize JSON"):
                write_json_atomic(root / "nan.json", {"value": float("nan")})

        self.assertEqual(parsed, {"a": {"b": 1, "d": 2}, "z": 1})
        self.assertLess(serialized.index('"a"'), serialized.index('"z"'))
        self.assertEqual(temporary_files, [])

    def test_jsonl_is_sorted_strict_and_rejects_nonfinite_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "records.jsonl"
            writer = JsonlWriter(target)
            writer.write({"z": 1, "a": 2})
            with self.assertRaisesRegex(OutputMediaError, "Cannot serialize JSONL"):
                writer.write({"value": float("inf")})
            writer.close()
            serialized = target.read_text(encoding="utf-8")

        self.assertEqual(serialized, '{"a":2,"z":1}\n')
        self.assertEqual(writer.records_written, 1)

    def test_collision_allocator_is_deterministic_across_requested_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "run_summary.json").write_text("{}", encoding="utf-8")
            (root / "run_2_detections.jsonl").write_text("", encoding="utf-8")

            selected = allocate_output_stem(
                root,
                "run",
                ("_summary.json", "_detections.jsonl"),
            )

        self.assertEqual(selected, "run_3")
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(OutputMediaError):
                allocate_output_stem(Path(temp_dir), "../escape", (".json",))

    def test_cleanup_attempts_every_resource_and_aggregates_failures(self) -> None:
        calls: list[str] = []
        with self.assertLogs("pyrovision.streaming", level="ERROR"):
            error = close_resources(
                FailingResource("display", calls),
                SuccessfulResource("jsonl", calls),
                FailingResource("video", calls),
            )

        self.assertEqual(calls, ["display", "jsonl", "video"])
        self.assertIsInstance(error, OutputMediaError)
        self.assertIn("display close failed", str(error))
        self.assertIn("video close failed", str(error))

    def test_output_frame_contract_rejects_non_uint8_three_channel_arrays(self) -> None:
        invalid_frames = (
            np.zeros((10, 10), dtype=np.uint8),
            np.zeros((10, 10, 4), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FakeOpenWriter()
            writer = AnnotatedVideoWriter(
                Path(temp_dir) / "output.avi",
                codec="MJPG",
                fps=10.0,
                width=10,
                height=10,
                writer_factory=lambda *args: backend,
            )
            for frame in invalid_frames:
                with self.subTest(shape=frame.shape, dtype=frame.dtype):
                    with self.assertRaisesRegex(OutputMediaError, "uint8 BGR"):
                        writer.write(frame)
            writer.close()

        self.assertTrue(backend.released)


if __name__ == "__main__":
    unittest.main()
