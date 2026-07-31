from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pyrovision.errors import InputMediaError, OutputMediaError  # noqa: E402
from pyrovision.sources import VideoReader, classify_media_path  # noqa: E402
from pyrovision.types import BoundingBox, Detection, FrameResult  # noqa: E402
from pyrovision.video import infer_video  # noqa: E402


def create_video(path: Path, frame_count: int = 6, fps: float = 6.0) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (96, 64)
    )
    if not writer.isOpened():
        raise RuntimeError("Test environment cannot create MJPG/AVI video")
    try:
        for index in range(frame_count):
            frame = np.full((64, 96, 3), index * 30, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


class StubVideoEngine:
    checkpoint = SimpleNamespace(sha256="c" * 64)
    device = SimpleNamespace(value="cpu")
    class_names = ("smoke", "fire")

    def __init__(self, with_detections: bool = True) -> None:
        self.with_detections = with_detections
        self.calls: list[tuple[int, float]] = []

    def predict_frame(
        self,
        frame: np.ndarray,
        *,
        source: str,
        frame_index: int,
        timestamp_ms: float,
    ) -> FrameResult:
        self.calls.append((frame_index, timestamp_ms))
        detections = []
        if self.with_detections and frame_index % 2 == 0:
            detections.append(
                Detection(
                    class_id=1,
                    class_name="fire",
                    confidence=0.8,
                    bbox=BoundingBox(10.0, 10.0, 50.0, 40.0),
                )
            )
        return FrameResult(
            source=source,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            width=frame.shape[1],
            height=frame.shape[0],
            detections=detections,
        )


class InterruptingEngine(StubVideoEngine):
    def predict_frame(self, *args: object, **kwargs: object) -> FrameResult:
        if len(self.calls) >= 2:
            raise KeyboardInterrupt
        return super().predict_frame(*args, **kwargs)


class ClosedWriter:
    def __init__(self) -> None:
        self.released = False

    def isOpened(self) -> bool:
        return False

    def release(self) -> None:
        self.released = True


class EmptyCapture:
    def __init__(self) -> None:
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: 96.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 64.0,
            cv2.CAP_PROP_FPS: 6.0,
            cv2.CAP_PROP_FRAME_COUNT: 0.0,
        }
        return values.get(property_id, 0.0)

    def read(self) -> tuple[bool, None]:
        return False, None

    def release(self) -> None:
        self.released = True


class VideoInferenceTest(unittest.TestCase):
    def test_video_reader_preserves_frame_order_and_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.avi"
            create_video(source, frame_count=4, fps=5.0)
            self.assertEqual(classify_media_path(source), "video")
            with VideoReader(source) as reader:
                frames = list(reader.frames())
                metadata = reader.metadata

        self.assertEqual([frame.frame_index for frame in frames], [0, 1, 2, 3])
        self.assertEqual(reader.frames_read, 4)
        self.assertAlmostEqual(metadata.fps, 5.0, places=1)
        timestamps = [frame.timestamp_ms for frame in frames]
        self.assertEqual(timestamps, sorted(timestamps))
        for actual, expected in zip(timestamps, (0.0, 200.0, 400.0, 600.0)):
            self.assertAlmostEqual(actual, expected, delta=2.0)

    def test_video_pipeline_writes_ordered_video_jsonl_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.avi"
            create_video(source, frame_count=6, fps=6.0)
            engine = StubVideoEngine()

            output = infer_video(
                engine,
                source,
                output_directory=root / "outputs",
                frame_skip=1,
                codec="MJPG",
                video_extension=".avi",
            )
            summary = output.summary
            records = [
                json.loads(line)
                for line in Path(summary.detections_file).read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            summary_record = json.loads(
                Path(summary.summary_file).read_text(encoding="utf-8")
            )
            capture = cv2.VideoCapture(summary.annotated_media)
            output_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            output_fps = float(capture.get(cv2.CAP_PROP_FPS))
            capture.release()

        self.assertEqual(summary.status, "complete")
        self.assertEqual(summary.frames_read, 6)
        self.assertEqual(summary.frames_processed, 3)
        self.assertEqual(summary.frames_written, 3)
        self.assertEqual([record["frame_index"] for record in records], [0, 2, 4])
        self.assertEqual([record["processed_index"] for record in records], [0, 1, 2])
        self.assertEqual(summary.detections_per_class, {"smoke": 0, "fire": 3})
        self.assertEqual(output_frames, 3)
        self.assertAlmostEqual(output_fps, 3.0, places=1)
        self.assertEqual(summary_record["source"]["declared_frames"], 6)
        self.assertTrue(output.to_dict()["success"])

    def test_graceful_stop_retains_valid_partial_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.avi"
            create_video(source, frame_count=6)
            calls = 0

            def stop_requested() -> bool:
                nonlocal calls
                calls += 1
                return calls > 2

            output = infer_video(
                StubVideoEngine(with_detections=False),
                source,
                output_directory=root / "outputs",
                codec="MJPG",
                video_extension=".avi",
                stop_requested=stop_requested,
            )
            summary = output.summary
            lines = Path(summary.detections_file).read_text(
                encoding="utf-8"
            ).splitlines()
            capture = cv2.VideoCapture(summary.annotated_media)
            output_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()

        self.assertEqual(summary.status, "interrupted")
        self.assertEqual(summary.interruption_reason, "stop_requested")
        self.assertEqual(summary.frames_read, 3)
        self.assertEqual(summary.frames_processed, 2)
        self.assertEqual(summary.frames_written, 2)
        self.assertEqual(len(lines), 2)
        self.assertEqual(output_frames, 2)
        self.assertTrue(output.to_dict()["success"])

    def test_codec_open_failure_is_reported_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.avi"
            create_video(source, frame_count=1)
            closed_writer = ClosedWriter()

            with self.assertRaisesRegex(OutputMediaError, "Cannot open video writer"):
                infer_video(
                    StubVideoEngine(),
                    source,
                    output_directory=root / "outputs",
                    codec="BAD!",
                    video_extension=".avi",
                    writer_factory=lambda *_: closed_writer,
                )

            summary = json.loads(
                (root / "outputs" / "source_summary.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(closed_writer.released)
        self.assertEqual(summary["status"], "failed")
        self.assertIsNone(summary["annotated_media"])

    def test_video_outputs_use_deterministic_collision_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.avi"
            create_video(source, frame_count=2)
            output_dir = root / "outputs"

            first = infer_video(
                StubVideoEngine(),
                source,
                output_directory=output_dir,
                codec="MJPG",
                video_extension=".avi",
            )
            second = infer_video(
                StubVideoEngine(),
                source,
                output_directory=output_dir,
                codec="MJPG",
                video_extension=".avi",
            )

        self.assertTrue(first.summary.summary_file.endswith("source_summary.json"))
        self.assertTrue(second.summary.summary_file.endswith("source_2_summary.json"))
        self.assertNotEqual(first.summary.annotated_media, second.summary.annotated_media)

    def test_zero_frame_video_fails_with_summary_and_releases_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "empty.mp4"
            source.write_bytes(b"capture is injected")
            capture = EmptyCapture()

            with self.assertRaisesRegex(InputMediaError, "no decodable frames"):
                infer_video(
                    StubVideoEngine(),
                    source,
                    output_directory=root / "outputs",
                    save_media=False,
                    save_detections=False,
                    capture_factory=lambda path: capture,
                )
            summary = json.loads(
                (root / "outputs" / "empty_summary.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["frames_processed"], 0)
        self.assertTrue(capture.released)

    def test_capture_factory_exception_is_wrapped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.mp4"
            source.write_bytes(b"capture is injected")

            def fail_capture(path: str) -> object:
                raise RuntimeError("capture factory exploded")

            with self.assertRaisesRegex(InputMediaError, "initialize video capture"):
                VideoReader(source, capture_factory=fail_capture)

    def test_keyboard_interrupt_closes_partial_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.avi"
            create_video(source, frame_count=6)

            output = infer_video(
                InterruptingEngine(with_detections=False),
                source,
                output_directory=root / "outputs",
                codec="MJPG",
                video_extension=".avi",
            )
            summary = output.summary
            records = Path(summary.detections_file).read_text(
                encoding="utf-8"
            ).splitlines()
            capture = cv2.VideoCapture(summary.annotated_media)
            output_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()

        self.assertEqual(summary.status, "interrupted")
        self.assertEqual(summary.interruption_reason, "keyboard_interrupt")
        self.assertEqual(summary.frames_processed, 2)
        self.assertEqual(summary.frames_written, 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(output_frames, 2)

    def test_video_reader_rejects_undecodable_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "corrupt.mp4"
            source.write_bytes(b"not a video")
            with self.assertRaises(InputMediaError):
                VideoReader(source)


if __name__ == "__main__":
    unittest.main()
