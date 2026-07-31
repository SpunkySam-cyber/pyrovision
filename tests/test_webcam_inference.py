from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from infer import build_parser  # noqa: E402
from pyrovision.errors import InputMediaError, OutputMediaError  # noqa: E402
from pyrovision.outputs import LiveDisplay  # noqa: E402
from pyrovision.sources import WebcamReader  # noqa: E402
from pyrovision.types import BoundingBox, Detection, FrameResult  # noqa: E402
from pyrovision.webcam import infer_webcam  # noqa: E402


def make_frames(count: int, width: int = 96, height: int = 64) -> list[np.ndarray]:
    return [
        np.full((height, width, 3), index % 255, dtype=np.uint8)
        for index in range(count)
    ]


class FakeCapture:
    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        fps: float = 12.0,
        opened: bool = True,
    ) -> None:
        self.frames = list(frames)
        self.fps = fps
        self.opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FPS:
            return self.fps
        return 0.0

    def release(self) -> None:
        self.released = True


class IncrementingClock:
    def __init__(self, increment_seconds: float = 0.01) -> None:
        self.value = -increment_seconds
        self.increment_seconds = increment_seconds

    def __call__(self) -> float:
        self.value += self.increment_seconds
        return self.value


class StubWebcamEngine:
    checkpoint = SimpleNamespace(sha256="d" * 64)
    device = SimpleNamespace(value="cpu")
    class_names = ("smoke", "fire")

    def __init__(self, interrupt_after: int | None = None) -> None:
        self.calls: list[int] = []
        self.interrupt_after = interrupt_after

    def predict_frame(
        self,
        frame: np.ndarray,
        *,
        source: str,
        frame_index: int,
        timestamp_ms: float,
    ) -> FrameResult:
        if self.interrupt_after is not None and len(self.calls) >= self.interrupt_after:
            raise KeyboardInterrupt
        self.calls.append(frame_index)
        detections = []
        if frame_index % 2 == 0:
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


class FakeDisplay:
    def __init__(self, window_name: str, stop_after: int = 2) -> None:
        self.window_name = window_name
        self.stop_after = stop_after
        self.frames_shown = 0
        self.closed = False

    def show(self, frame: np.ndarray) -> bool:
        self.frames_shown += 1
        return self.frames_shown >= self.stop_after

    def close(self) -> None:
        self.closed = True


class WebcamInferenceTest(unittest.TestCase):
    def test_cli_parses_webcam_display_and_recording_options(self) -> None:
        configured = build_parser().parse_args(["--webcam"])
        explicit = build_parser().parse_args(
            [
                "--webcam",
                "2",
                "--display",
                "--no-record",
                "--max-frames",
                "25",
            ]
        )

        self.assertTrue(hasattr(configured, "webcam"))
        self.assertIsNone(configured.webcam)
        self.assertEqual(explicit.webcam, 2)
        self.assertTrue(explicit.display)
        self.assertFalse(explicit.save_media)
        self.assertEqual(explicit.max_frames, 25)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--source", "image.jpg", "--webcam", "0"])

    def test_live_display_stops_on_q_and_closes_its_window(self) -> None:
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        with (
            patch("pyrovision.outputs.cv2.imshow") as show,
            patch("pyrovision.outputs.cv2.waitKey", return_value=ord("q")),
            patch("pyrovision.outputs.cv2.destroyWindow") as destroy,
        ):
            display = LiveDisplay("Test Window")
            self.assertTrue(display.show(frame))
            display.close()

        show.assert_called_once_with("Test Window", frame)
        destroy.assert_called_once_with("Test Window")

    def test_reader_uses_actual_dimensions_fallback_fps_and_releases(self) -> None:
        capture = FakeCapture(make_frames(3, width=80, height=60), fps=0.0)
        clock = IncrementingClock(0.02)

        with WebcamReader(
            2,
            capture_factory=lambda index: capture,
            clock=clock,
        ) as reader:
            first = reader.read()
            second = reader.read()
            metadata = reader.metadata

        self.assertEqual(metadata.index, 2)
        self.assertEqual((metadata.width, metadata.height), (80, 60))
        self.assertEqual(metadata.fps, 30.0)
        self.assertEqual(metadata.fps_source, "fallback")
        self.assertEqual((first.frame_index, second.frame_index), (0, 1))
        self.assertLess(first.timestamp_ms, second.timestamp_ms)
        self.assertTrue(capture.released)

    def test_unavailable_or_non_reading_camera_fails_cleanly(self) -> None:
        unavailable = FakeCapture([], opened=False)
        with self.assertRaisesRegex(InputMediaError, "could not open webcam"):
            WebcamReader(4, capture_factory=lambda index: unavailable)
        self.assertTrue(unavailable.released)

        no_frames = FakeCapture([], opened=True)
        with self.assertRaisesRegex(InputMediaError, "did not return a valid frame"):
            WebcamReader(0, capture_factory=lambda index: no_frames)
        self.assertTrue(no_frames.released)

        def fail_capture(index: int) -> object:
            raise RuntimeError("capture factory exploded")

        with self.assertRaisesRegex(InputMediaError, "initialize webcam"):
            WebcamReader(0, capture_factory=fail_capture)

    def test_pipeline_records_video_jsonl_and_summary_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            capture = FakeCapture(make_frames(8), fps=12.0)
            output = infer_webcam(
                StubWebcamEngine(),
                1,
                output_directory=root,
                frame_skip=1,
                record=True,
                save_detections=True,
                codec="MJPG",
                video_extension=".avi",
                max_frames=3,
                capture_factory=lambda index: capture,
                clock=IncrementingClock(),
                run_name="ordered",
            )
            summary = output.summary
            records = [
                json.loads(line)
                for line in Path(summary.detections_file).read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            stored_summary = json.loads(
                Path(summary.summary_file).read_text(encoding="utf-8")
            )
            video = cv2.VideoCapture(summary.annotated_media)
            output_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            output_fps = float(video.get(cv2.CAP_PROP_FPS))
            video.release()

        self.assertEqual(summary.status, "complete")
        self.assertEqual(summary.termination_reason, "max_frames")
        self.assertEqual(summary.frames_read, 5)
        self.assertEqual(summary.frames_processed, 3)
        self.assertEqual(summary.frames_written, 3)
        self.assertEqual([record["frame_index"] for record in records], [0, 2, 4])
        self.assertEqual(summary.detections_per_class, {"smoke": 0, "fire": 3})
        self.assertEqual(output_frames, 3)
        self.assertAlmostEqual(output_fps, 6.0, places=1)
        self.assertEqual(stored_summary["source"]["kind"], "webcam")
        self.assertTrue(capture.released)

    def test_display_quit_closes_window_capture_and_optional_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            capture = FakeCapture(make_frames(5))
            display = FakeDisplay("unused")
            output = infer_webcam(
                StubWebcamEngine(),
                0,
                output_directory=Path(temp_dir),
                record=False,
                save_detections=False,
                display=True,
                capture_factory=lambda index: capture,
                display_factory=lambda name: display,
                clock=IncrementingClock(),
                run_name="display",
            )

        self.assertEqual(output.summary.status, "complete")
        self.assertEqual(output.summary.termination_reason, "display_quit")
        self.assertEqual(output.summary.frames_processed, 2)
        self.assertIsNone(output.summary.annotated_media)
        self.assertIsNone(output.summary.detections_file)
        self.assertTrue(display.closed)
        self.assertTrue(capture.released)

    def test_webcam_outputs_avoid_collisions_and_reject_path_like_run_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_capture = FakeCapture(make_frames(2))
            second_capture = FakeCapture(make_frames(2))
            first = infer_webcam(
                StubWebcamEngine(),
                0,
                output_directory=root,
                record=False,
                save_detections=False,
                max_frames=1,
                capture_factory=lambda index: first_capture,
                clock=IncrementingClock(),
                run_name="session",
            )
            second = infer_webcam(
                StubWebcamEngine(),
                0,
                output_directory=root,
                record=False,
                save_detections=False,
                max_frames=1,
                capture_factory=lambda index: second_capture,
                clock=IncrementingClock(),
                run_name="session",
            )
            with self.assertRaisesRegex(OutputMediaError, "must not contain a path"):
                infer_webcam(
                    StubWebcamEngine(),
                    0,
                    output_directory=root,
                    record=False,
                    save_detections=False,
                    max_frames=1,
                    capture_factory=lambda index: FakeCapture(make_frames(2)),
                    run_name="../escape",
                )

        self.assertTrue(first.summary.summary_file.endswith("session_summary.json"))
        self.assertTrue(second.summary.summary_file.endswith("session_2_summary.json"))

    def test_display_setup_failure_writes_summary_and_releases_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            capture = FakeCapture(make_frames(3))

            def fail_display(name: str) -> object:
                raise OutputMediaError("display setup failed")

            with self.assertRaisesRegex(OutputMediaError, "display setup failed"):
                infer_webcam(
                    StubWebcamEngine(),
                    0,
                    output_directory=root,
                    record=False,
                    save_detections=True,
                    display=True,
                    capture_factory=lambda index: capture,
                    display_factory=fail_display,
                    clock=IncrementingClock(),
                    run_name="display_failure",
                )
            summary = json.loads(
                (root / "display_failure_summary.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["frames_processed"], 0)
        self.assertTrue(capture.released)

    def test_keyboard_interrupt_retains_partial_outputs_and_releases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            capture = FakeCapture(make_frames(6))
            output = infer_webcam(
                StubWebcamEngine(interrupt_after=2),
                0,
                output_directory=root,
                record=False,
                save_detections=True,
                capture_factory=lambda index: capture,
                clock=IncrementingClock(),
                run_name="interrupt",
            )
            records = Path(output.summary.detections_file).read_text(
                encoding="utf-8"
            ).splitlines()

        self.assertEqual(output.summary.status, "interrupted")
        self.assertEqual(output.summary.termination_reason, "keyboard_interrupt")
        self.assertEqual(output.summary.frames_processed, 2)
        self.assertEqual(len(records), 2)
        self.assertTrue(capture.released)

    def test_read_failure_writes_failed_summary_and_closes_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            capture = FakeCapture(make_frames(2))
            with self.assertRaisesRegex(InputMediaError, "failed to return"):
                infer_webcam(
                    StubWebcamEngine(),
                    0,
                    output_directory=root,
                    record=False,
                    save_detections=True,
                    capture_factory=lambda index: capture,
                    clock=IncrementingClock(),
                    run_name="failure",
                )
            summary = json.loads(
                (root / "failure_summary.json").read_text(encoding="utf-8")
            )
            records = (root / "failure_detections.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["termination_reason"], "error")
        self.assertEqual(summary["frames_processed"], 2)
        self.assertEqual(len(records), 2)
        self.assertTrue(capture.released)

    def test_simulated_long_session_has_stable_counts_and_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            capture = FakeCapture(make_frames(301))
            engine = StubWebcamEngine()
            output = infer_webcam(
                engine,
                0,
                output_directory=Path(temp_dir),
                record=False,
                save_detections=False,
                max_frames=300,
                capture_factory=lambda index: capture,
                clock=IncrementingClock(0.001),
                run_name="long_session",
            )

        self.assertEqual(output.summary.frames_read, 300)
        self.assertEqual(output.summary.frames_processed, 300)
        self.assertEqual(len(engine.calls), 300)
        self.assertEqual(output.summary.termination_reason, "max_frames")
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
