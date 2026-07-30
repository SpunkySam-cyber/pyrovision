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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pyrovision.checkpoints import ResolvedCheckpoint  # noqa: E402
from pyrovision.config import (  # noqa: E402
    CheckpointConfig,
    InferenceConfig,
    InputConfig,
    ModelConfig,
    OutputConfig,
)
from pyrovision.device import ResolvedDevice  # noqa: E402
from pyrovision.errors import InputMediaError  # noqa: E402
from pyrovision.images import infer_image  # noqa: E402
from pyrovision.model import DetectorEngine  # noqa: E402
from pyrovision.types import BoundingBox, Detection, FrameResult  # noqa: E402
from infer import build_parser  # noqa: E402


class FakeTensor:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> list[object]:
        return self.values


class FakeBoxes:
    def __init__(
        self,
        coordinates: list[list[float]],
        confidences: list[float],
        classes: list[float],
    ) -> None:
        self.xyxy = FakeTensor(coordinates)
        self.conf = FakeTensor(confidences)
        self.cls = FakeTensor(classes)


class FakeModel:
    names = {0: "smoke", 1: "fire"}

    def __init__(self, boxes: FakeBoxes | None) -> None:
        self.boxes = boxes
        self.arguments: dict[str, object] | None = None

    def predict(self, **arguments: object) -> list[SimpleNamespace]:
        self.arguments = arguments
        return [SimpleNamespace(boxes=self.boxes)]


def make_config(root: Path, model: ModelConfig | None = None) -> InferenceConfig:
    return InferenceConfig(
        schema_version=1,
        checkpoint=CheckpointConfig(
            path=root / "best.pt",
            verify_sha256=False,
            expected_classes=("smoke", "fire"),
        ),
        device="cpu",
        model=model or ModelConfig(),
        input=InputConfig(),
        output=OutputConfig(directory=root / "outputs"),
        project_root=root,
    )


def make_engine(root: Path, model: FakeModel, config: InferenceConfig) -> DetectorEngine:
    checkpoint = ResolvedCheckpoint(
        path=root / "best.pt",
        sha256="a" * 64,
        expected_sha256=None,
        epoch=54,
        source="test",
    )
    device = ResolvedDevice(
        requested="cpu", value="cpu", is_cuda=False, index=None, name="CPU"
    )
    return DetectorEngine(model, config, checkpoint, device)


class StubImageEngine:
    checkpoint = SimpleNamespace(sha256="b" * 64)
    device = SimpleNamespace(value="cpu")

    def predict_frame(
        self,
        frame: np.ndarray,
        *,
        source: str,
        frame_index: int,
        timestamp_ms: float,
    ) -> FrameResult:
        height, width = frame.shape[:2]
        return FrameResult(
            source=source,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            width=width,
            height=height,
            detections=[
                Detection(
                    class_id=1,
                    class_name="fire",
                    confidence=0.91234567,
                    bbox=BoundingBox(20.0, 15.0, 100.0, 75.0),
                )
            ],
        )


class ImageInferenceTest(unittest.TestCase):
    def test_cli_parses_global_and_class_threshold_overrides(self) -> None:
        arguments = build_parser().parse_args(
            [
                "--source",
                "sample.jpg",
                "--confidence",
                "0.4",
                "--class-threshold",
                "fire=0.3",
            ]
        )
        self.assertEqual(arguments.confidence, 0.4)
        self.assertEqual(arguments.class_threshold, [("fire", 0.3)])

    def test_engine_applies_class_thresholds_and_stable_sorting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = make_config(
                root,
                ModelConfig(
                    image_size=320,
                    confidence_threshold=0.35,
                    class_thresholds={"smoke": 0.5, "fire": 0.3},
                    iou_threshold=0.6,
                ),
            )
            model = FakeModel(
                FakeBoxes(
                    coordinates=[
                        [5.0, 5.0, 50.0, 50.0],
                        [10.0, 10.0, 60.0, 60.0],
                        [15.0, 15.0, 70.0, 70.0],
                        [-10.0, -20.0, 500.0, 500.0],
                    ],
                    confidences=[0.4, 0.44, 0.29, 0.6],
                    classes=[0.0, 1.0, 1.0, 0.0],
                )
            )
            engine = make_engine(root, model, config)
            frame = np.zeros((100, 120, 3), dtype=np.uint8)

            result = engine.predict_frame(frame, source="frame.jpg")

        self.assertEqual(engine.candidate_confidence, 0.3)
        self.assertEqual(
            [(item.class_name, item.confidence) for item in result.detections],
            [("smoke", 0.6), ("fire", 0.44)],
        )
        self.assertEqual(result.detections[0].bbox.to_list(), [0.0, 0.0, 120.0, 100.0])
        self.assertEqual(model.arguments["imgsz"], 320)
        self.assertEqual(model.arguments["conf"], 0.3)
        self.assertEqual(model.arguments["iou"], 0.6)
        self.assertEqual(model.arguments["device"], "cpu")
        self.assertFalse(model.arguments["half"])
        self.assertFalse(model.arguments["save"])

    def test_engine_returns_stable_empty_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = make_config(root)
            model = FakeModel(FakeBoxes([], [], []))
            engine = make_engine(root, model, config)
            result = engine.predict_frame(
                np.zeros((32, 48, 3), dtype=np.uint8),
                source="negative.jpg",
            )

        self.assertEqual(result.detections, ())
        self.assertEqual(result.to_dict()["detections"], [])

    def test_image_pipeline_writes_annotation_and_structured_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "sample.png"
            original = np.zeros((90, 140, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(source), original))
            output_dir = root / "outputs"

            output = infer_image(
                StubImageEngine(),
                source,
                output_directory=output_dir,
                save_media=True,
                save_detections=True,
            )

            self.assertTrue(output.annotated_media.is_file())
            self.assertTrue(output.detections_file.is_file())
            annotated = cv2.imread(str(output.annotated_media), cv2.IMREAD_COLOR)
            self.assertIsNotNone(annotated)
            self.assertTrue(np.any(annotated != original))
            record = json.loads(output.detections_file.read_text(encoding="utf-8"))

        self.assertTrue(record["success"])
        self.assertEqual(record["device"], "cpu")
        self.assertEqual(record["result"]["detections"][0]["class"], "fire")
        self.assertEqual(record["result"]["detections"][0]["confidence"], 0.912346)

    def test_image_pipeline_can_run_without_writing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "sample.jpg"
            self.assertTrue(
                cv2.imwrite(str(source), np.zeros((90, 140, 3), dtype=np.uint8))
            )
            output_dir = root / "not-created"

            output = infer_image(
                StubImageEngine(),
                source,
                output_directory=output_dir,
                save_media=False,
                save_detections=False,
            )

            self.assertIsNone(output.annotated_media)
            self.assertIsNone(output.detections_file)
            self.assertFalse(output_dir.exists())

    def test_image_pipeline_rejects_unsupported_and_corrupt_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            unsupported = root / "sample.txt"
            unsupported.write_text("not an image", encoding="utf-8")
            corrupt = root / "corrupt.jpg"
            corrupt.write_bytes(b"not a jpeg")

            with self.assertRaises(InputMediaError):
                infer_image(
                    StubImageEngine(),
                    unsupported,
                    output_directory=root / "outputs",
                )
            with self.assertRaises(InputMediaError):
                infer_image(
                    StubImageEngine(),
                    corrupt,
                    output_directory=root / "outputs",
                )


if __name__ == "__main__":
    unittest.main()
