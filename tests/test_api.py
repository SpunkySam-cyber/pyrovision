from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pyrovision.api.app import create_app  # noqa: E402
from pyrovision.api.config import BackendConfig, load_backend_config  # noqa: E402
from pyrovision.config import (  # noqa: E402
    CheckpointConfig,
    InferenceConfig,
    InputConfig,
    ModelConfig,
    OutputConfig,
)
from pyrovision.errors import ConfigurationError, InferenceError  # noqa: E402
from pyrovision.types import BoundingBox, Detection, FrameResult  # noqa: E402


def make_backend_config(root: Path, max_upload_size: int = 1024 * 1024) -> BackendConfig:
    inference = InferenceConfig(
        schema_version=1,
        checkpoint=CheckpointConfig(verify_sha256=False),
        device="cpu",
        model=ModelConfig(half=False),
        input=InputConfig(),
        output=OutputConfig(video_codec="MJPG", video_extension=".avi"),
        project_root=root,
    )
    return BackendConfig(
        inference=inference,
        cors_origins=("http://localhost:3000",),
        max_upload_size_bytes=max_upload_size,
        output_directory=root / "api-outputs",
        temporary_directory=root / "api-temp",
    )


class StubApiEngine:
    checkpoint = SimpleNamespace(sha256="d" * 64)
    device = SimpleNamespace(value="cpu")
    class_names = ("smoke", "fire")

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def predict_frame(
        self,
        frame: np.ndarray,
        *,
        source: str,
        frame_index: int,
        timestamp_ms: float,
    ) -> FrameResult:
        self.calls += 1
        if self.fail:
            raise InferenceError("stub inference failed")
        detections = (
            Detection(
                class_id=1,
                class_name="fire",
                confidence=0.875,
                bbox=BoundingBox(2.0, 3.0, 20.0, 22.0),
            ),
        )
        return FrameResult(
            source=source,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            width=frame.shape[1],
            height=frame.shape[0],
            detections=detections,
        )


def image_bytes() -> bytes:
    success, encoded = cv2.imencode(
        ".jpg",
        np.zeros((32, 48, 3), dtype=np.uint8),
    )
    if not success:
        raise RuntimeError("Test environment cannot encode JPEG")
    return encoded.tobytes()


def video_bytes(root: Path, frame_count: int = 3) -> bytes:
    path = root / "input.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (48, 32),
    )
    if not writer.isOpened():
        raise RuntimeError("Test environment cannot encode MJPG/AVI")
    try:
        for index in range(frame_count):
            writer.write(np.full((32, 48, 3), index * 30, dtype=np.uint8))
    finally:
        writer.release()
    return path.read_bytes()


class ApiTest(unittest.TestCase):
    def test_health_and_model_are_initialized_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            engine = StubApiEngine()
            load_calls = 0

            def factory(config: InferenceConfig) -> StubApiEngine:
                nonlocal load_calls
                load_calls += 1
                self.assertEqual(config.device, "cpu")
                return engine

            app = create_app(make_backend_config(root), engine_factory=factory)
            with TestClient(app) as client:
                first = client.get("/health")
                second = client.get("/health")
                response = client.post(
                    "/predict/image",
                    files={"file": ("sample.jpg", image_bytes(), "image/jpeg")},
                )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(load_calls, 1)
        self.assertEqual(first.json()["checkpoint_identifier"], "d" * 12)
        self.assertTrue(first.json()["model_loaded"])

    def test_image_upload_returns_schema_and_published_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = make_backend_config(root)
            app = create_app(config, engine_factory=lambda _: StubApiEngine())
            with TestClient(app) as client:
                response = client.post(
                    "/predict/image",
                    files={"file": ("warehouse.jpg", image_bytes(), "image/jpeg")},
                )
                body = response.json()
                media = client.get(body["annotated_output"]["url"])
                detections = client.get(body["detections_output"]["url"])

            temporary_files = list(config.temporary_directory.iterdir())

        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["success"])
        self.assertEqual(body["original_filename"], "warehouse.jpg")
        self.assertEqual(body["detections"][0]["class"], "fire")
        self.assertEqual(body["detections"][0]["bbox"], [2.0, 3.0, 20.0, 22.0])
        self.assertEqual(body["processing"]["media_type"], "image")
        self.assertEqual(media.status_code, 200)
        self.assertEqual(detections.status_code, 200)
        self.assertEqual(temporary_files, [])

    def test_video_upload_returns_frames_summary_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app = create_app(
                make_backend_config(root),
                engine_factory=lambda _: StubApiEngine(),
            )
            payload = video_bytes(root)
            with TestClient(app) as client:
                response = client.post(
                    "/predict/video",
                    files={"file": ("camera.avi", payload, "video/x-msvideo")},
                )
                body = response.json()
                published = {
                    name: client.get(reference["url"])
                    for name, reference in body["output"].items()
                }

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["processed_frames"], 3)
        self.assertEqual(len(body["detections"]), 3)
        self.assertEqual(body["detections"][2]["frame_index"], 2)
        self.assertEqual(body["summary"]["detections_total"], 3)
        self.assertEqual(body["summary"]["status"], "complete")
        self.assertTrue(all(result.status_code == 200 for result in published.values()))

    def test_unsupported_missing_and_mismatched_uploads_are_structured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = create_app(
                make_backend_config(Path(temp_dir)),
                engine_factory=lambda _: StubApiEngine(),
            )
            with TestClient(app) as client:
                unsupported = client.post(
                    "/predict/image",
                    files={"file": ("sample.txt", b"text", "text/plain")},
                )
                mismatched = client.post(
                    "/predict/image",
                    files={"file": ("sample.jpg", image_bytes(), "text/plain")},
                )
                missing = client.post("/predict/image")

        self.assertEqual(unsupported.status_code, 415)
        self.assertEqual(
            unsupported.json()["error"]["code"],
            "unsupported_media_type",
        )
        self.assertEqual(mismatched.status_code, 415)
        self.assertEqual(missing.status_code, 422)
        self.assertEqual(missing.json()["error"]["code"], "validation_error")

    def test_corrupt_and_empty_media_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = create_app(
                make_backend_config(Path(temp_dir)),
                engine_factory=lambda _: StubApiEngine(),
            )
            with TestClient(app) as client:
                corrupt_image = client.post(
                    "/predict/image",
                    files={"file": ("bad.jpg", b"not jpeg", "image/jpeg")},
                )
                corrupt_video = client.post(
                    "/predict/video",
                    files={"file": ("bad.mp4", b"not mp4", "video/mp4")},
                )
                empty = client.post(
                    "/predict/image",
                    files={"file": ("empty.jpg", b"", "image/jpeg")},
                )

        self.assertEqual(corrupt_image.status_code, 422)
        self.assertEqual(corrupt_video.status_code, 422)
        self.assertEqual(corrupt_image.json()["error"]["code"], "invalid_media")
        self.assertEqual(corrupt_video.json()["error"]["code"], "invalid_media")
        self.assertEqual(empty.json()["error"]["code"], "empty_upload")

    def test_oversized_upload_is_removed_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = make_backend_config(root, max_upload_size=8)
            app = create_app(config, engine_factory=lambda _: StubApiEngine())
            with TestClient(app) as client:
                response = client.post(
                    "/predict/image",
                    files={"file": ("large.jpg", b"x" * 32, "image/jpeg")},
                )
            temporary_files = list(config.temporary_directory.iterdir())

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "upload_too_large")
        self.assertEqual(temporary_files, [])

    def test_inference_failure_is_structured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = create_app(
                make_backend_config(Path(temp_dir)),
                engine_factory=lambda _: StubApiEngine(fail=True),
            )
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/predict/image",
                    files={"file": ("sample.jpg", image_bytes(), "image/jpeg")},
                )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "inference_failed")

    def test_startup_failure_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = create_app(
                make_backend_config(Path(temp_dir)),
                engine_factory=lambda _: (_ for _ in ()).throw(
                    InferenceError("checkpoint load failed")
                ),
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "PyroVision model startup failed: checkpoint load failed",
            ):
                with TestClient(app):
                    pass

    def test_openapi_contract_and_webcam_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = create_app(
                make_backend_config(Path(temp_dir)),
                engine_factory=lambda _: StubApiEngine(),
            )
            with TestClient(app) as client:
                schema = client.get("/openapi.json").json()
                swagger = client.get("/docs")
                webcam = client.post("/predict/webcam")

        self.assertIn("/health", schema["paths"])
        self.assertIn("/predict/image", schema["paths"])
        self.assertIn("/predict/video", schema["paths"])
        self.assertNotIn("/predict/webcam", schema["paths"])
        self.assertIn("ImagePredictionResponse", schema["components"]["schemas"])
        self.assertEqual(swagger.status_code, 200)
        self.assertEqual(webcam.status_code, 404)
        self.assertEqual(webcam.json()["error"]["code"], "not_found")

    def test_environment_configuration_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir()
            source = PROJECT_ROOT / "configs" / "inference.yaml"
            (root / "configs" / "inference.yaml").write_text(
                source.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            config = load_backend_config(
                {
                    "PYROVISION_API_PORT": "9000",
                    "PYROVISION_API_CORS_ORIGINS": "https://one.test,https://two.test",
                    "PYROVISION_API_MAX_UPLOAD_SIZE_BYTES": "4096",
                },
                project_root=root,
            )
            with self.assertRaises(ConfigurationError):
                load_backend_config(
                    {"PYROVISION_API_PORT": "70000"},
                    project_root=root,
                )

        self.assertEqual(config.port, 9000)
        self.assertEqual(config.max_upload_size_bytes, 4096)
        self.assertEqual(
            config.cors_origins,
            ("https://one.test", "https://two.test"),
        )


if __name__ == "__main__":
    unittest.main()
