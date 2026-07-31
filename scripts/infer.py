"""Run local PyroVision inference on an image, video, or webcam."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pyrovision.config import load_inference_config  # noqa: E402
from pyrovision.errors import ConfigurationError, PyroVisionError  # noqa: E402
from pyrovision.images import infer_image, infer_image_directory  # noqa: E402
from pyrovision.model import DetectorEngine  # noqa: E402
from pyrovision.sources import classify_media_path  # noqa: E402
from pyrovision.video import infer_video  # noqa: E402
from pyrovision.webcam import infer_webcam  # noqa: E402


LOGGER = logging.getLogger("pyrovision.cli")


def class_threshold(value: str) -> tuple[str, float]:
    """Parse a repeatable CLASS=CONFIDENCE CLI override."""
    class_name, separator, threshold_text = value.partition("=")
    if not separator or not class_name.strip():
        raise argparse.ArgumentTypeError("class thresholds must use CLASS=CONFIDENCE")
    try:
        threshold = float(threshold_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("class confidence must be numeric") from exc
    if not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("class confidence must be between 0 and 1")
    return class_name.strip(), threshold


def non_negative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


def positive_integer(value: str) -> int:
    parsed = non_negative_integer(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--source", type=Path, help="Input image or video path")
    source_group.add_argument(
        "--webcam",
        nargs="?",
        type=non_negative_integer,
        const=None,
        default=argparse.SUPPRESS,
        metavar="INDEX",
        help="Use a webcam; omit INDEX to use input.webcam_index",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "inference.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="WARNING",
        type=str.upper,
    )
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--confidence", type=float)
    parser.add_argument(
        "--class-threshold",
        type=class_threshold,
        action="append",
        default=[],
        metavar="CLASS=CONFIDENCE",
    )
    parser.add_argument("--iou", type=float)
    parser.add_argument("--frame-skip", type=non_negative_integer)
    parser.add_argument(
        "--max-frames",
        type=positive_integer,
        help="Stop a webcam run after this many processed frames",
    )
    parser.add_argument("--codec", help="Four-character OpenCV video codec")
    parser.add_argument(
        "--video-extension", choices=(".avi", ".mkv", ".mov", ".mp4")
    )
    parser.add_argument(
        "--save-media",
        "--record",
        dest="save_media",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save annotated image/video or record the webcam session",
    )
    parser.add_argument(
        "--save-detections", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--display", action=argparse.BooleanOptionalAction, default=None
    )
    return parser


def validate_mode_arguments(args: argparse.Namespace) -> bool:
    """Validate cross-argument mode rules and return webcam selection."""
    webcam_requested = hasattr(args, "webcam")
    if not webcam_requested and args.max_frames is not None:
        raise ConfigurationError("--max-frames can only be used with --webcam")
    return webcam_requested


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(name)s: %(message)s",
    )
    try:
        config_path = (
            args.config.resolve()
            if args.config.is_absolute()
            else (PROJECT_ROOT / args.config).resolve()
        )
        config = load_inference_config(config_path, project_root=PROJECT_ROOT)
        if args.device is not None:
            config = replace(config, device=args.device.strip().lower())
        model_config = config.model
        if args.confidence is not None:
            global_thresholds = {
                class_name: args.confidence
                for class_name in config.checkpoint.expected_classes
            }
            model_config = replace(
                model_config,
                confidence_threshold=args.confidence,
                class_thresholds=global_thresholds,
            )
        if args.class_threshold:
            class_thresholds = dict(model_config.class_thresholds)
            class_thresholds.update(dict(args.class_threshold))
            model_config = replace(
                model_config, class_thresholds=class_thresholds
            )
        if args.iou is not None:
            model_config = replace(model_config, iou_threshold=args.iou)
        output_config = config.output
        if args.output_dir is not None:
            output_path = (
                args.output_dir.resolve()
                if args.output_dir.is_absolute()
                else (PROJECT_ROOT / args.output_dir).resolve()
            )
            output_config = replace(output_config, directory=output_path)
        if args.save_media is not None:
            output_config = replace(output_config, save_media=args.save_media)
        if args.save_detections is not None:
            output_config = replace(
                output_config, save_detections=args.save_detections
            )
        if args.display is not None:
            output_config = replace(output_config, display=args.display)
        if args.codec is not None:
            output_config = replace(output_config, video_codec=args.codec)
        if args.video_extension is not None:
            output_config = replace(
                output_config, video_extension=args.video_extension
            )
        input_config = config.input
        if args.frame_skip is not None:
            input_config = replace(input_config, frame_skip=args.frame_skip)
        config = replace(
            config, model=model_config, output=output_config, input=input_config
        )

        webcam_requested = validate_mode_arguments(args)
        if webcam_requested:
            webcam_index = (
                config.input.webcam_index if args.webcam is None else args.webcam
            )
            engine = DetectorEngine.from_config(config)
            LOGGER.info("Starting webcam inference on index %d", webcam_index)
            output = infer_webcam(
                engine,
                webcam_index,
                output_directory=config.output.directory,
                frame_skip=config.input.frame_skip,
                record=config.output.save_media,
                save_detections=config.output.save_detections,
                display=config.output.display,
                codec=config.output.video_codec,
                video_extension=config.output.video_extension,
                max_frames=args.max_frames,
            )
            mode = "webcam"
        else:
            configured_source = args.source or config.input.source
            if configured_source is None:
                raise ValueError(
                    "An input is required through --source, input.source, or --webcam"
                )
            source = Path(configured_source)
            if not source.is_absolute():
                source = PROJECT_ROOT / source
            media_kind = classify_media_path(source)
            engine = DetectorEngine.from_config(config)
            if media_kind == "image":
                LOGGER.info("Starting image inference for %s", source)
                output = infer_image(
                    engine,
                    source,
                    output_directory=config.output.directory,
                    save_media=config.output.save_media,
                    save_detections=config.output.save_detections,
                )
            elif media_kind == "image_directory":
                LOGGER.info("Starting image-directory inference for %s", source)
                output = infer_image_directory(
                    engine,
                    source,
                    output_directory=config.output.directory,
                    save_media=config.output.save_media,
                    save_detections=config.output.save_detections,
                )
            else:
                LOGGER.info("Starting video inference for %s", source)
                output = infer_video(
                    engine,
                    source,
                    output_directory=config.output.directory,
                    frame_skip=config.input.frame_skip,
                    save_media=config.output.save_media,
                    save_detections=config.output.save_detections,
                    codec=config.output.video_codec,
                    video_extension=config.output.video_extension,
                )
            mode = media_kind
        print(json.dumps(output.to_dict(), indent=2))
        LOGGER.info("Inference completed successfully in %s mode", mode)
        if mode in {"video", "webcam"} and output.summary.status == "interrupted":
            return 130
        return 0
    except (PyroVisionError, OSError, ValueError) as exc:
        LOGGER.debug("Expected inference failure", exc_info=True)
        print(
            json.dumps(
                {
                    "success": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        print(
            json.dumps(
                {
                    "success": False,
                    "error_type": "KeyboardInterrupt",
                    "error": "Inference interrupted by user",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        LOGGER.exception("Unexpected inference failure")
        print(
            json.dumps(
                {
                    "success": False,
                    "error_type": type(exc).__name__,
                    "error": "Unexpected inference failure; rerun with "
                    "--log-level DEBUG for details",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
