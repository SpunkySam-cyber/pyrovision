"""Run local PyroVision inference on a still image."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pyrovision.config import load_inference_config  # noqa: E402
from pyrovision.errors import PyroVisionError  # noqa: E402
from pyrovision.images import infer_image  # noqa: E402
from pyrovision.model import DetectorEngine  # noqa: E402


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Input image path")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "inference.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
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
    parser.add_argument(
        "--save-media", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--save-detections", action=argparse.BooleanOptionalAction, default=None
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
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
        config = replace(config, model=model_config, output=output_config)

        configured_source = args.source or config.input.source
        if configured_source is None:
            raise ValueError("An image source is required through --source or input.source")
        source = Path(configured_source)
        if not source.is_absolute():
            source = PROJECT_ROOT / source

        engine = DetectorEngine.from_config(config)
        output = infer_image(
            engine,
            source,
            output_directory=config.output.directory,
            save_media=config.output.save_media,
            save_detections=config.output.save_detections,
        )
        print(json.dumps(output.to_dict(), indent=2))
        return 0
    except (PyroVisionError, OSError, ValueError) as exc:
        print(
            json.dumps({"success": False, "error": str(exc)}, indent=2),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
