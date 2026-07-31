"""Run the versioned PyroVision CPU/CUDA benchmark suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pyrovision.benchmarking import (  # noqa: E402
    BenchmarkConfig,
    load_benchmark_config,
    merge_device_reports,
    run_benchmarks,
)
from pyrovision.errors import PyroVisionError  # noqa: E402
from pyrovision.outputs import write_json_atomic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark stabilized PyroVision inference on CPU and CUDA."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "benchmark.yaml",
        help="Versioned benchmark configuration YAML.",
    )
    parser.add_argument("--device", help=argparse.SUPPRESS)
    parser.add_argument("--report", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _run_isolated_devices(
    config: BenchmarkConfig,
    config_path: Path,
) -> dict[str, object]:
    reports: list[dict[str, object]] = []
    for device in config.devices:
        safe_device = device.replace(":", "-")
        partial_path = config.output_directory / f"partial-{safe_device}.json"
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--config",
            str(config_path.resolve()),
            "--device",
            device,
            "--report",
            str(partial_path),
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"{device} benchmark failed: {message}")
        reports.append(json.loads(partial_path.read_text(encoding="utf-8")))

    merged = merge_device_reports(reports, config, PROJECT_ROOT)
    write_json_atomic(config.report_path, merged)
    return merged


def main() -> int:
    args = parse_args()
    try:
        config = load_benchmark_config(args.config, PROJECT_ROOT)
        if args.device:
            report_path = args.report.resolve() if args.report else config.report_path
            config = replace(
                config,
                devices=(args.device,),
                report_path=report_path,
            )
            report = run_benchmarks(config, PROJECT_ROOT)
        elif len(config.devices) > 1:
            report = _run_isolated_devices(config, args.config)
        else:
            report = run_benchmarks(config, PROJECT_ROOT)
    except (PyroVisionError, OSError, ValueError, RuntimeError) as exc:
        print(
            json.dumps(
                {
                    "success": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "success": True,
                "report": str(config.report_path),
                "devices": sorted(report["devices"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
