"""Capture reproducible hardware, software, Git, and dataset context as JSON."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_context() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"installed": False, "cuda_available": False}

    cuda_available = torch.cuda.is_available()
    context: dict[str, object] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        context["device"] = torch.cuda.get_device_name(0)
        context["device_count"] = torch.cuda.device_count()
        context["vram_bytes"] = properties.total_memory
        context["compute_capability"] = list(torch.cuda.get_device_capability(0))
    return context


def nvidia_context() -> dict[str, str] | None:
    output = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None
    name, driver, total, free, capability = (
        value.strip() for value in output.splitlines()[0].split(",")
    )
    return {
        "name": name,
        "driver": driver,
        "memory_total_mib": total,
        "memory_free_mib": free,
        "compute_capability": capability,
    }


def collect_environment() -> dict[str, object]:
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(PROJECT_ROOT))
    preparation_summary = PROJECT_ROOT / "data" / "processed" / "dfire" / "preparation_summary.json"
    dataset: dict[str, object] | None = None
    if preparation_summary.is_file():
        source_summary = json.loads(preparation_summary.read_text(encoding="utf-8"))
        dataset = {
            "total_images": source_summary.get("total_images"),
            "ratios": source_summary.get("ratios"),
            "splits": source_summary.get("splits"),
            "annotation_cleanup": source_summary.get("annotation_cleanup"),
        }

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "packages": {
            name: package_version(name)
            for name in ("torch", "torchvision", "ultralytics", "numpy", "opencv-python", "Pillow")
        },
        "torch": torch_context(),
        "nvidia_smi": nvidia_context(),
        "system": {
            "ram_total_bytes": memory.total,
            "ram_available_bytes": memory.available,
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
        },
        "dataset": dataset,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit non-zero unless CUDA is available through PyTorch",
    )
    args = parser.parse_args()

    report = collect_environment()
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    if args.require_cuda and not report["torch"]["cuda_available"]:
        print("CUDA is not available through this Python environment.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
