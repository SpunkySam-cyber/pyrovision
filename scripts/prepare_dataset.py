"""Prepare a deterministic, stratified YOLO dataset split.

The script accepts either an unsplit dataset or a dataset that already has
train/val/test folders. It finds every directory named ``images`` recursively,
pairs it with the corresponding ``labels`` directory, merges the records, and
creates a fresh 70/20/10 split stratified by image category:
negative, fire-only, smoke-only, or fire+smoke.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Record:
    image: Path
    label: Path | None
    output_stem: str
    class_ids: frozenset[int]
    category: str


def parse_label(label_path: Path | None, class_names: Sequence[str]) -> frozenset[int]:
    """Read and minimally validate one YOLO label file."""
    if label_path is None:
        return frozenset()

    class_ids: set[int] = set()
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"{label_path}:{line_number}: expected 5 YOLO fields, got {len(fields)}"
            )
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number}: non-numeric label value") from exc
        if not 0 <= class_id < len(class_names):
            raise ValueError(
                f"{label_path}:{line_number}: class {class_id} is outside "
                f"0..{len(class_names) - 1}"
            )
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"{label_path}:{line_number}: non-finite coordinate")
        class_ids.add(class_id)
    return frozenset(class_ids)


def category_for(class_ids: frozenset[int], class_names: Sequence[str]) -> str:
    if not class_ids:
        return "negative"
    return "+".join(class_names[class_id] for class_id in sorted(class_ids))


def paired_label_path(image: Path, source: Path) -> Path:
    relative = image.relative_to(source)
    parts = list(relative.parts)
    image_dir_indices = [
        index for index, part in enumerate(parts[:-1]) if part.casefold() == "images"
    ]
    if not image_dir_indices:
        raise ValueError(
            f"Image is not below a directory named 'images': {image}. "
            "Expected a YOLO layout such as images/foo.jpg and labels/foo.txt."
        )
    index = image_dir_indices[-1]
    parts[index] = "labels"
    return (source / Path(*parts)).with_suffix(".txt")


def discover_records(
    source: Path, class_names: Sequence[str], allow_missing_labels: bool
) -> list[Record]:
    images = sorted(
        (
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.as_posix().casefold(),
    )
    if not images:
        raise ValueError(f"No supported images found below {source}")

    records: list[Record] = []
    seen_stems: dict[str, Path] = {}
    for image in images:
        label = paired_label_path(image, source)
        if not label.is_file():
            if not allow_missing_labels:
                raise ValueError(
                    f"Missing label for {image}: expected {label}. "
                    "Use --allow-missing-labels only when missing labels are known negatives."
                )
            label = None

        stem_key = image.stem.casefold()
        if stem_key in seen_stems:
            raise ValueError(
                "Duplicate image stem would collide in the prepared dataset: "
                f"{seen_stems[stem_key]} and {image}"
            )
        seen_stems[stem_key] = image

        class_ids = parse_label(label, class_names)
        records.append(
            Record(
                image=image,
                label=label,
                output_stem=image.stem,
                class_ids=class_ids,
                category=category_for(class_ids, class_names),
            )
        )
    return records


def allocate_counts(size: int, ratios: Sequence[float]) -> list[int]:
    exact = [size * ratio for ratio in ratios]
    counts = [math.floor(value) for value in exact]
    remainder = size - sum(counts)
    order = sorted(
        range(len(ratios)), key=lambda index: (exact[index] - counts[index], -index), reverse=True
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def stratified_split(
    records: Iterable[Record], ratios: Sequence[float], seed: int
) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.category].append(record)

    rng = random.Random(seed)
    result = {split: [] for split in SPLITS}
    for category in sorted(grouped):
        group = grouped[category]
        rng.shuffle(group)
        counts = allocate_counts(len(group), ratios)
        offset = 0
        for split, count in zip(SPLITS, counts):
            result[split].extend(group[offset : offset + count])
            offset += count

    for split in SPLITS:
        rng.shuffle(result[split])
    return result


def safe_clear_output(output: Path, source: Path) -> None:
    output = output.resolve()
    source = source.resolve()
    dangerous = {Path(output.anchor), Path.cwd().resolve(), source}
    if output in dangerous or output == source.parent:
        raise ValueError(f"Refusing to clear unsafe output path: {output}")
    shutil.rmtree(output)


def write_split(
    split_records: dict[str, list[Record]], output: Path, source: Path
) -> None:
    manifest_rows: list[dict[str, str]] = []
    for split in SPLITS:
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for record in split_records[split]:
            image_name = f"{record.output_stem}{record.image.suffix.lower()}"
            label_name = f"{record.output_stem}.txt"
            shutil.copy2(record.image, image_dir / image_name)
            if record.label is None:
                (label_dir / label_name).write_text("", encoding="utf-8")
            else:
                shutil.copy2(record.label, label_dir / label_name)
            manifest_rows.append(
                {
                    "split": split,
                    "category": record.category,
                    "image": (Path("images") / split / image_name).as_posix(),
                    "label": (Path("labels") / split / label_name).as_posix(),
                    "source": record.image.relative_to(source).as_posix(),
                }
            )

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("split", "category", "image", "label", "source")
        )
        writer.writeheader()
        writer.writerows(manifest_rows)


def prepare(
    source: Path,
    output: Path,
    class_names: Sequence[str],
    ratios: Sequence[float],
    seed: int,
    allow_missing_labels: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")
    if source == output or source in output.parents:
        raise ValueError("Output must not be the source directory or a child of it")
    if len(ratios) != len(SPLITS) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("Split ratios must contain train/val/test values summing to 1")
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("Every split ratio must be positive")

    records = discover_records(source, class_names, allow_missing_labels)
    split_records = stratified_split(records, ratios, seed)

    if output.exists():
        if not overwrite:
            raise ValueError(f"Output already exists: {output}; use --overwrite to replace it")
        safe_clear_output(output, source)
    output.mkdir(parents=True)
    write_split(split_records, output, source)

    summary: dict[str, object] = {
        "source": str(source),
        "output": str(output),
        "seed": seed,
        "ratios": dict(zip(SPLITS, ratios)),
        "class_names": list(class_names),
        "total_images": len(records),
        "source_categories": dict(sorted(Counter(record.category for record in records).items())),
        "splits": {
            split: {
                "images": len(items),
                "categories": dict(sorted(Counter(item.category for item in items).items())),
            }
            for split, items in split_records.items()
        },
    }
    (output / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Extracted source dataset root")
    parser.add_argument("--output", type=Path, required=True, help="Prepared dataset root")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=("smoke", "fire"),
        help="Class names in source label ID order (default: smoke fire)",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.7, 0.2, 0.1),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Treat images without label files as negative samples",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = prepare(
            source=args.source,
            output=args.output,
            class_names=args.class_names,
            ratios=args.ratios,
            seed=args.seed,
            allow_missing_labels=args.allow_missing_labels,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Dataset preparation failed: {exc}") from exc
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

