"""Verify a prepared YOLO detection dataset and emit distribution statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Sequence

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def split_directories(dataset: Path, split: str) -> tuple[Path, Path]:
    """Resolve either images/train or train/images YOLO directory layouts."""
    layouts = (
        (dataset / "images" / split, dataset / "labels" / split),
        (dataset / split / "images", dataset / split / "labels"),
    )
    for image_dir, label_dir in layouts:
        if image_dir.is_dir() and label_dir.is_dir():
            return image_dir, label_dir
    return layouts[0]


def image_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_and_validate_label(
    label: Path, class_names: Sequence[str], errors: list[str]
) -> tuple[Counter[int], frozenset[int]]:
    boxes: Counter[int] = Counter()
    present: set[int] = set()
    try:
        lines = label.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        errors.append(f"Cannot read label {label}: {exc}")
        return boxes, frozenset()

    for line_number, raw_line in enumerate(lines, start=1):
        fields = raw_line.strip().split()
        if not fields:
            continue
        if len(fields) != 5:
            errors.append(f"{label}:{line_number}: expected 5 fields, got {len(fields)}")
            continue
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = (float(value) for value in fields[1:])
        except ValueError:
            errors.append(f"{label}:{line_number}: non-numeric YOLO value")
            continue
        values = (x_center, y_center, width, height)
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{label}:{line_number}: non-finite coordinate")
            continue
        if not 0 <= class_id < len(class_names):
            errors.append(f"{label}:{line_number}: invalid class ID {class_id}")
            continue
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            errors.append(f"{label}:{line_number}: box center is outside [0, 1]")
        if not (0 < width <= 1 and 0 < height <= 1):
            errors.append(f"{label}:{line_number}: box width/height is outside (0, 1]")
        tolerance = 1e-4
        if (
            x_center - width / 2 < -tolerance
            or x_center + width / 2 > 1 + tolerance
            or y_center - height / 2 < -tolerance
            or y_center + height / 2 > 1 + tolerance
        ):
            errors.append(f"{label}:{line_number}: box extends outside normalized image bounds")
        boxes[class_id] += 1
        present.add(class_id)
    return boxes, frozenset(present)


def verify(
    dataset: Path,
    class_names: Sequence[str],
    expected_ratios: Sequence[float],
    ratio_tolerance: float,
    distribution_tolerance: float,
    check_hashes: bool,
) -> tuple[dict[str, object], list[str]]:
    dataset = dataset.resolve()
    errors: list[str] = []
    stats: dict[str, object] = {"dataset": str(dataset), "class_names": list(class_names), "splits": {}}
    all_stems: dict[str, str] = {}
    all_hashes: dict[str, tuple[str, Path]] = {}
    total_images = 0
    category_totals: Counter[str] = Counter()

    for split in SPLITS:
        image_dir, label_dir = split_directories(dataset, split)
        if not image_dir.is_dir():
            errors.append(f"Missing image directory: {image_dir}")
            continue
        if not label_dir.is_dir():
            errors.append(f"Missing label directory: {label_dir}")
            continue

        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        )
        labels = {path.stem.casefold(): path for path in label_dir.glob("*.txt")}
        image_stems = {path.stem.casefold() for path in images}
        orphan_labels = sorted(set(labels) - image_stems)
        if orphan_labels:
            errors.append(f"{split}: {len(orphan_labels)} labels have no matching image")

        box_counts: Counter[int] = Counter()
        image_class_counts: Counter[int] = Counter()
        categories: Counter[str] = Counter()
        corrupt_images = 0
        for image in images:
            stem = image.stem.casefold()
            if stem in all_stems:
                errors.append(f"Split leakage by filename: {image.name} in {all_stems[stem]} and {split}")
            all_stems[stem] = split

            label = labels.get(stem)
            if label is None:
                errors.append(f"Missing label for image: {image}")
                class_ids = frozenset()
            else:
                boxes, class_ids = parse_and_validate_label(label, class_names, errors)
                box_counts.update(boxes)
                image_class_counts.update(class_ids)

            category = "negative" if not class_ids else "+".join(
                class_names[class_id] for class_id in sorted(class_ids)
            )
            categories[category] += 1
            category_totals[category] += 1

            try:
                with Image.open(image) as opened:
                    opened.verify()
                if image.suffix.casefold() in {".jpg", ".jpeg"}:
                    with image.open("rb") as handle:
                        handle.seek(-2, 2)
                        if handle.read() != b"\xff\xd9":
                            raise OSError("JPEG is missing the end-of-image marker")
            except (OSError, UnidentifiedImageError) as exc:
                corrupt_images += 1
                errors.append(f"Unreadable image {image}: {exc}")

            if check_hashes:
                digest = image_digest(image)
                previous = all_hashes.get(digest)
                if previous is not None and previous[0] != split:
                    errors.append(
                        f"Split leakage by content hash: {previous[1]} ({previous[0]}) and {image} ({split})"
                    )
                else:
                    all_hashes[digest] = (split, image)

        total_images += len(images)
        stats["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "negative_images": categories.get("negative", 0),
            "categories": dict(sorted(categories.items())),
            "images_per_class": {
                class_names[class_id]: image_class_counts[class_id]
                for class_id in range(len(class_names))
            },
            "boxes_per_class": {
                class_names[class_id]: box_counts[class_id]
                for class_id in range(len(class_names))
            },
            "corrupt_images": corrupt_images,
        }

    stats["total_images"] = total_images
    stats["total_categories"] = dict(sorted(category_totals.items()))

    if total_images == 0:
        errors.append("Dataset contains no images")
        return stats, errors

    for split, expected in zip(SPLITS, expected_ratios):
        split_stats = stats["splits"].get(split)
        if not split_stats:
            continue
        actual = split_stats["images"] / total_images
        split_stats["actual_ratio"] = round(actual, 6)
        if abs(actual - expected) > ratio_tolerance:
            errors.append(
                f"{split}: ratio {actual:.4f} differs from expected {expected:.4f} "
                f"by more than {ratio_tolerance:.4f}"
            )
        for class_name in class_names:
            if split_stats["images_per_class"][class_name] == 0:
                errors.append(f"{split}: no images contain class '{class_name}'")
        if split_stats["negative_images"] == 0:
            errors.append(f"{split}: no negative images")

    for category, total in category_totals.items():
        overall_share = total / total_images
        for split in SPLITS:
            split_stats = stats["splits"].get(split)
            if not split_stats or split_stats["images"] == 0:
                continue
            split_share = split_stats["categories"].get(category, 0) / split_stats["images"]
            if abs(split_share - overall_share) > distribution_tolerance:
                errors.append(
                    f"{split}: category '{category}' share {split_share:.4f} differs from "
                    f"overall {overall_share:.4f} by more than {distribution_tolerance:.4f}"
                )

    stats["valid"] = not errors
    stats["error_count"] = len(errors)
    return stats, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--class-names", nargs="+", default=("smoke", "fire"))
    parser.add_argument(
        "--expected-ratios", type=float, nargs=3, default=(0.7, 0.2, 0.1)
    )
    parser.add_argument("--ratio-tolerance", type=float, default=0.01)
    parser.add_argument("--distribution-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip duplicate-content leakage detection (faster but less strict)",
    )
    parser.add_argument(
        "--report", type=Path, help="Optional JSON report destination"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    stats, errors = verify(
        dataset=args.dataset,
        class_names=args.class_names,
        expected_ratios=args.expected_ratios,
        ratio_tolerance=args.ratio_tolerance,
        distribution_tolerance=args.distribution_tolerance,
        check_hashes=not args.skip_hash_check,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    if errors:
        print("\nValidation errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\nDataset verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
