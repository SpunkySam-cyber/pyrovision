from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from prepare_dataset import prepare  # noqa: E402
from verify_dataset import verify  # noqa: E402


class DatasetToolsTest(unittest.TestCase):
    def test_prepare_and_verify_stratified_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            image_dir = source / "existing_split" / "images"
            label_dir = source / "existing_split" / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            categories = (
                "",
                "0 0.5 0.5 0.2 0.2\n",
                "1 0.5 0.5 0.2 0.2\n",
                "0 0.4 0.4 0.2 0.2\n1 0.6 0.6 0.2 0.2\n",
            )
            for index in range(40):
                stem = f"sample_{index:03d}"
                Image.new(
                    "RGB",
                    (32, 32),
                    color=((index * 5) % 256, (index * 11) % 256, (index * 17) % 256),
                ).save(image_dir / f"{stem}.png")
                (label_dir / f"{stem}.txt").write_text(
                    categories[index % len(categories)], encoding="utf-8"
                )

            output = root / "prepared"
            summary = prepare(
                source=source,
                output=output,
                class_names=("smoke", "fire"),
                ratios=(0.7, 0.2, 0.1),
                seed=42,
            )
            self.assertEqual(summary["total_images"], 40)

            stats, errors = verify(
                dataset=output,
                class_names=("smoke", "fire"),
                expected_ratios=(0.7, 0.2, 0.1),
                ratio_tolerance=0.01,
                distribution_tolerance=0.02,
                check_hashes=True,
            )
            self.assertEqual(errors, [])
            self.assertTrue(stats["valid"])
            self.assertEqual(stats["splits"]["train"]["images"], 28)
            self.assertEqual(stats["splits"]["val"]["images"], 8)
            self.assertEqual(stats["splits"]["test"]["images"], 4)


if __name__ == "__main__":
    unittest.main()
