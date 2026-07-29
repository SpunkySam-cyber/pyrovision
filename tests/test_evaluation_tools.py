from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate import select_sanity_samples  # noqa: E402


class EvaluationToolsTest(unittest.TestCase):
    def test_sanity_samples_cover_each_category_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.csv"
            rows = []
            for category in ("negative", "smoke", "fire", "smoke+fire"):
                for index in range(3):
                    image = Path("images") / "test" / f"{category}_{index}.jpg"
                    (root / image).parent.mkdir(parents=True, exist_ok=True)
                    (root / image).touch()
                    rows.append(
                        {
                            "split": "test",
                            "category": category,
                            "image": image.as_posix(),
                            "label": "",
                            "source": "",
                        }
                    )
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)

            first = select_sanity_samples(manifest, root, seed=42)
            second = select_sanity_samples(manifest, root, seed=42)

            self.assertEqual(first, second)
            self.assertEqual(
                [sample["category"] for sample in first],
                ["negative", "smoke", "fire", "smoke+fire"],
            )


if __name__ == "__main__":
    unittest.main()
