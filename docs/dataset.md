# D-Fire dataset record

## Source and license

- Dataset: D-Fire fire and smoke object-detection dataset
- Local raw source: `archive/` (Git-ignored and preserved unchanged)
- Local prepared data: `data/processed/dfire/` (Git-ignored)
- Kaggle mirror license declaration: CC0 1.0 Public Domain
- Class order: `0: smoke`, `1: fire`
- Negative samples use empty YOLO label files; `normal` is not an object class.

## Reproducible preparation

```powershell
python scripts/prepare_dataset.py `
  --source archive `
  --output data/processed/dfire `
  --ratios 0.7 0.2 0.1 `
  --seed 42
```

The source archive contained recoverable boundary-crossing boxes and a small
number of degenerate boxes. Preparation clips boxes to normalized image bounds
and drops zero-area boxes without modifying the raw source.

## Verified split

| Split | Images | Smoke images | Fire images | Negative images |
| --- | ---: | ---: | ---: | ---: |
| Train | 15,068 | 7,367 | 4,075 | 6,886 |
| Validation | 4,306 | 2,105 | 1,165 | 1,968 |
| Test | 2,153 | 1,053 | 582 | 984 |

- Valid smoke boxes: 11,854
- Valid fire boxes: 14,685
- Source JPEGs repaired in the processed copy because they lacked an
  end-of-image marker: 91
- Final dataset verification errors: 0
- Exact-content leakage across splits: none detected

The raw archive remains unchanged. The final strict verification loaded every
processed image successfully and confirmed that no JPEG was missing its
end-of-image marker.

The test split remained untouched until the one-time Step 3 evaluation.
Pre-training and post-training comparisons in Step 2 used only validation.

The D-Fire data keeps its source license and citation requirements. The MIT
license at the repository root applies to PyroVision project code and
documentation, not to third-party datasets, checkpoints, or dependencies.
