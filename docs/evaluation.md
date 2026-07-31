# Step 3 — held-out test evaluation

Status: **complete**

The selected epoch-54 YOLO11s checkpoint was evaluated once on the untouched
D-Fire test split. No training, threshold tuning, or checkpoint selection used
these test results.

## Evaluation input

- Checkpoint: `best.pt` from epoch 54
- SHA-256: `21812ec7917bda5ad004fc085ba6a9d8ee1b375c95db2efe754463fc430d28c3`
- Test images: 2,153
- Negative images: 984
- Smoke instances: 1,208
- Fire instances: 1,502
- Image size: 640 px

## Test metrics

| Scope | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| Overall | 0.7657 | 0.6992 | 0.7642 | 0.4526 |
| Smoke | 0.8189 | 0.7806 | 0.8368 | 0.5378 |
| Fire | 0.7125 | 0.6178 | 0.6916 | 0.3675 |

Inference time was 4.30 ms/image during batched test evaluation. This is a
different workload from the single-frame and end-to-end Step 6 benchmark and
must not be compared as if batch size and pipeline scope were identical.

## Generalization from validation to test

| Metric | Validation | Test | Absolute change |
| --- | ---: | ---: | ---: |
| Precision | 0.7905 | 0.7657 | -0.0248 |
| Recall | 0.7118 | 0.6992 | -0.0126 |
| mAP50 | 0.7893 | 0.7642 | -0.0252 |
| mAP50–95 | 0.4669 | 0.4526 | -0.0143 |

The modest decline indicates reasonable split generalization. Smoke remains
substantially stronger than fire, especially for localization across IoU
thresholds. Fire recall of 0.6178 is the main accuracy limitation.

## Confusion matrix and curves

The raw confusion matrix stored in the metrics record is arranged as predicted
rows by true columns, including background:

```text
                 true smoke  true fire  background
pred smoke              996          8         264
pred fire                12       1058         559
background              200        436           0
```

The generated normalized confusion matrix, PR curve, F1-confidence curve,
precision-confidence curve, and recall-confidence curve are stored under the
local ignored run directory. The PR curve reports AP50 0.837 for smoke and
0.692 for fire. The combined F1 curve peaks around 0.73 at confidence 0.324;
this is a useful initial threshold for Step 4 but still requires real-time
video testing.

## Visual sanity check

Four deterministic test images were inferred at confidence 0.25:

- Negative: no detection.
- Smoke-only: one smoke detection at 0.729 confidence.
- Fire-only: one fire detection at 0.730 confidence.
- Smoke and fire: smoke at 0.867 and fire at 0.631 confidence.

Manual inspection found the boxes visually plausible, including a distant
smoke plume and a small night-time flame. This four-image check demonstrates
the expected inference behavior but does not replace aggregate evaluation.

## Step 3 decision

The evaluation pipeline, metrics, plots, and held-out inference sanity check
all passed. The model is suitable to advance to Step 4 local real-time
inference as a project baseline. It is not yet suitable for safety-critical
deployment: fire recall and background false detections require further model,
data, and operating-threshold work.
