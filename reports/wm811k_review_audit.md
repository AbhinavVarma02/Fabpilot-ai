# WM-811K Wafer Review Audit

_Reproducible selective-prediction audit of the deployed FabPilot AI wafer review rule._

Regenerate with:

```bash
python scripts/evaluate_wafer_review.py
```

## Evaluation setup

- **Data:** `data/demo/wm811k_demo_sample.npz` — 180 WM-811K demo wafer maps, 20 per class across 9 classes (Center, Donut, Edge-Loc, Edge-Ring, Local, Near-full, None, Random, Scratch).
- **Model:** `artifacts/wafer_cnn.pt` — the compact PyTorch wafer-map CNN baseline.
- **Inference path:** the deployed `src.agent.nodes.wafer_defect_classification_node` — identical wafer preprocessing, CNN load, softmax, and top-2 margin used by the Streamlit app.
- **Thresholds:** imported directly from `src/agent/nodes.py`, so this audit cannot drift from the deployed review rule.
- Inference is deterministic (CPU, eval mode); rerunning reproduces the same counts.

## Review rule

A wafer prediction is **auto-accepted** only when both conditions hold:

- softmax confidence ≥ 40.0% (`WAFER_REVIEW_CONFIDENCE_THRESHOLD = 0.4`)
- top-2 class margin ≥ 10.0% (`WAFER_REVIEW_MARGIN_THRESHOLD = 0.1`)

Otherwise the wafer is **routed to human review**. Accuracy is measured against WM-811K benchmark ground-truth labels.

## Overall results

| Metric | Value |
| --- | --- |
| Total wafers | 180 |
| All-sample benchmark accuracy | 57.8% (104/180) |
| Coverage (auto-accepted / total) | 47.2% (85/180) |
| Selective risk (accepted-case error rate) | 11.8% (10/85) |

The 57.8% figure is the all-sample baseline: the accuracy if every wafer were auto-accepted with no review routing.

## Accepted vs. routed

| Group | Count | Benchmark accuracy |
| --- | --- | --- |
| Auto-accepted (high confidence & margin) | 85 | 88.2% (75/85) |
| Routed to human review (lower certainty) | 95 | 30.5% (29/95) |

The accepted subset achieved 88.2% benchmark accuracy — this is the accuracy on the auto-accepted wafers, **not** the overall accuracy of the CNN. Lower-certainty wafers were routed to human review rather than auto-accepted.

## Per-class benchmark accuracy

| Class | N | Correct | Accuracy |
| --- | --- | --- | --- |
| Center | 20 | 12 | 60.0% |
| Donut | 20 | 15 | 75.0% |
| Edge-Loc | 20 | 9 | 45.0% |
| Edge-Ring | 20 | 15 | 75.0% |
| Local | 20 | 1 | 5.0% |
| Near-full | 20 | 18 | 90.0% |
| None | 20 | 0 | 0.0% |
| Random | 20 | 18 | 90.0% |
| Scratch | 20 | 16 | 80.0% |

## Confusion matrix

Rows = ground truth, columns = predicted class.

| true \\ pred | Center | Donut | Edge-Loc | Edge-Ring | Local | Near-full | None | Random | Scratch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Center** | 12 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| **Donut** | 4 | 15 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| **Edge-Loc** | 0 | 1 | 9 | 0 | 0 | 0 | 0 | 1 | 9 |
| **Edge-Ring** | 0 | 0 | 3 | 15 | 0 | 0 | 0 | 1 | 1 |
| **Local** | 5 | 1 | 3 | 0 | 1 | 0 | 0 | 0 | 10 |
| **Near-full** | 0 | 0 | 0 | 0 | 0 | 18 | 0 | 2 | 0 |
| **None** | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 18 |
| **Random** | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 18 | 0 |
| **Scratch** | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 16 |

## Top-2 margin statistics

| Statistic | Value |
| --- | --- |
| Min margin | 0.4% |
| Max margin | 95.7% |
| Mean margin | 33.9% |
| Median margin | 16.3% |
| Mean margin, accepted group | 60.4% |
| Mean margin, routed group | 10.1% |

## Interpretation: the selective-prediction trade-off

The review rule trades coverage for reliability. By auto-accepting only wafers where the CNN is both confident (softmax ≥ 40.0%) and decisive (top-2 margin ≥ 10.0%), the accepted subset reached 88.2% benchmark accuracy on 85 of 180 maps (47.2% coverage), versus a 57.8% all-sample baseline. The 95 lower-certainty maps — where accuracy would have been only 30.5% — are routed to human review rather than auto-accepted. This concentrates automation on the cases the model handles well and escalates the rest.

## Scope notes

- The accepted subset achieved 88.2% benchmark accuracy; the CNN is not 88.2% accurate overall (its all-sample accuracy is 57.8%).
- Lower-certainty predictions are routed to human review, not discarded.
- Thresholds reflect observed demo behavior on a compact balanced public benchmark sample, not a production-calibrated operating point.
- This is wafer-map defect *pattern* classification with human-in-the-loop review; it does not claim confirmed physical root-cause detection.
