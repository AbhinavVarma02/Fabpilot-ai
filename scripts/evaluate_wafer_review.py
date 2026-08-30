"""Reproducible audit of the deployed WM-811K wafer selective-prediction rule.

This script regenerates the exact selective-prediction metrics reported for the
deployed FabPilot AI wafer review logic. It reuses the *deployed* wafer
preprocessing / CNN inference path (``wafer_defect_classification_node``) and the
*deployed* review thresholds (imported from ``src.agent.nodes``), so the audit
cannot drift from the running Streamlit app.

Review rule (unchanged, imported from the deployed code):

    accept only if softmax confidence >= WAFER_REVIEW_CONFIDENCE_THRESHOLD (0.40)
    AND top-2 margin >= WAFER_REVIEW_MARGIN_THRESHOLD (0.10)
    otherwise route the wafer to human review.

Accuracy is measured against WM-811K benchmark ground-truth labels. Inference is
deterministic (CPU, eval mode), so rerunning reproduces the same counts.

Run:

    python scripts/evaluate_wafer_review.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the DEPLOYED thresholds and the DEPLOYED wafer inference node so this
# audit is guaranteed to match the app. Nothing here re-implements the model
# logic, preprocessing, or thresholds.
from src.agent.nodes import (  # noqa: E402
    WAFER_REVIEW_CONFIDENCE_THRESHOLD,
    WAFER_REVIEW_MARGIN_THRESHOLD,
    wafer_defect_classification_node,
)
from src.data.demo_data import wafer_demo_sample_path  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "reports"
JSON_REPORT = REPORTS_DIR / "wm811k_review_audit.json"
MD_REPORT = REPORTS_DIR / "wm811k_review_audit.md"
PER_CLASS_CSV = REPORTS_DIR / "wm811k_review_per_class.csv"


def load_demo_wafers() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the 180 public demo wafer maps exactly as the dashboard does."""

    path = wafer_demo_sample_path()
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], data["class_names"].astype(str).tolist()


def _predict_records(
    X: np.ndarray, y: np.ndarray, class_names: list[str]
) -> list[dict[str, Any]]:
    """Score every demo wafer through the deployed inference node."""

    records: list[dict[str, Any]] = []
    for i in range(len(X)):
        # Feed the wafer through the exact deployed node (same preprocessing,
        # CNN load, softmax, and top-2 margin the Streamlit app uses).
        state = wafer_defect_classification_node({"wafer_image": X[i, 0]})
        pred_class = state["defect_class"]
        confidence = float(state["defect_confidence"])
        top2_margin = float(state["defect_top2_margin"])
        true_class = class_names[int(y[i])]

        # Apply the deployed review rule using the deployed threshold constants.
        accepted = (
            confidence >= WAFER_REVIEW_CONFIDENCE_THRESHOLD
            and top2_margin >= WAFER_REVIEW_MARGIN_THRESHOLD
        )
        records.append(
            {
                "index": i,
                "true_class": true_class,
                "pred_class": pred_class,
                "confidence": confidence,
                "top2_margin": top2_margin,
                "correct": pred_class == true_class,
                "accepted": accepted,
            }
        )
    return records


def _safe_rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator) if denominator else 0.0


def _confusion_matrix(records: list[dict[str, Any]], class_names: list[str]) -> list[list[int]]:
    """Return a 9x9 confusion matrix (rows = ground truth, cols = predicted)."""

    idx = {name: i for i, name in enumerate(class_names)}
    matrix = [[0 for _ in class_names] for _ in class_names]
    for rec in records:
        matrix[idx[rec["true_class"]]][idx[rec["pred_class"]]] += 1
    return matrix


def _per_class(records: list[dict[str, Any]], class_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in class_names:
        group = [r for r in records if r["true_class"] == name]
        correct = sum(1 for r in group if r["correct"])
        rows.append(
            {
                "class": name,
                "n": len(group),
                "correct": correct,
                "accuracy": _safe_rate(correct, len(group)),
            }
        )
    return rows


def _margin_stats(records: list[dict[str, Any]]) -> dict[str, float]:
    margins = [r["top2_margin"] for r in records]
    accepted = [r["top2_margin"] for r in records if r["accepted"]]
    routed = [r["top2_margin"] for r in records if not r["accepted"]]
    return {
        "min": min(margins),
        "max": max(margins),
        "mean": statistics.mean(margins),
        "median": statistics.median(margins),
        "accepted_mean": statistics.mean(accepted) if accepted else 0.0,
        "routed_mean": statistics.mean(routed) if routed else 0.0,
    }


def evaluate_wafer_review() -> dict[str, Any]:
    """Run the full selective-prediction audit and return a structured result.

    This function is side-effect free (it does not write any files) so it can be
    reused by the regression test.
    """

    X, y, class_names = load_demo_wafers()
    records = _predict_records(X, y, class_names)

    total = len(records)
    overall_correct = sum(1 for r in records if r["correct"])
    accepted_records = [r for r in records if r["accepted"]]
    routed_records = [r for r in records if not r["accepted"]]
    accepted = len(accepted_records)
    routed = len(routed_records)
    accepted_correct = sum(1 for r in accepted_records if r["correct"])
    routed_correct = sum(1 for r in routed_records if r["correct"])

    counts = {
        "total": total,
        "overall_correct": overall_correct,
        "accepted": accepted,
        "accepted_correct": accepted_correct,
        "routed": routed,
        "routed_correct": routed_correct,
    }

    overall_accuracy = _safe_rate(overall_correct, total)
    accepted_accuracy = _safe_rate(accepted_correct, accepted)
    routed_accuracy = _safe_rate(routed_correct, routed)
    rates = {
        "overall_accuracy": overall_accuracy,
        "accepted_accuracy": accepted_accuracy,
        "routed_accuracy": routed_accuracy,
        "coverage": _safe_rate(accepted, total),
        # Selective risk = error rate on the auto-accepted subset.
        "selective_risk": 1.0 - accepted_accuracy if accepted else 0.0,
        "overall_error_rate": 1.0 - overall_accuracy if total else 0.0,
        "routed_error_rate": 1.0 - routed_accuracy if routed else 0.0,
    }

    return {
        "dataset": str(wafer_demo_sample_path().relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "model": "artifacts/wafer_cnn.pt",
        "inference_path": "src.agent.nodes.wafer_defect_classification_node",
        "class_names": class_names,
        "thresholds": {
            "confidence": WAFER_REVIEW_CONFIDENCE_THRESHOLD,
            "margin": WAFER_REVIEW_MARGIN_THRESHOLD,
        },
        "counts": counts,
        "rates": rates,
        "per_class": _per_class(records, class_names),
        "confusion_matrix": _confusion_matrix(records, class_names),
        "margin_stats": _margin_stats(records),
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(results: dict[str, Any]) -> str:
    counts = results["counts"]
    rates = results["rates"]
    thr = results["thresholds"]
    class_names = results["class_names"]

    lines: list[str] = []
    lines.append("# WM-811K Wafer Review Audit")
    lines.append("")
    lines.append(
        "_Reproducible selective-prediction audit of the deployed FabPilot AI wafer review rule._"
    )
    lines.append("")
    lines.append("Regenerate with:")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/evaluate_wafer_review.py")
    lines.append("```")
    lines.append("")

    # Evaluation setup
    lines.append("## Evaluation setup")
    lines.append("")
    lines.append(
        f"- **Data:** `{results['dataset']}` — {counts['total']} WM-811K demo wafer maps, "
        f"20 per class across {len(class_names)} classes "
        f"({', '.join(class_names)})."
    )
    lines.append(
        f"- **Model:** `{results['model']}` — the compact PyTorch wafer-map CNN baseline."
    )
    lines.append(
        f"- **Inference path:** the deployed `{results['inference_path']}` — identical wafer "
        "preprocessing, CNN load, softmax, and top-2 margin used by the Streamlit app."
    )
    lines.append(
        "- **Thresholds:** imported directly from `src/agent/nodes.py`, so this audit cannot "
        "drift from the deployed review rule."
    )
    lines.append(
        "- Inference is deterministic (CPU, eval mode); rerunning reproduces the same counts."
    )
    lines.append("")

    # Review rule
    lines.append("## Review rule")
    lines.append("")
    lines.append("A wafer prediction is **auto-accepted** only when both conditions hold:")
    lines.append("")
    lines.append(
        f"- softmax confidence ≥ {_pct(thr['confidence'])} "
        f"(`WAFER_REVIEW_CONFIDENCE_THRESHOLD = {thr['confidence']}`)"
    )
    lines.append(
        f"- top-2 class margin ≥ {_pct(thr['margin'])} "
        f"(`WAFER_REVIEW_MARGIN_THRESHOLD = {thr['margin']}`)"
    )
    lines.append("")
    lines.append(
        "Otherwise the wafer is **routed to human review**. Accuracy is measured against "
        "WM-811K benchmark ground-truth labels."
    )
    lines.append("")

    # Overall results
    lines.append("## Overall results")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Total wafers | {counts['total']} |")
    lines.append(
        f"| All-sample benchmark accuracy | {_pct(rates['overall_accuracy'])} "
        f"({counts['overall_correct']}/{counts['total']}) |"
    )
    lines.append(
        f"| Coverage (auto-accepted / total) | {_pct(rates['coverage'])} "
        f"({counts['accepted']}/{counts['total']}) |"
    )
    lines.append(
        f"| Selective risk (accepted-case error rate) | {_pct(rates['selective_risk'])} "
        f"({counts['accepted'] - counts['accepted_correct']}/{counts['accepted']}) |"
    )
    lines.append("")
    lines.append(
        f"The {_pct(rates['overall_accuracy'])} figure is the all-sample baseline: the accuracy "
        "if every wafer were auto-accepted with no review routing."
    )
    lines.append("")

    # Accepted vs routed
    lines.append("## Accepted vs. routed")
    lines.append("")
    lines.append("| Group | Count | Benchmark accuracy |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| Auto-accepted (high confidence & margin) | {counts['accepted']} | "
        f"{_pct(rates['accepted_accuracy'])} ({counts['accepted_correct']}/{counts['accepted']}) |"
    )
    lines.append(
        f"| Routed to human review (lower certainty) | {counts['routed']} | "
        f"{_pct(rates['routed_accuracy'])} ({counts['routed_correct']}/{counts['routed']}) |"
    )
    lines.append("")
    lines.append(
        f"The accepted subset achieved {_pct(rates['accepted_accuracy'])} benchmark accuracy — this "
        "is the accuracy on the auto-accepted wafers, **not** the overall accuracy of the CNN. "
        "Lower-certainty wafers were routed to human review rather than auto-accepted."
    )
    lines.append("")

    # Per-class
    lines.append("## Per-class benchmark accuracy")
    lines.append("")
    lines.append("| Class | N | Correct | Accuracy |")
    lines.append("| --- | --- | --- | --- |")
    for row in results["per_class"]:
        lines.append(
            f"| {row['class']} | {row['n']} | {row['correct']} | {_pct(row['accuracy'])} |"
        )
    lines.append("")

    # Confusion matrix
    lines.append("## Confusion matrix")
    lines.append("")
    lines.append("Rows = ground truth, columns = predicted class.")
    lines.append("")
    header = "| true \\\\ pred | " + " | ".join(class_names) + " |"
    lines.append(header)
    lines.append("| --- | " + " | ".join("---" for _ in class_names) + " |")
    matrix = results["confusion_matrix"]
    for name, row in zip(class_names, matrix):
        lines.append(f"| **{name}** | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")

    # Margin stats
    ms = results["margin_stats"]
    lines.append("## Top-2 margin statistics")
    lines.append("")
    lines.append("| Statistic | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Min margin | {_pct(ms['min'])} |")
    lines.append(f"| Max margin | {_pct(ms['max'])} |")
    lines.append(f"| Mean margin | {_pct(ms['mean'])} |")
    lines.append(f"| Median margin | {_pct(ms['median'])} |")
    lines.append(f"| Mean margin, accepted group | {_pct(ms['accepted_mean'])} |")
    lines.append(f"| Mean margin, routed group | {_pct(ms['routed_mean'])} |")
    lines.append("")

    # Interpretation
    lines.append("## Interpretation: the selective-prediction trade-off")
    lines.append("")
    lines.append(
        f"The review rule trades coverage for reliability. By auto-accepting only wafers where the "
        f"CNN is both confident (softmax ≥ {_pct(thr['confidence'])}) and decisive "
        f"(top-2 margin ≥ {_pct(thr['margin'])}), the accepted subset reached "
        f"{_pct(rates['accepted_accuracy'])} benchmark accuracy on {counts['accepted']} of "
        f"{counts['total']} maps ({_pct(rates['coverage'])} coverage), versus a "
        f"{_pct(rates['overall_accuracy'])} all-sample baseline. The {counts['routed']} "
        f"lower-certainty maps — where accuracy would have been only "
        f"{_pct(rates['routed_accuracy'])} — are routed to human review rather than auto-accepted. "
        "This concentrates automation on the cases the model handles well and escalates the rest."
    )
    lines.append("")

    # Scope / wording notes
    lines.append("## Scope notes")
    lines.append("")
    lines.append(
        f"- The accepted subset achieved {_pct(rates['accepted_accuracy'])} benchmark accuracy; the "
        f"CNN is not {_pct(rates['accepted_accuracy'])} accurate overall (its all-sample accuracy "
        f"is {_pct(rates['overall_accuracy'])})."
    )
    lines.append("- Lower-certainty predictions are routed to human review, not discarded.")
    lines.append(
        "- Thresholds reflect observed demo behavior on a compact balanced public benchmark "
        "sample, not a production-calibrated operating point."
    )
    lines.append(
        "- This is wafer-map defect *pattern* classification with human-in-the-loop review; it "
        "does not claim confirmed physical root-cause detection."
    )
    lines.append("")

    return "\n".join(lines)


def write_reports(results: dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    JSON_REPORT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    MD_REPORT.write_text(render_markdown(results), encoding="utf-8")

    csv_lines = ["class,n,correct,accuracy"]
    for row in results["per_class"]:
        csv_lines.append(
            f"{row['class']},{row['n']},{row['correct']},{row['accuracy']:.6f}"
        )
    PER_CLASS_CSV.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")


def print_summary(results: dict[str, Any]) -> None:
    counts = results["counts"]
    rates = results["rates"]
    thr = results["thresholds"]

    print("=" * 64)
    print("WM-811K deployed wafer review audit")
    print("=" * 64)
    print(f"Data:  {results['dataset']}")
    print(f"Model: {results['model']}")
    print(
        f"Rule:  accept if softmax confidence >= {thr['confidence']:.2f} "
        f"AND top-2 margin >= {thr['margin']:.2f}; else route to human review"
    )
    print("-" * 64)
    print(
        f"Total wafers:              {counts['total']:>4}"
    )
    print(
        f"All-sample accuracy:       {_pct(rates['overall_accuracy']):>6} "
        f"({counts['overall_correct']}/{counts['total']})"
    )
    print(
        f"Auto-accepted:             {counts['accepted']:>4}   "
        f"accuracy {_pct(rates['accepted_accuracy'])} "
        f"({counts['accepted_correct']}/{counts['accepted']})"
    )
    print(
        f"Routed to human review:    {counts['routed']:>4}   "
        f"accuracy {_pct(rates['routed_accuracy'])} "
        f"({counts['routed_correct']}/{counts['routed']})"
    )
    print(
        f"Coverage:                  {_pct(rates['coverage']):>6}   "
        f"selective risk {_pct(rates['selective_risk'])}"
    )
    print("-" * 64)
    print("Per-class benchmark accuracy:")
    for row in results["per_class"]:
        print(
            f"  {row['class']:<10} {row['correct']:>2}/{row['n']:<2}  {_pct(row['accuracy'])}"
        )
    print("=" * 64)


def main() -> None:
    results = evaluate_wafer_review()
    print_summary(results)
    write_reports(results)
    print(
        f"Wrote:\n  {MD_REPORT.relative_to(PROJECT_ROOT)}"
        f"\n  {JSON_REPORT.relative_to(PROJECT_ROOT)}"
        f"\n  {PER_CLASS_CSV.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
