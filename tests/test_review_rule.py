"""Regression test for the deployed WM-811K wafer selective-prediction rule.

This pins the reproducible audit numbers so the deployed review thresholds and
inference path cannot silently drift. It runs as a plain script:

    python tests/test_review_rule.py

and is also compatible with pytest (``pytest tests/test_review_rule.py``) if that
framework is ever added. pytest is intentionally not a project dependency, so the
test uses standard-library ``assert`` only.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for extra in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from src.agent.nodes import (  # noqa: E402
    WAFER_REVIEW_CONFIDENCE_THRESHOLD,
    WAFER_REVIEW_MARGIN_THRESHOLD,
)
from evaluate_wafer_review import evaluate_wafer_review  # noqa: E402

# Deployed constants and expected public audit outcome.
EXPECTED_CONFIDENCE_THRESHOLD = 0.40
EXPECTED_MARGIN_THRESHOLD = 0.10

EXPECTED_TOTAL = 180
EXPECTED_OVERALL_CORRECT = 104
EXPECTED_ACCEPTED = 85
EXPECTED_ACCEPTED_CORRECT = 75
EXPECTED_ROUTED = 95
EXPECTED_ROUTED_CORRECT = 29

# Percentage tolerance for the reported rates.
PCT_TOLERANCE = 0.005  # 0.5 percentage points


def test_thresholds_match_deployed_constants() -> None:
    assert WAFER_REVIEW_CONFIDENCE_THRESHOLD == EXPECTED_CONFIDENCE_THRESHOLD
    assert WAFER_REVIEW_MARGIN_THRESHOLD == EXPECTED_MARGIN_THRESHOLD


def test_audit_counts_match_expected() -> None:
    results = evaluate_wafer_review()

    # The evaluation must use the same deployed thresholds it is auditing.
    assert results["thresholds"]["confidence"] == EXPECTED_CONFIDENCE_THRESHOLD
    assert results["thresholds"]["margin"] == EXPECTED_MARGIN_THRESHOLD

    counts = results["counts"]
    assert counts["total"] == EXPECTED_TOTAL
    assert counts["overall_correct"] == EXPECTED_OVERALL_CORRECT
    assert counts["accepted"] == EXPECTED_ACCEPTED
    assert counts["accepted_correct"] == EXPECTED_ACCEPTED_CORRECT
    assert counts["routed"] == EXPECTED_ROUTED
    assert counts["routed_correct"] == EXPECTED_ROUTED_CORRECT

    # Accepted + routed must partition the full demo set.
    assert counts["accepted"] + counts["routed"] == EXPECTED_TOTAL


def test_audit_rates_within_tolerance() -> None:
    results = evaluate_wafer_review()
    rates = results["rates"]

    assert abs(rates["overall_accuracy"] - 104 / 180) <= PCT_TOLERANCE
    assert abs(rates["accepted_accuracy"] - 75 / 85) <= PCT_TOLERANCE
    assert abs(rates["routed_accuracy"] - 29 / 95) <= PCT_TOLERANCE
    assert abs(rates["coverage"] - 85 / 180) <= PCT_TOLERANCE
    # Selective risk is the accepted-case error rate.
    assert abs(rates["selective_risk"] - 10 / 85) <= PCT_TOLERANCE


def _run() -> int:
    tests = [
        test_thresholds_match_deployed_constants,
        test_audit_counts_match_expected,
        test_audit_rates_within_tolerance,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
        else:
            print(f"PASS  {test.__name__}")
    print("-" * 48)
    if failures:
        print(f"{failures} of {len(tests)} test(s) failed.")
    else:
        print(f"All {len(tests)} tests passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run())
