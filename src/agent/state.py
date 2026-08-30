"""LangGraph state definition for FabPilot AI."""

from __future__ import annotations

from typing import Any, TypedDict


class FabPilotState(TypedDict, total=False):
    sensor_data: Any
    wafer_image: Any
    yield_prediction: str | None
    yield_risk_score: float | None
    yield_confidence: float | None
    shap_features: list[dict[str, Any]] | None
    defect_class: str | None
    defect_confidence: float | None
    defect_top2_margin: float | None
    observed_signals: list[str]
    model_outputs: list[dict[str, Any]]
    hypotheses: list[str]
    recommended_checks: list[str]
    human_review_required: bool
    final_summary: str | None
