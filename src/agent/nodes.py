"""LangGraph nodes for FabPilot model routing and summary assembly."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from src.agent.prompts import SAFE_INTERPRETATION_NOTE
from src.agent.state import FabPilotState
from src.data.demo_data import load_secom_background as load_demo_secom_background
from src.data.preprocess_wafer_maps import resize_wafer_map
from src.explainability.shap_explainer import explain_sample
from src.models.model_utils import load_joblib_artifact, project_root
from src.models.wafer_cnn import build_wafer_cnn

YIELD_REVIEW_CONFIDENCE_THRESHOLD = 0.65
WAFER_REVIEW_CONFIDENCE_THRESHOLD = 0.40
WAFER_REVIEW_MARGIN_THRESHOLD = 0.10


# The model bundle and SECOM background are read-only, so load them once per
# process and reuse them. This keeps repeated dashboard runs cheap and avoids
# re-reading the same artifact in both the yield and SHAP nodes.
@lru_cache(maxsize=1)
def _load_yield_bundle() -> dict[str, Any]:
    return load_joblib_artifact(project_root() / "artifacts" / "secom_yield_model.joblib")


@lru_cache(maxsize=1)
def _load_secom_background(sample_size: int = 200) -> pd.DataFrame:
    return load_demo_secom_background(sample_size=sample_size, random_state=42)


def _ensure_lists(state: FabPilotState) -> FabPilotState:
    state.setdefault("observed_signals", [])
    state.setdefault("model_outputs", [])
    state.setdefault("hypotheses", [])
    state.setdefault("recommended_checks", [])
    state.setdefault("human_review_required", False)
    return state


def _sensor_frame(sensor_data: Any) -> pd.DataFrame:
    if isinstance(sensor_data, pd.DataFrame):
        return sensor_data
    if isinstance(sensor_data, pd.Series):
        return sensor_data.to_frame().T
    if isinstance(sensor_data, dict):
        return pd.DataFrame([sensor_data])
    return pd.DataFrame(sensor_data)


def _format_probability_for_text(value: float) -> str:
    if value < 0.001:
        return "<0.1%"
    if value > 0.999:
        return ">99.9%"
    return f"{value * 100:.1f}%"


def _wafer_array_for_inference(wafer_image: Any) -> np.ndarray:
    """Return a 64x64 float32 wafer tensor in the CNN training value range."""

    array = np.asarray(wafer_image)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]

    if (
        array.ndim == 2
        and array.shape == (64, 64)
        and np.issubdtype(array.dtype, np.floating)
        and np.isfinite(array).all()
        and float(array.min()) >= 0.0
        and float(array.max()) <= 1.0
    ):
        return array.astype(np.float32, copy=False)

    return resize_wafer_map(array, output_size=64)


def yield_prediction_node(state: FabPilotState) -> FabPilotState:
    """Run the saved SECOM yield-risk model if sensor data is present."""

    state = _ensure_lists(state)
    if state.get("sensor_data") is None:
        state["observed_signals"].append("No SECOM sensor data was provided.")
        return state

    model_path = project_root() / "artifacts" / "secom_yield_model.joblib"
    if not model_path.exists():
        state["observed_signals"].append("SECOM yield model artifact is missing.")
        state["human_review_required"] = True
        return state

    bundle = _load_yield_bundle()
    sample = _sensor_frame(state["sensor_data"])
    probability = float(bundle["model"].predict_proba(sample)[:, 1][0])
    prediction = "fail" if probability >= 0.5 else "pass"
    confidence = probability if prediction == "fail" else 1.0 - probability

    state["yield_prediction"] = prediction
    state["yield_risk_score"] = probability
    state["yield_confidence"] = confidence
    state["observed_signals"].append(
        f"SECOM model predicted `{prediction}` with model-estimated fail risk {_format_probability_for_text(probability)}."
    )
    state["model_outputs"].append(
        {
            "tool": "yield_prediction_model",
            "prediction": prediction,
            "yield_risk_score": probability,
            "confidence": confidence,
        }
    )
    return state


def shap_explanation_node(state: FabPilotState) -> FabPilotState:
    """Generate SHAP feature contributions when yield prediction is available."""

    state = _ensure_lists(state)
    if state.get("sensor_data") is None or state.get("yield_prediction") is None:
        return state

    model_path = project_root() / "artifacts" / "secom_yield_model.joblib"
    if not model_path.exists():
        return state

    bundle = _load_yield_bundle()
    background = _load_secom_background()
    sample = _sensor_frame(state["sensor_data"])
    explanation = explain_sample(bundle, background=background, sample=sample, top_n=5)
    top_features = explanation["top_contributions"].to_dict(orient="records")

    state["shap_features"] = top_features
    feature_names = ", ".join(feature["feature"] for feature in top_features[:3])
    state["observed_signals"].append(
        f"SHAP identified {feature_names} as top contributing sensor features for this sample."
    )
    state["model_outputs"].append(
        {
            "tool": "shap_explanation",
            "top_features": top_features,
            "note": explanation["safe_interpretation_note"],
        }
    )
    return state


def wafer_defect_classification_node(state: FabPilotState) -> FabPilotState:
    """Run the saved wafer CNN if a wafer image is present."""

    state = _ensure_lists(state)
    if state.get("wafer_image") is None:
        state["observed_signals"].append("No wafer map image was provided.")
        return state

    import torch

    weights_path = project_root() / "artifacts" / "wafer_cnn.pt"
    if not weights_path.exists():
        state["observed_signals"].append("Wafer CNN artifact is missing.")
        state["human_review_required"] = True
        return state

    checkpoint = torch.load(weights_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    model = build_wafer_cnn(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wafer_array = _wafer_array_for_inference(state["wafer_image"])
    tensor = torch.from_numpy(wafer_array[None, None, :, :]).float()
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).numpy()[0]

    ranked_indices = np.argsort(probabilities)[::-1]
    class_index = int(ranked_indices[0])
    defect_class = class_names[class_index]
    confidence = float(probabilities[class_index])
    top2_margin = (
        float(probabilities[ranked_indices[0]] - probabilities[ranked_indices[1]])
        if len(ranked_indices) > 1
        else confidence
    )

    state["defect_class"] = defect_class
    state["defect_confidence"] = confidence
    state["defect_top2_margin"] = top2_margin
    state["observed_signals"].append(
        f"Wafer CNN predicted `{defect_class}` defect pattern with softmax confidence "
        f"{_format_probability_for_text(confidence)} and top-2 margin {_format_probability_for_text(top2_margin)}."
    )
    state["model_outputs"].append(
        {
            "tool": "wafer_defect_classifier",
            "defect_class": defect_class,
            "confidence": confidence,
            "top2_margin": top2_margin,
        }
    )
    return state


def confidence_check_node(state: FabPilotState) -> FabPilotState:
    """Flag cases where model confidence is low or evidence is incomplete."""

    state = _ensure_lists(state)
    # The SECOM model keeps the original conservative confidence check. The
    # wafer CNN uses an MVP rule measured against deployed demo behavior: review
    # low-confidence predictions and predictions where the top two classes are
    # too close to treat the result as a clear decision-support signal.
    low_confidence = False

    if (
        state.get("yield_confidence") is not None
        and state["yield_confidence"] < YIELD_REVIEW_CONFIDENCE_THRESHOLD
    ):
        low_confidence = True
        state["recommended_checks"].append("Review the SECOM yield prediction manually because confidence is below threshold.")

    if (
        state.get("defect_confidence") is not None
        and state["defect_confidence"] < WAFER_REVIEW_CONFIDENCE_THRESHOLD
    ):
        low_confidence = True
        state["recommended_checks"].append(
            "Review the wafer defect prediction manually because softmax confidence is below the MVP wafer threshold."
        )

    if (
        state.get("defect_top2_margin") is not None
        and state["defect_top2_margin"] < WAFER_REVIEW_MARGIN_THRESHOLD
    ):
        low_confidence = True
        state["recommended_checks"].append(
            "Review the wafer defect prediction manually because the top-2 class margin is small."
        )

    if state.get("sensor_data") is None or state.get("wafer_image") is None:
        low_confidence = True
        state["recommended_checks"].append("Treat the investigation as incomplete because one evidence source is missing.")

    state["human_review_required"] = bool(state.get("human_review_required") or low_confidence)
    return state


def human_review_node(state: FabPilotState) -> FabPilotState:
    """Add human-review guidance when needed."""

    state = _ensure_lists(state)
    if state.get("human_review_required"):
        state["recommended_checks"].append("Validate model outputs with process engineering judgment before acting.")
    return state


def _format_model_output_for_summary(output: dict[str, Any]) -> str:
    tool = output.get("tool", "model")
    if tool == "yield_prediction_model":
        return (
            f"Yield model predicted `{output.get('prediction')}`; estimated fail risk "
            f"{_format_probability_for_text(float(output.get('yield_risk_score', 0.0)))}; "
            f"decision confidence {_format_probability_for_text(float(output.get('confidence', 0.0)))}."
        )
    if tool == "shap_explanation":
        return (
            f"SHAP explainer returned {len(output.get('top_features', []))} top anonymized "
            "SECOM sensor-feature contributions."
        )
    if tool == "wafer_defect_classifier":
        return (
            f"Wafer CNN predicted `{output.get('defect_class')}`; softmax confidence "
            f"{_format_probability_for_text(float(output.get('confidence', 0.0)))}; "
            f"top-2 margin {_format_probability_for_text(float(output.get('top2_margin', 0.0)))}."
        )
    return str(tool)


def summary_generation_node(state: FabPilotState) -> FabPilotState:
    """Generate a deterministic structured investigation summary."""

    state = _ensure_lists(state)

    # Hypotheses are interpretations derived from the evidence above. They stay in
    # their own bucket and are never mixed back into observed signals.
    if state.get("yield_prediction") == "fail" and state.get("defect_class"):
        state["hypotheses"].append(
            "The combination of elevated SECOM yield risk and wafer-map pattern may be worth reviewing as an investigation hypothesis."
        )
    elif state.get("yield_prediction") == "fail":
        state["hypotheses"].append(
            "Elevated SECOM yield risk may warrant review of similar high-risk sensor profiles."
        )
    elif state.get("defect_class"):
        state["hypotheses"].append(
            "The predicted wafer-map pattern may warrant review against similar labeled wafer examples."
        )

    summary = [
        "Observed Signals:",
        *[f"- {item}" for item in state["observed_signals"]],
        "",
        "Model Outputs:",
        *[f"- {_format_model_output_for_summary(output)}" for output in state["model_outputs"]],
        "",
        "Investigation Hypotheses:",
        *[f"- {item}" for item in state["hypotheses"]],
        "",
        "Recommended Checks:",
        *[f"- {item}" for item in dict.fromkeys(state["recommended_checks"])],
        "",
        f"Human Review Required: {state.get('human_review_required', False)}",
        "",
        "Safety Note:",
        f"- {SAFE_INTERPRETATION_NOTE}",
    ]
    state["final_summary"] = "\n".join(summary)
    return state
