"""SHAP helpers for SECOM yield-risk model explanations."""

from __future__ import annotations

from typing import Any

import pandas as pd


def transformed_feature_names(model_bundle: dict[str, Any]) -> list[str]:
    """Return feature names after the saved model pipeline preprocessing."""

    pipeline = model_bundle["model"]
    preprocess = pipeline.named_steps["preprocess"]
    dropper = preprocess.named_steps["drop_high_missing"]
    variance_filter = preprocess.named_steps["variance_filter"]

    kept_names = list(dropper.keep_columns_)
    support = variance_filter.get_support()
    return [name for name, keep in zip(kept_names, support) if keep]


def transform_features(model_bundle: dict[str, Any], features: pd.DataFrame) -> pd.DataFrame:
    """Apply the saved preprocessing pipeline and return a named DataFrame."""

    pipeline = model_bundle["model"]
    preprocess = pipeline.named_steps["preprocess"]
    transformed = preprocess.transform(features)
    return pd.DataFrame(transformed, columns=transformed_feature_names(model_bundle), index=features.index)


def _linear_shap_values(model_bundle: dict[str, Any], background: pd.DataFrame, sample: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute exact linear SHAP values for a fitted linear model.

    For Logistic Regression this explains the model log-odds output relative to
    the transformed-background mean. This is the standard linear SHAP form for
    independent features and avoids slow notebook execution for one sample.
    """

    pipeline = model_bundle["model"]
    estimator = pipeline.named_steps["model"]
    background_transformed = transform_features(model_bundle, background)
    sample_transformed = transform_features(model_bundle, sample)

    coefficients = pd.Series(estimator.coef_[0], index=sample_transformed.columns)
    baseline = background_transformed.mean(axis=0)
    shap_values = (sample_transformed.iloc[0] - baseline) * coefficients
    return shap_values, sample_transformed.iloc[0]


def _fallback_shap_values(model_bundle: dict[str, Any], background: pd.DataFrame, sample: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    import shap

    pipeline = model_bundle["model"]
    estimator = pipeline.named_steps["model"]
    background_transformed = transform_features(model_bundle, background)
    sample_transformed = transform_features(model_bundle, sample)
    explainer = shap.Explainer(estimator.predict_proba, background_transformed)
    shap_values = explainer(sample_transformed).values[0]

    if getattr(shap_values, "ndim", 1) > 1:
        shap_values = shap_values[:, 1]

    return pd.Series(shap_values, index=sample_transformed.columns), sample_transformed.iloc[0]


def explain_sample(
    model_bundle: dict[str, Any],
    background: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    top_n: int = 10,
) -> dict[str, Any]:
    """Explain one SECOM sample with SHAP feature contributions.

    Returned SHAP values explain model behavior for the selected sample. They do
    not establish physical causality.
    """

    pipeline = model_bundle["model"]
    estimator = pipeline.named_steps["model"]

    if estimator.__class__.__name__ == "LogisticRegression":
        shap_values, transformed_sample = _linear_shap_values(model_bundle, background, sample)
        value_units = "log_odds"
    else:
        shap_values, transformed_sample = _fallback_shap_values(model_bundle, background, sample)
        value_units = "model_output"

    probability = float(pipeline.predict_proba(sample)[:, 1][0])
    prediction = "fail" if probability >= 0.5 else "pass"
    confidence = probability if prediction == "fail" else 1.0 - probability

    contributions = pd.DataFrame(
        {
            "feature": transformed_sample.index,
            "feature_value": transformed_sample.to_numpy(),
            "shap_value": shap_values.to_numpy(),
        }
    )
    contributions["abs_shap_value"] = contributions["shap_value"].abs()
    contributions["direction"] = contributions["shap_value"].map(
        lambda value: "pushes_toward_failure" if value > 0 else "pushes_toward_pass"
    )
    top_contributions = contributions.sort_values("abs_shap_value", ascending=False).head(top_n)

    return {
        "prediction": prediction,
        "yield_risk_score": probability,
        "confidence": confidence,
        "top_contributions": top_contributions.reset_index(drop=True),
        "shap_value_units": value_units,
        "safe_interpretation_note": (
            "SHAP values describe model feature contributions for this sample; "
            "they do not prove physical root causes."
        ),
    }
