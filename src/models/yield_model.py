"""SECOM yield-risk modeling pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


def make_binary_target(labels: pd.DataFrame) -> pd.Series:
    """Map raw SECOM labels to binary modeling targets.

    The raw SECOM target convention is `-1` for pass and `1` for fail.
    For modeling metrics, FabPilot uses `0` for pass and `1` for fail.
    """

    if "target" not in labels.columns:
        raise KeyError("Expected labels DataFrame to include a `target` column.")

    unexpected = set(labels["target"].unique()) - {-1, 1}
    if unexpected:
        raise ValueError(f"Unexpected SECOM target values: {sorted(unexpected)}")

    return (labels["target"] == 1).astype(int)


@dataclass
class HighMissingFeatureDropper(BaseEstimator, TransformerMixin):
    """Drop columns whose missing-value share exceeds a threshold."""

    threshold: float = 0.5

    def fit(self, X: Any, y: Any = None) -> "HighMissingFeatureDropper":
        frame = self._as_frame(X)
        missing_share = frame.isna().mean()
        self.feature_names_in_ = frame.columns.to_numpy()
        self.keep_columns_ = missing_share[missing_share <= self.threshold].index.to_list()
        self.drop_columns_ = missing_share[missing_share > self.threshold].index.to_list()
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        frame = self._as_frame(X)
        return frame.loc[:, self.keep_columns_]

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.keep_columns_)

    def _as_frame(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


def build_preprocessing_pipeline(*, scale: bool = False, missing_threshold: float = 0.5) -> Pipeline:
    """Build shared preprocessing steps for SECOM tabular models."""

    steps: list[tuple[str, Any]] = [
        ("drop_high_missing", HighMissingFeatureDropper(threshold=missing_threshold)),
        ("imputer", SimpleImputer(strategy="median")),
        ("variance_filter", VarianceThreshold(threshold=0.0)),
    ]

    if scale:
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps)


def build_logistic_regression_pipeline(*, missing_threshold: float = 0.5, random_state: int = 42) -> Pipeline:
    """Build a class-balanced Logistic Regression baseline."""

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessing_pipeline(scale=True, missing_threshold=missing_threshold)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )


def build_random_forest_pipeline(*, missing_threshold: float = 0.5, random_state: int = 42) -> Pipeline:
    """Build a class-balanced Random Forest baseline."""

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessing_pipeline(scale=False, missing_threshold=missing_threshold)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(
    *,
    scale_pos_weight: float,
    missing_threshold: float = 0.5,
    random_state: int = 42,
) -> Pipeline:
    """Build an XGBoost candidate model."""

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Install `xgboost` to build the XGBoost yield model.") from exc

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessing_pipeline(scale=False, missing_threshold=missing_threshold)),
            (
                "model",
                XGBClassifier(
                    n_estimators=250,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_lightgbm_pipeline(
    *,
    missing_threshold: float = 0.5,
    random_state: int = 42,
) -> Pipeline:
    """Build a LightGBM candidate model."""

    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("Install `lightgbm` to build the LightGBM yield model.") from exc

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessing_pipeline(scale=False, missing_threshold=missing_threshold)),
            (
                "model",
                LGBMClassifier(
                    n_estimators=250,
                    learning_rate=0.05,
                    num_leaves=15,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )
