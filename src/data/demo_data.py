"""Small public demo data loaders for the Streamlit app.

The full raw datasets stay local and ignored. These helpers prefer the compact
`data/demo/` files used by Hugging Face Spaces, then fall back to local raw data
for development checkouts that have the original archives extracted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.load_secom import load_secom, project_root


def demo_data_dir() -> Path:
    """Return the compact public demo-data directory."""

    return project_root() / "data" / "demo"


def secom_demo_features_path() -> Path:
    """Return the compact SECOM demo feature CSV path."""

    return demo_data_dir() / "secom_demo_features.csv"


def wafer_demo_sample_path() -> Path:
    """Return the compact WM-811K demo sample NPZ path."""

    return demo_data_dir() / "wm811k_demo_sample.npz"


def load_secom_demo_features(path: str | Path | None = None) -> pd.DataFrame:
    """Load label-free SECOM sensor-feature rows for dashboard inference."""

    return pd.read_csv(Path(path) if path is not None else secom_demo_features_path())


def load_dashboard_secom_features() -> pd.DataFrame:
    """Load SECOM rows for the dashboard, preferring public demo data."""

    demo_path = secom_demo_features_path()
    if demo_path.exists():
        return load_secom_demo_features(demo_path)

    features, _ = load_secom(project_root() / "data" / "raw" / "secom")
    return features


def load_secom_background(sample_size: int = 200, random_state: int = 42) -> pd.DataFrame:
    """Load a compact background sample for local SHAP-style explanations."""

    features = load_dashboard_secom_features()
    return features.sample(n=min(sample_size, len(features)), random_state=random_state)
