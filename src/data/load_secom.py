"""Utilities for loading and validating the raw SECOM dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SecomPaths:
    """Resolved paths for the expected SECOM raw-data files."""

    raw_dir: Path
    features: Path
    labels: Path
    names: Path


def project_root() -> Path:
    """Return the repository root based on this module location."""

    return Path(__file__).resolve().parents[2]


def resolve_secom_paths(raw_dir: str | Path | None = None) -> SecomPaths:
    """Resolve expected SECOM file paths.

    Args:
        raw_dir: Optional directory containing `secom.data`, `secom_labels.data`,
            and `secom.names`. Defaults to `data/raw/secom` under the repo root.
    """

    resolved_raw_dir = Path(raw_dir) if raw_dir is not None else project_root() / "data" / "raw" / "secom"
    resolved_raw_dir = resolved_raw_dir.expanduser().resolve()

    return SecomPaths(
        raw_dir=resolved_raw_dir,
        features=resolved_raw_dir / "secom.data",
        labels=resolved_raw_dir / "secom_labels.data",
        names=resolved_raw_dir / "secom.names",
    )


def validate_secom_files(raw_dir: str | Path | None = None) -> SecomPaths:
    """Validate that all expected SECOM raw files exist."""

    paths = resolve_secom_paths(raw_dir)
    missing_paths = [path for path in (paths.features, paths.labels, paths.names) if not path.exists()]

    if missing_paths:
        expected = "\n".join(f"- {path}" for path in (paths.features, paths.labels, paths.names))
        missing = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Missing expected SECOM raw-data files.\n\n"
            f"Expected files:\n{expected}\n\n"
            f"Missing files:\n{missing}\n\n"
            "Place the attached secom.zip contents in data/raw/secom/."
        )

    return paths


def _import_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to load SECOM data. Install project dependencies with "
            "`pip install -r requirements.txt` before running this loader."
        ) from exc

    return pd


def load_secom(raw_dir: str | Path | None = None) -> tuple[Any, Any]:
    """Load SECOM features and labels.

    Returns:
        A tuple of `(features, labels)` pandas DataFrames. Features are named
        `sensor_feature_000`, `sensor_feature_001`, etc. Labels include the raw
        target plus a parsed timestamp column.
    """

    pd = _import_pandas()
    paths = validate_secom_files(raw_dir)

    features = pd.read_csv(
        paths.features,
        sep=r"\s+",
        header=None,
        na_values=["NaN"],
        engine="python",
    )
    labels = pd.read_csv(
        paths.labels,
        sep=r"\s+",
        header=None,
        names=["target", "date", "time"],
        engine="python",
    )

    features.columns = [f"sensor_feature_{idx:03d}" for idx in range(features.shape[1])]
    labels["target"] = labels["target"].astype(int)
    labels["timestamp"] = pd.to_datetime(
        labels["date"] + " " + labels["time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    if len(features) != len(labels):
        raise ValueError(f"SECOM row mismatch: features={len(features)} labels={len(labels)}")

    unexpected_labels = set(labels["target"].unique()) - {-1, 1}
    if unexpected_labels:
        raise ValueError(f"Unexpected SECOM label values: {sorted(unexpected_labels)}")

    return features, labels


def summarize_secom(features: Any, labels: Any) -> dict[str, Any]:
    """Return a compact summary for EDA and validation logs."""

    target_counts = labels["target"].value_counts().sort_index().to_dict()
    missing_values = int(features.isna().sum().sum())
    unique_counts = features.nunique(dropna=True)

    return {
        "feature_rows": int(features.shape[0]),
        "feature_columns": int(features.shape[1]),
        "label_rows": int(labels.shape[0]),
        "missing_feature_values": missing_values,
        "constant_or_single_value_features": int((unique_counts <= 1).sum()),
        "target_counts": target_counts,
    }

