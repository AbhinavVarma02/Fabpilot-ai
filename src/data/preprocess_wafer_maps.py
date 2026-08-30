"""WM-811K wafer-map loading and preprocessing helpers."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


CANONICAL_LABELS = {
    "none": "None",
    "Center": "Center",
    "Donut": "Donut",
    "Edge-Loc": "Edge-Loc",
    "Edge-Ring": "Edge-Ring",
    "Loc": "Local",
    "Random": "Random",
    "Scratch": "Scratch",
    "Near-full": "Near-full",
}


def _install_legacy_pandas_pickle_aliases() -> None:
    """Install module aliases needed by older pandas pickles."""

    sys.modules["pandas.indexes"] = pd.core.indexes
    sys.modules["pandas.indexes.base"] = pd.core.indexes.base
    if hasattr(pd.core.indexes, "numeric"):
        sys.modules["pandas.indexes.numeric"] = pd.core.indexes.numeric


def load_wm811k_pickle(path: str | Path) -> pd.DataFrame:
    """Load the legacy WM-811K pickle with compatibility shims."""

    _install_legacy_pandas_pickle_aliases()
    with Path(path).open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def normalize_nested_label(value: Any) -> str | None:
    """Normalize WM-811K nested label values to a plain class string."""

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.ravel()[0]
    elif isinstance(value, list):
        if not value:
            return None
        while isinstance(value, list) and value:
            value = value[0]

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    label = str(value)
    if label in {"", "[]", "nan", "None"}:
        return None
    return CANONICAL_LABELS.get(label, label)


def add_normalized_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized failure and train/test labels."""

    labeled = df.copy()
    labeled["failure_label"] = labeled["failureType"].map(normalize_nested_label)
    labeled["split_label"] = labeled["trianTestLabel"].map(normalize_nested_label)
    return labeled


def resize_wafer_map(wafer_map: Any, output_size: int = 64) -> np.ndarray:
    """Resize a variable-size wafer map to a square float32 array."""

    array = np.asarray(wafer_map, dtype=np.uint8)
    image = Image.fromarray(array)
    resized = image.resize((output_size, output_size), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.float32) / 2.0


def build_balanced_labeled_sample(
    df: pd.DataFrame,
    *,
    max_per_class: int = 400,
    output_size: int = 64,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Build a balanced labeled wafer-map sample suitable for a small CNN."""

    labeled = add_normalized_labels(df)
    labeled = labeled[labeled["failure_label"].notna()].copy()

    sampled_parts = []
    for _, group in labeled.groupby("failure_label"):
        n = min(max_per_class, len(group))
        sampled_parts.append(group.sample(n=n, random_state=random_state))

    sample_df = pd.concat(sampled_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    class_names = sorted(sample_df["failure_label"].unique().tolist())
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    X = np.stack([resize_wafer_map(wafer_map, output_size=output_size) for wafer_map in sample_df["waferMap"]])
    X = X[:, None, :, :].astype(np.float32)
    y = sample_df["failure_label"].map(class_to_idx).to_numpy(dtype=np.int64)

    return X, y, class_names, sample_df


def save_processed_sample(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    sample_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save processed wafer-map sample arrays, metadata, and class mapping."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays_path = output_dir / "wm811k_sample.npz"
    metadata_path = output_dir / "wm811k_sample_metadata.csv"
    class_mapping_path = output_dir / "wm811k_class_mapping.json"

    np.savez_compressed(arrays_path, X=X, y=y, class_names=np.array(class_names))
    sample_df.drop(columns=["waferMap"]).to_csv(metadata_path, index=False)
    class_mapping_path.write_text(
        json.dumps({class_name: idx for idx, class_name in enumerate(class_names)}, indent=2),
        encoding="utf-8",
    )

    return {
        "arrays": arrays_path,
        "metadata": metadata_path,
        "class_mapping": class_mapping_path,
    }


def stratified_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Create stratified train/validation/test splits for processed arrays."""

    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
        stratify=y[train_val_idx],
        random_state=random_state,
    )

    return {"train": train_idx, "val": val_idx, "test": test_idx}
