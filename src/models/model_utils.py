"""Shared model artifact helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def project_root() -> Path:
    """Return the repository root based on this module location."""

    return Path(__file__).resolve().parents[2]


def artifacts_dir() -> Path:
    """Return the default local artifacts directory."""

    path = project_root() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_joblib_artifact(obj: Any, filename: str, directory: str | Path | None = None) -> Path:
    """Save an object with joblib and return the output path."""

    output_dir = Path(directory) if directory is not None else artifacts_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    joblib.dump(obj, output_path)
    return output_path


def load_joblib_artifact(path: str | Path) -> Any:
    """Load a joblib artifact from disk."""

    return joblib.load(Path(path))
