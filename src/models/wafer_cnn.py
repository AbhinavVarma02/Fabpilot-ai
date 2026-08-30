"""Small PyTorch CNN for WM-811K wafer-map classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class WaferTrainingConfig:
    batch_size: int = 64
    epochs: int = 4
    learning_rate: float = 1e-3
    random_state: int = 42


def build_wafer_cnn(num_classes: int) -> Any:
    """Build a compact CNN for 64x64 single-channel wafer maps."""

    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(64 * 4 * 4, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    )


def make_tensor_dataset(X: np.ndarray, y: np.ndarray) -> Any:
    """Create a TensorDataset from numpy wafer arrays."""

    import torch
    from torch.utils.data import TensorDataset

    return TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())


def predict_probabilities(model: Any, X: np.ndarray, *, batch_size: int = 256) -> np.ndarray:
    """Return softmax probabilities for wafer maps."""

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float()), batch_size=batch_size, shuffle=False)
    probabilities = []
    with torch.no_grad():
        for (batch_X,) in loader:
            logits = model(batch_X)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probabilities)
