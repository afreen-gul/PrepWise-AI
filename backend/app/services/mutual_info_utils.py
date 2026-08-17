"""Lightweight mutual-information helpers (sklearn preferred, numpy fallback)."""

from __future__ import annotations

import numpy as np


def _histogram_mi_discrete(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate MI for discrete/binned variables via joint histogram."""
    if len(x) == 0:
        return 0.0
    _, x_inv = np.unique(x, return_inverse=True)
    _, y_inv = np.unique(y, return_inverse=True)
    n_x = int(x_inv.max()) + 1
    n_y = int(y_inv.max()) + 1
    if n_x <= 1 or n_y <= 1:
        return 0.0
    joint = np.bincount(x_inv * n_y + y_inv, minlength=n_x * n_y).reshape(n_x, n_y).astype(float)
    total = joint.sum()
    if total <= 0:
        return 0.0
    joint /= total
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((joint > 0) & (px > 0) & (py > 0), joint / (px * py), 1.0)
        mi = float(np.nansum(joint * np.log(ratio)))
    return float(max(mi, 0.0))


def _bin_continuous(values: np.ndarray, max_bins: int = 10) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.zeros(len(values), dtype=int)
    n_unique = len(np.unique(finite))
    bins = max(2, min(max_bins, n_unique))
    try:
        digitized = np.digitize(
            values,
            np.linspace(finite.min(), finite.max() + 1e-12, bins + 1)[1:-1],
        )
    except Exception:
        digitized = np.zeros(len(values), dtype=int)
    return digitized.astype(int)


def mutual_info_scores(
    X: np.ndarray,
    y: np.ndarray,
    *,
    discrete_features: np.ndarray,
    task: str,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute MI for each feature column in X against y.

    Uses scikit-learn when available; otherwise a histogram-based estimator.
    """
    n_features = X.shape[1]
    try:
        from sklearn.feature_selection import (
            mutual_info_classif,
            mutual_info_regression,
        )

        if task == "classification":
            return np.asarray(
                mutual_info_classif(
                    X,
                    y,
                    discrete_features=discrete_features,
                    random_state=random_state,
                ),
                dtype=float,
            )
        return np.asarray(
            mutual_info_regression(
                X,
                y,
                discrete_features=discrete_features,
                random_state=random_state,
            ),
            dtype=float,
        )
    except Exception:
        pass

    # Fallback: bin continuous vars and use discrete MI
    if task == "classification":
        y_binned = y.astype(int) if np.issubdtype(y.dtype, np.integer) else _bin_continuous(y.astype(float))
    else:
        y_binned = _bin_continuous(np.asarray(y, dtype=float))

    scores = np.zeros(n_features, dtype=float)
    for i in range(n_features):
        col = X[:, i]
        if discrete_features[i]:
            x_binned = col.astype(int) if np.issubdtype(col.dtype, np.integer) else _bin_continuous(col)
        else:
            x_binned = _bin_continuous(col.astype(float))
        scores[i] = _histogram_mi_discrete(x_binned, y_binned)
    return scores
