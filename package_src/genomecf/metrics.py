from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins[1:-1], right=True)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probs[mask].mean()))
    return float(ece)


def brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    return float(np.mean((probs - labels) ** 2))


def _safe_metric(fn: Callable[[], float]) -> float:
    try:
        return float(fn())
    except ValueError:
        return float("nan")


def standard_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= 0.5).astype(int)
    return {
        "auroc": _safe_metric(lambda: roc_auc_score(labels, probs)),
        "auprc": _safe_metric(lambda: average_precision_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": _safe_metric(lambda: balanced_accuracy_score(labels, preds)),
        "mcc": _safe_metric(lambda: matthews_corrcoef(labels, preds)),
        "f1": _safe_metric(lambda: f1_score(labels, preds)),
        "ece": expected_calibration_error(labels, probs),
        "brier": brier_score(labels, probs),
    }


def counterfactual_metrics(labels: np.ndarray, original_probs: np.ndarray, perturbed_probs: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    original_probs = np.asarray(original_probs, dtype=float)
    perturbed_probs = np.asarray(perturbed_probs, dtype=float)
    positive_mask = labels == 1
    return {
        "mean_abs_delta": float(np.abs(original_probs - perturbed_probs).mean()),
        "flip_rate": float(((original_probs >= 0.5) != (perturbed_probs >= 0.5)).mean()),
        "positive_prob_drop": float(original_probs[positive_mask].mean() - perturbed_probs[positive_mask].mean()) if positive_mask.any() else 0.0,
        "calibration_shift": float(expected_calibration_error(labels, perturbed_probs) - expected_calibration_error(labels, original_probs)),
    }


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray], float],
    n_items: int,
    seed: int,
    n_bootstrap: int = 200,
    ci: float = 0.95,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_items, size=n_items)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = metric_fn(idx)
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        return float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))
