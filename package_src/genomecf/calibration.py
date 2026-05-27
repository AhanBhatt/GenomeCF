from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProbabilityCalibrator:
    method: str
    temperature: float = 1.0

    def transform(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        if self.method == "none" or self.temperature == 1.0:
            return probs
        clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped))
        scaled = logits / max(self.temperature, 1e-6)
        return 1.0 / (1.0 + np.exp(-scaled))


def fit_calibrator(method: str, validation_probs: np.ndarray, validation_labels: np.ndarray) -> ProbabilityCalibrator:
    normalized = method.strip().lower()
    if normalized in {"", "none"}:
        return ProbabilityCalibrator(method="none", temperature=1.0)
    if normalized == "temperature":
        probs = np.asarray(validation_probs, dtype=float)
        labels = np.asarray(validation_labels, dtype=float)
        best_temp = 1.0
        best_loss = float("inf")
        for temp in np.linspace(0.5, 5.0, 46):
            clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
            logits = np.log(clipped / (1.0 - clipped))
            scaled = 1.0 / (1.0 + np.exp(-(logits / temp)))
            loss = float(np.mean(-(labels * np.log(np.clip(scaled, 1e-6, 1.0)) + (1.0 - labels) * np.log(np.clip(1.0 - scaled, 1e-6, 1.0)))))
            if loss < best_loss:
                best_loss = loss
                best_temp = float(temp)
        return ProbabilityCalibrator(method="temperature", temperature=best_temp)
    return ProbabilityCalibrator(method=normalized, temperature=1.0)
