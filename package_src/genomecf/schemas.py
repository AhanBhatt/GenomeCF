from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ResultRow:
    task: str
    suite: str
    track: str
    split_name: str
    split_fold: str
    model: str
    model_family: str
    mode: str
    seed: int
    perturbation: str
    metric_scope: str
    auroc: float | None = None
    auprc: float | None = None
    accuracy: float | None = None
    balanced_accuracy: float | None = None
    mcc: float | None = None
    f1: float | None = None
    ece: float | None = None
    brier: float | None = None
    calibration_shift: float | None = None
    mean_abs_delta: float | None = None
    flip_rate: float | None = None
    positive_prob_drop: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    runtime_s: float | None = None
    peak_memory_mb: float | None = None
    model_hash: str | None = None
    package_versions: str | None = None
    metadata_json: str | None = None
    task_id: str | None = None
    task_readable_name: str | None = None
    tier: str | None = None
    species: str | None = None
    split_id: str | None = None
    model_id: str | None = None
    model_readable_name: str | None = None
    ensemble_id: str | None = None
    perturbation_id: str | None = None
    train_count: int | None = None
    val_count: int | None = None
    test_count: int | None = None
    positive_count: int | None = None
    negative_count: int | None = None
    sequence_length_summary: str | None = None
    device: str | None = None
    runtime_seconds: float | None = None
    calibration_method: str | None = None
    intervention_id: str | None = None
    model_checkpoint: str | None = None
    tokenizer_name: str | None = None
    pooling: str | None = None
    max_length: int | None = None
    embedding_dim: int | None = None
    batch_size: int | None = None
    truncation: str | None = None
    config_hash: str | None = None
    data_hash: str | None = None
    created_at: str | None = None
    source_artifact: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
