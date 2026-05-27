from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .paths import CONFIG_ROOT, EXTERNAL_ROOT, LOCAL_RUNTIME_ROOT


def _parse_length_range(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return None
    if "-" in text:
        left, right = text.split("-", maxsplit=1)
        return [int(left), int(right)]
    return [int(text), int(text)]


@dataclass
class TaskSpec:
    task_id: str
    readable_name: str
    tier: str
    suite: str
    species: str
    source: str
    classification_type: str
    train_test_source: str
    raw_dataset_id: str
    sequence_length_range: list[int] | None
    current_model_coverage: list[str] = field(default_factory=list)
    allowed_perturbations: list[str] = field(default_factory=list)
    available_metadata_fields: list[str] = field(default_factory=list)
    recommended_split_protocols: list[str] = field(default_factory=list)
    citation_key: str = ""
    track: str = "short_context"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitSpec:
    split_id: str
    readable_name: str
    kind: str
    description: str
    chrom_folds: dict[str, list[str]]
    matching_rules: dict[str, Any]
    train_per_label: int | None = None
    val_per_label: int | None = None
    test_per_label: int | None = None
    status: str = "completed"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    model_id: str
    readable_name: str
    family: str
    mode_support: list[str]
    input_length_limits: int | None
    expected_embedding_dimension: int | None
    gpu_requirement: str
    citation_key: str = ""
    status: str = "completed"
    model_checkpoint: str | None = None
    tokenizer_name: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerturbationSpec:
    perturbation_id: str
    readable_name: str
    kind: str
    description: str
    params: dict[str, Any]
    deterministic_with_seed: bool = True
    status: str = "completed"
    raw: dict[str, Any] = field(default_factory=dict)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=1)
def _task_manifest_index() -> dict[str, TaskSpec]:
    rows = _load_jsonl(CONFIG_ROOT / "task_manifests.jsonl")
    index: dict[str, TaskSpec] = {}
    for row in rows:
        tier = str(row.get("tier", "core"))
        task_id = str(row["task_id"])
        index[task_id] = TaskSpec(
            task_id=task_id,
            readable_name=str(row.get("readable_name", task_id)),
            tier=tier,
            suite=tier,
            species=str(row.get("species", "unknown")),
            source=str(row.get("source", "")),
            classification_type=str(row.get("classification_type", "binary_sequence_classification")),
            train_test_source=str(row.get("train_test_source", "")),
            raw_dataset_id=str(row.get("raw_dataset_id", task_id)),
            sequence_length_range=_parse_length_range(row.get("sequence_length_or_range")),
            current_model_coverage=list(row.get("current_model_coverage", [])),
            allowed_perturbations=list(row.get("allowed_perturbations", [])),
            available_metadata_fields=list(row.get("available_metadata_fields", [])),
            recommended_split_protocols=list(row.get("recommended_split_protocols", [])),
            citation_key=str(row.get("citation_key", "")),
            track="synthetic" if tier == "synthetic" else "short_context",
            raw=row,
        )
    return index


@lru_cache(maxsize=1)
def _split_manifest_index() -> dict[str, SplitSpec]:
    rows = _load_jsonl(CONFIG_ROOT / "split_manifests.jsonl")
    index: dict[str, SplitSpec] = {}
    for row in rows:
        split_id = str(row["split_id"])
        chrom_folds = {str(key): [str(item) for item in value] for key, value in dict(row.get("chrom_folds", {})).items()}
        index[split_id] = SplitSpec(
            split_id=split_id,
            readable_name=str(row.get("readable_name", split_id)),
            kind=str(row.get("kind", "official")),
            description=str(row.get("description", "")),
            chrom_folds=chrom_folds,
            matching_rules=dict(row.get("matching_rules", {})),
            train_per_label=row.get("train_per_label"),
            val_per_label=row.get("val_per_label"),
            test_per_label=row.get("test_per_label"),
            status=str(row.get("status", "completed")),
            raw=row,
        )
    return index


def _default_checkpoint(model_id: str) -> str | None:
    local_dnabert = EXTERNAL_ROOT / "dnabert2_local"
    sibling_dnabert = LOCAL_RUNTIME_ROOT / "external" / "dnabert2_local"
    resolved_dnabert = local_dnabert if local_dnabert.exists() else sibling_dnabert
    defaults = {
        "dnabert2": str(resolved_dnabert) if resolved_dnabert.exists() else "zhihan1996/DNABERT-2-117M",
        "caduceus_ph": "kuleshov-group/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3",
        "nucleotide_transformer_v2": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    }
    return defaults.get(model_id)


@lru_cache(maxsize=1)
def _model_manifest_index() -> dict[str, ModelSpec]:
    rows = _load_jsonl(CONFIG_ROOT / "model_manifests.jsonl")
    index: dict[str, ModelSpec] = {}
    for row in rows:
        model_id = str(row["model_id"])
        checkpoint = row.get("model_checkpoint") or _default_checkpoint(model_id)
        tokenizer_name = row.get("tokenizer_name") or checkpoint
        index[model_id] = ModelSpec(
            model_id=model_id,
            readable_name=str(row.get("readable_name", model_id)),
            family=str(row.get("family", "unknown")),
            mode_support=list(row.get("mode_support", [])),
            input_length_limits=row.get("input_length_limits"),
            expected_embedding_dimension=row.get("expected_embedding_dimension"),
            gpu_requirement=str(row.get("GPU_requirement", "optional")),
            citation_key=str(row.get("citation_key", "")),
            status=str(row.get("status", "completed")),
            model_checkpoint=str(checkpoint) if checkpoint else None,
            tokenizer_name=str(tokenizer_name) if tokenizer_name else None,
            raw=row,
        )
    return index


@lru_cache(maxsize=1)
def _perturbation_manifest_index() -> dict[str, PerturbationSpec]:
    rows = _load_jsonl(CONFIG_ROOT / "perturbation_manifests.jsonl")
    index: dict[str, PerturbationSpec] = {}
    for row in rows:
        perturbation_id = str(row["perturbation_id"])
        index[perturbation_id] = PerturbationSpec(
            perturbation_id=perturbation_id,
            readable_name=str(row.get("readable_name", perturbation_id)),
            kind=str(row.get("kind", "")),
            description=str(row.get("description", "")),
            params=dict(row.get("params", {})),
            deterministic_with_seed=bool(row.get("deterministic_with_seed", True)),
            status=str(row.get("status", "completed")),
            raw=row,
        )
    return index


def get_task_spec(task_id: str) -> TaskSpec:
    return _task_manifest_index()[task_id]


def get_split_spec(split_id: str) -> SplitSpec:
    return _split_manifest_index()[split_id]


def get_model_spec(model_id: str) -> ModelSpec:
    return _model_manifest_index()[model_id]


def get_perturbation_spec(perturbation_id: str) -> PerturbationSpec:
    return _perturbation_manifest_index()[perturbation_id]


def list_specs(kind: str) -> list[str]:
    normalized = kind.strip().lower()
    if normalized in {"task", "tasks"}:
        return sorted(_task_manifest_index())
    if normalized in {"split", "splits"}:
        return sorted(_split_manifest_index())
    if normalized in {"model", "models"}:
        return sorted(_model_manifest_index())
    if normalized in {"perturbation", "perturbations"}:
        return sorted(_perturbation_manifest_index())
    raise KeyError(f"Unknown manifest kind: {kind}")
