from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .paths import GENOMECF_REGISTRY_ROOT, RELEASE_ROOT, normalize_legacy_path_text


def _normalize_registry_paths(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in ["source_artifact", "metadata_json", "package_versions"]:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(lambda value: normalize_legacy_path_text(value) if pd.notna(value) else value)
    return normalized


def summarize_release_registry(registry: pd.DataFrame) -> pd.DataFrame:
    if registry.empty:
        return pd.DataFrame()
    base_columns = [
        "task_id",
        "task_readable_name",
        "tier",
        "track",
        "species",
        "split_id",
        "split_fold",
        "model_id",
        "model_readable_name",
        "model_family",
        "mode",
        "calibration_method",
        "intervention_id",
    ]
    for column in base_columns:
        if column not in registry.columns:
            registry[column] = None
    rows: list[dict[str, object]] = []
    grouped = registry.groupby(base_columns, dropna=False, as_index=False)
    for _, group in grouped:
        original = group[group["perturbation_id"].fillna(group["perturbation"]) == "original"]
        original_row = original.iloc[0] if not original.empty else group.iloc[0]
        row = {
            "task_id": original_row.get("task_id"),
            "task_readable_name": original_row.get("task_readable_name"),
            "tier": original_row.get("tier"),
            "track": original_row.get("track"),
            "species": original_row.get("species"),
            "split_id": original_row.get("split_id"),
            "split_fold": original_row.get("split_fold", "none"),
            "model_id": original_row.get("model_id"),
            "model_readable_name": original_row.get("model_readable_name"),
            "model_family": original_row.get("model_family"),
            "mode": original_row.get("mode"),
            "calibration_method": original_row.get("calibration_method", "none"),
            "intervention_id": original_row.get("intervention_id", "standard"),
            "seeds": int(group["seed"].nunique()) if "seed" in group.columns else 1,
            "auroc": original_row.get("auroc"),
            "auprc": original_row.get("auprc"),
            "accuracy": original_row.get("accuracy"),
            "balanced_accuracy": original_row.get("balanced_accuracy"),
            "mcc": original_row.get("mcc"),
            "f1": original_row.get("f1"),
            "ece": original_row.get("ece"),
            "brier": original_row.get("brier"),
            "train_count": original_row.get("train_count"),
            "val_count": original_row.get("val_count"),
            "test_count": original_row.get("test_count"),
            "positive_count": original_row.get("positive_count"),
            "negative_count": original_row.get("negative_count"),
            "mean_abs_delta": original_row.get("mean_abs_delta"),
            "flip_rate": original_row.get("flip_rate"),
            "positive_prob_drop": original_row.get("positive_prob_drop"),
            "calibration_shift": original_row.get("calibration_shift"),
            "sequence_length_summary": original_row.get("sequence_length_summary"),
            "gc_only_auroc": original_row.get("gc_only_auroc"),
            "gc_only_explainability_ratio": original_row.get("gc_only_explainability_ratio"),
            "shortcut_score": original_row.get("shortcut_score"),
        }
        mapping = {
            "reverse_complement": "rc",
            "k1_shuffle": "mono",
            "k2_shuffle": "dinuc",
            "motif_disruption": "motif",
        }
        for perturbation_id, prefix in mapping.items():
            subset = group[group["perturbation_id"].fillna(group["perturbation"]) == perturbation_id]
            if subset.empty:
                row[f"{prefix}_mean_abs_delta"] = float("nan")
                row[f"{prefix}_flip_rate"] = float("nan")
                row[f"{prefix}_positive_prob_drop"] = float("nan")
                row[f"{prefix}_calibration_shift"] = float("nan")
            else:
                cf_row = subset.iloc[0]
                row[f"{prefix}_mean_abs_delta"] = cf_row.get("mean_abs_delta")
                row[f"{prefix}_flip_rate"] = cf_row.get("flip_rate")
                row[f"{prefix}_positive_prob_drop"] = cf_row.get("positive_prob_drop")
                row[f"{prefix}_calibration_shift"] = cf_row.get("calibration_shift")
        if pd.notna(row.get("mono_positive_prob_drop")):
            row["mono_retention"] = 1.0 + float(row["mono_positive_prob_drop"])
        if pd.notna(row.get("dinuc_positive_prob_drop")):
            row["dinuc_retention"] = 1.0 + float(row["dinuc_positive_prob_drop"])
        rows.append(row)
    return pd.DataFrame(rows)


def _build_model_task_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    subset = summary[
        (summary["split_id"] == "official")
        & (summary["calibration_method"] == "none")
        & (summary["intervention_id"] == "standard")
    ].copy()
    if subset.empty:
        return subset
    return subset.pivot_table(index="task_id", columns="model_id", values="auroc", aggfunc="mean").reset_index()


def build_release_registry(project_root: Path | None = None) -> dict[str, str]:
    release_root = RELEASE_ROOT if project_root is None else Path(project_root) / "results" / "release"
    registry_csv = release_root / "benchmark_registry.csv"
    registry_jsonl = release_root / "benchmark_registry.jsonl"
    summary_csv = release_root / "benchmark_summary.csv"
    matrix_csv = release_root / "model_task_matrix.csv"

    if not registry_csv.exists():
        source = GENOMECF_REGISTRY_ROOT / "result_registry.csv"
        if not source.exists():
            raise FileNotFoundError("Could not find a source registry to build the release registry.")
        frame = pd.read_csv(source)
        frame = _normalize_registry_paths(frame)
        release_root.mkdir(parents=True, exist_ok=True)
        frame.to_csv(registry_csv, index=False)
    else:
        frame = _normalize_registry_paths(pd.read_csv(registry_csv))
        frame.to_csv(registry_csv, index=False)

    if not registry_jsonl.exists():
        with registry_jsonl.open("w", encoding="utf-8") as handle:
            for record in frame.to_dict(orient="records"):
                handle.write(json.dumps(record) + "\n")
    if summary_csv.exists():
        summary = pd.read_csv(summary_csv)
    else:
        summary = summarize_release_registry(frame)
        summary.to_csv(summary_csv, index=False)
    if matrix_csv.exists():
        matrix = pd.read_csv(matrix_csv)
    else:
        matrix = _build_model_task_matrix(summary)
        matrix.to_csv(matrix_csv, index=False)
    return {
        "registry_csv": str(registry_csv),
        "registry_jsonl": str(registry_jsonl),
        "summary_csv": str(summary_csv),
        "matrix_csv": str(matrix_csv),
    }
