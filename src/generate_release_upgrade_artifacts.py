from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = PROJECT_ROOT / "package_src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from genomecf.config import get_split_spec, get_task_spec
from genomecf.data import build_split_frames, load_task_frame
from genomecf.paths import DISABLE_LOCAL_RUNTIME_ASSETS
from genomecf.release import build_release_registry, summarize_release_registry


RESULTS_ROOT = PROJECT_ROOT / "results" / "release"
FIGURES_ROOT = PROJECT_ROOT / "figures"
CURRENT_REGISTRY = PROJECT_ROOT / "results" / "genomecf_registry" / "result_registry.csv"
SEED = 2026
CORE_TASKS = [
    "human_nontata_promoters",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ocr_ensembl",
]
MATCHED_TASKS = [
    "human_nontata_promoters",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
]
FOCAL_TASKS = ["human_nontata_promoters", "human_enhancers_cohn"]
MODEL_ORDER = ["kmer_logistic_regression", "small_cnn", "small_cnn_rc_aug"]
CV_MODEL_ORDER = ["kmer_logistic_regression", "small_cnn", "small_cnn_rc_aug", "dnabert2", "caduceus_ph"]
MODEL_LABELS = {
    "gc_only": "GC-only logistic regression",
    "cpg_only": "CpG-only logistic regression",
    "length_only": "Length-only logistic regression",
    "repeat_only": "Repeat-only logistic regression",
    "kmer_logistic_regression": "6-mer logistic regression",
    "small_cnn": "CNN",
    "small_cnn_rc_aug": "RC-aug CNN",
    "dnabert2": "DNABERT-2",
    "caduceus_ph": "Caduceus-Ph",
    "nucleotide_transformer_v2": "Nucleotide Transformer v2",
}
TASK_LABELS = {
    "human_nontata_promoters": "Promoters",
    "human_enhancers_cohn": "Enhancers (Cohn)",
    "human_enhancers_ensembl": "Enhancers (Ensembl)",
    "human_ocr_ensembl": "Open chromatin",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _safe_float(value: object, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    return float(value)


def _standard_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        (summary["calibration_method"] == "none")
        & (summary["intervention_id"] == "standard")
    ].copy()


def _pick_summary_row(
    preferred: pd.DataFrame,
    fallback: pd.DataFrame,
    *,
    task_id: str,
    split_id: str,
    model_id: str,
    calibration_method: str = "none",
    intervention_id: str = "standard",
) -> pd.Series | None:
    mask = (
        (preferred["task_id"] == task_id)
        & (preferred["split_id"] == split_id)
        & (preferred["model_id"] == model_id)
        & (preferred["calibration_method"] == calibration_method)
        & (preferred["intervention_id"] == intervention_id)
    )
    subset = preferred[mask]
    if not subset.empty:
        return subset.iloc[0]
    mask = (
        (fallback["task_id"] == task_id)
        & (fallback["split_id"] == split_id)
        & (fallback["model_id"] == model_id)
        & (fallback["calibration_method"] == calibration_method)
        & (fallback["intervention_id"] == intervention_id)
    )
    subset = fallback[mask]
    if not subset.empty:
        return subset.iloc[0]
    return None


def build_chromosome_cv_tables(summary: pd.DataFrame, registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv = _standard_rows(summary)
    cv = cv[(cv["split_id"] == "chromosome_5fold_cv") & (cv["task_id"].isin(CORE_TASKS)) & (cv["model_id"].isin(CV_MODEL_ORDER))].copy()
    cv["task_label"] = cv["task_id"].map(TASK_LABELS)
    cv["model_label"] = cv["model_id"].map(MODEL_LABELS)

    summary_rows = (
        cv.groupby(["task_id", "task_label", "model_id", "model_label"], as_index=False)
        .agg(
            folds=("split_fold", "nunique"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            ece_mean=("ece", "mean"),
            brier_mean=("brier", "mean"),
            rc_mean_abs_delta_mean=("rc_mean_abs_delta", "mean"),
            mono_positive_prob_drop_mean=("mono_positive_prob_drop", "mean"),
            dinuc_positive_prob_drop_mean=("dinuc_positive_prob_drop", "mean"),
            test_count_total=("test_count", "sum"),
        )
        .sort_values(["task_id", "model_id"])
    )

    fold_rows = cv[
        [
            "task_id",
            "split_fold",
            "model_id",
            "auroc",
            "ece",
            "brier",
            "rc_mean_abs_delta",
            "mono_positive_prob_drop",
            "dinuc_positive_prob_drop",
            "train_count",
            "val_count",
            "test_count",
            "positive_count",
            "negative_count",
        ]
    ].copy()
    fold_rows["task_label"] = fold_rows["task_id"].map(TASK_LABELS)
    fold_rows["model_label"] = fold_rows["model_id"].map(MODEL_LABELS)
    fold_rows = fold_rows.sort_values(["task_id", "model_id", "split_fold"])
    return summary_rows, fold_rows


def _confounder_block(frame: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    block: dict[str, float | int] = {f"{prefix}_n": int(len(frame))}
    for label_value, label_prefix in [(0, "negative"), (1, "positive")]:
        subset = frame[frame["label"] == label_value]
        block[f"{prefix}_{label_prefix}_n"] = int(len(subset))
        for column in ["gc_fraction", "cpg_oe", "length", "repeat_fraction", "n_fraction"]:
            block[f"{prefix}_{label_prefix}_{column}_mean"] = float(subset[column].mean())
    return block


def _existing_matched_confounders() -> pd.DataFrame | None:
    path = RESULTS_ROOT / "matched_negative_confounders.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_matched_negative_tables(summary: pd.DataFrame, package_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    official_spec = get_split_spec("official")
    matched_spec = get_split_spec("matched_test")
    existing_confounders = _existing_matched_confounders()
    used_existing = False
    for task_name in MATCHED_TASKS:
        row: dict[str, object]
        try:
            frame = load_task_frame(get_task_spec(task_name), PROJECT_ROOT)
            official = build_split_frames(frame, official_spec, seed=SEED)["test"]
            matched = build_split_frames(frame, matched_spec, seed=SEED)["test"]
            row = {
                "task_id": task_name,
                "task_label": TASK_LABELS[task_name],
                **_confounder_block(official, "official"),
                **_confounder_block(matched, "matched"),
            }
        except FileNotFoundError:
            if existing_confounders is None:
                raise
            cached = existing_confounders[existing_confounders["task_id"] == task_name]
            if cached.empty:
                raise
            row = cached.iloc[0].to_dict()
            used_existing = True
        official_gc = summary[
            (summary["task_id"] == task_name)
            & (summary["split_id"] == "official")
            & (summary["model_id"] == "gc_only")
            & (summary["calibration_method"] == "none")
            & (summary["intervention_id"] == "standard")
        ]["auroc"]
        matched_gc = summary[
            (summary["task_id"] == task_name)
            & (summary["split_id"] == "matched_test")
            & (summary["model_id"] == "gc_only")
            & (summary["calibration_method"] == "none")
            & (summary["intervention_id"] == "standard")
        ]["auroc"]
        row["gc_only_auroc_official"] = float(official_gc.iloc[0]) if not official_gc.empty else np.nan
        row["gc_only_auroc_matched"] = float(matched_gc.iloc[0]) if not matched_gc.empty else np.nan
        rows.append(row)

    confounders = pd.DataFrame(rows)
    if used_existing:
        print("Reused committed matched-negative confounder summaries because raw task data is not available in the public repo checkout.")
    rows: list[dict[str, object]] = []
    requested_models = ["gc_only", "kmer_logistic_regression", "small_cnn", "small_cnn_rc_aug", "dnabert2", "caduceus_ph", "nucleotide_transformer_v2"]
    for task_name in MATCHED_TASKS:
        gc_lookup = confounders[confounders["task_id"] == task_name].iloc[0]
        for model_id in requested_models:
            official = _pick_summary_row(package_summary, summary, task_id=task_name, split_id="official", model_id=model_id)
            matched = _pick_summary_row(package_summary, summary, task_id=task_name, split_id="matched_test", model_id=model_id)
            if official is None or matched is None:
                continue
            rows.append(
                {
                    "task_id": task_name,
                    "split_id": "official",
                    "model_id": model_id,
                    "task_label": TASK_LABELS[task_name],
                    "model_label": MODEL_LABELS.get(model_id, model_id),
                    "auroc": float(official["auroc"]),
                    "ece": float(official["ece"]),
                    "brier": float(official["brier"]),
                    "mono_positive_prob_drop": float(official["mono_positive_prob_drop"]),
                    "dinuc_positive_prob_drop": float(official["dinuc_positive_prob_drop"]),
                    "gc_only_explainability_ratio": _safe_float(official.get("gc_only_explainability_ratio", np.nan)),
                    "gc_only_auroc_official": float(gc_lookup["gc_only_auroc_official"]),
                    "gc_only_auroc_matched": float(gc_lookup["gc_only_auroc_matched"]),
                }
            )
            rows.append(
                {
                    "task_id": task_name,
                    "split_id": "matched_test",
                    "model_id": model_id,
                    "task_label": TASK_LABELS[task_name],
                    "model_label": MODEL_LABELS.get(model_id, model_id),
                    "auroc": float(matched["auroc"]),
                    "ece": float(matched["ece"]),
                    "brier": float(matched["brier"]),
                    "mono_positive_prob_drop": float(matched["mono_positive_prob_drop"]),
                    "dinuc_positive_prob_drop": float(matched["dinuc_positive_prob_drop"]),
                    "gc_only_explainability_ratio": _safe_float(matched.get("gc_only_explainability_ratio", np.nan)),
                    "gc_only_auroc_official": float(gc_lookup["gc_only_auroc_official"]),
                    "gc_only_auroc_matched": float(gc_lookup["gc_only_auroc_matched"]),
                }
            )
    model_rows = pd.DataFrame(rows).sort_values(["task_id", "model_id", "split_id"])
    return confounders, model_rows


def build_mitigation_table(package_summary: pd.DataFrame) -> pd.DataFrame:
    subset = package_summary[
        (package_summary["task_id"].isin(FOCAL_TASKS + ["human_enhancers_ensembl"]))
        & (package_summary["split_id"] == "official")
        & (package_summary["model_id"].isin(["small_cnn", "small_cnn_rc_aug", "dnabert2", "caduceus_ph"]))
    ][
        [
            "task_id",
            "model_id",
            "calibration_method",
            "intervention_id",
            "auroc",
            "ece",
            "brier",
            "rc_mean_abs_delta",
            "mono_positive_prob_drop",
            "dinuc_positive_prob_drop",
            "motif_positive_prob_drop",
        ]
    ].copy()
    subset["task_label"] = subset["task_id"].map(TASK_LABELS)
    subset["model_label"] = subset["model_id"].map(MODEL_LABELS)
    subset["condition"] = np.where(
        subset["calibration_method"] == "temperature",
        "temperature_scaled",
        np.where(
            subset["intervention_id"] == "gc_balanced",
            "gc_balanced",
            np.where(subset["intervention_id"] == "matched_negative_retraining", "matched_negative_retraining", "standard"),
        ),
    )
    matched = package_summary[
        (package_summary["task_id"].isin(FOCAL_TASKS + ["human_enhancers_ensembl"]))
        & (package_summary["split_id"] == "matched_test")
        & (package_summary["model_id"].isin(["small_cnn", "small_cnn_rc_aug", "dnabert2", "caduceus_ph"]))
    ][
        [
            "task_id",
            "model_id",
            "calibration_method",
            "intervention_id",
            "auroc",
            "ece",
            "brier",
        ]
    ].copy()
    subset = subset.merge(
        matched.rename(
            columns={
                "auroc": "matched_auroc",
                "ece": "matched_ece",
                "brier": "matched_brier",
            }
        ),
        on=["task_id", "model_id", "calibration_method", "intervention_id"],
        how="left",
    )
    return subset.sort_values(["task_id", "model_id", "condition"])


def build_real_motif_table(package_summary: pd.DataFrame) -> pd.DataFrame:
    subset = package_summary[
        (package_summary["task_id"].isin(FOCAL_TASKS + ["human_enhancers_ensembl"]))
        & (package_summary["split_id"].isin(["official", "matched_test"]))
        & package_summary["motif_positive_prob_drop"].notna()
    ][
        [
            "task_id",
            "split_id",
            "model_id",
            "calibration_method",
            "intervention_id",
            "auroc",
            "motif_positive_prob_drop",
            "mono_positive_prob_drop",
            "dinuc_positive_prob_drop",
        ]
    ].copy()
    subset["task_label"] = subset["task_id"].map(TASK_LABELS)
    subset["model_label"] = subset["model_id"].map(MODEL_LABELS).fillna(subset["model_id"])
    return subset.sort_values(["task_id", "split_id", "model_id", "calibration_method", "intervention_id"])


def build_foundation_loader_status(package_summary: pd.DataFrame) -> pd.DataFrame:
    path = RESULTS_ROOT / "foundation_loader_status.csv"
    nt_validation_path = RESULTS_ROOT / "nt_validation_report.json"
    nt_validation = {}
    if nt_validation_path.exists():
        nt_validation = json.loads(nt_validation_path.read_text())
    caduceus_rows = package_summary[
        (package_summary["model_id"] == "caduceus_ph")
        & (package_summary["split_id"] == "official")
        & (package_summary["task_id"].isin(CORE_TASKS))
        & (package_summary["calibration_method"] == "none")
        & (package_summary["intervention_id"] == "standard")
    ]
    nt_rows = package_summary[
        (package_summary["model_id"] == "nucleotide_transformer_v2")
        & (package_summary["split_id"] == "official")
        & (package_summary["task_id"].isin(["human_nontata_promoters", "human_enhancers_cohn"]))
        & (package_summary["calibration_method"] == "none")
        & (package_summary["intervention_id"] == "standard")
    ]
    rows = [
        {
            "model_id": "dnabert2",
            "status": "completed",
            "notes": "Frozen embeddings completed on all four core tasks with a lower-data subset protocol.",
        },
        {
            "model_id": "caduceus_ph",
            "status": "completed" if len(caduceus_rows) >= 2 else "blocked_dependency",
            "notes": "Frozen official rows completed in the WSL2/Linux CUDA environment after installing mamba-ssm and a compatible CUDA compiler stack." if len(caduceus_rows) >= 2 else "Frozen-embedding evaluation remains blocked outside the WSL2/Linux CUDA environment because the model loader requires mamba_ssm.",
        },
        {
            "model_id": "nucleotide_transformer_v2",
            "status": "appendix_only_validation_warning" if (len(nt_rows) >= 2 and nt_validation and not nt_validation.get("validated", False)) else ("completed" if len(nt_rows) >= 2 else "blocked_compatibility"),
            "notes": (
                "Frozen official focal-task rows completed, but the validation report shows weak sanity-task behavior under the current mean-pooled frozen protocol; keep NT-v2 appendix-only."
                if (len(nt_rows) >= 2 and nt_validation and not nt_validation.get("validated", False))
                else (
                    "Frozen official focal-task rows completed after compatibility patches for the current Transformers stack."
                    if len(nt_rows) >= 2
                    else "Checkpoint failed under the installed Transformers stack with ESM compatibility errors."
                )
            ),
        },
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def plot_cv_summary(official: pd.DataFrame, cv_summary: pd.DataFrame) -> None:
    configure_matplotlib()
    compare = official[
        (official["task_id"].isin(CORE_TASKS))
        & (official["model_id"].isin(CV_MODEL_ORDER))
        & (official["split_id"] == "official")
        & (official["calibration_method"] == "none")
        & (official["intervention_id"] == "standard")
    ][["task_id", "model_id", "auroc"]].rename(columns={"auroc": "official_auroc"})
    merged = compare.merge(cv_summary[["task_id", "model_id", "auroc_mean", "auroc_std"]], on=["task_id", "model_id"], how="inner")
    merged["task_label"] = merged["task_id"].map(TASK_LABELS)
    merged["model_label"] = merged["model_id"].map(MODEL_LABELS)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.5), sharey=True)
    axes = axes.ravel()
    width = 0.36
    for ax, task_name in zip(axes, CORE_TASKS):
        available = [model_id for model_id in CV_MODEL_ORDER if model_id in set(merged[merged["task_id"] == task_name]["model_id"])]
        subset = merged[merged["task_id"] == task_name].set_index("model_id").loc[available]
        x = np.arange(len(available))
        ax.bar(x - width / 2, subset["official_auroc"], width=width, color="#355070", label="Official")
        ax.bar(x + width / 2, subset["auroc_mean"], width=width, yerr=subset["auroc_std"], color="#e07a5f", label="Chromosome CV", capsize=3)
        ax.set_title(TASK_LABELS[task_name])
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[name] for name in available], rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.18)
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("AUROC")
    axes[2].set_ylabel("AUROC")
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "genomecf_chromosome_cv_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_matched_negative_summary(model_summary: pd.DataFrame) -> None:
    configure_matplotlib()
    subset = model_summary[model_summary["model_id"].isin(["gc_only", "kmer_logistic_regression"])].copy()
    tasks = MATCHED_TASKS
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, task_name in zip(axes, tasks):
        task = subset[subset["task_id"] == task_name].copy()
        x = np.arange(2)
        width = 0.36
        official = task[task["split_id"] == "official"].set_index("model_id")
        matched = task[task["split_id"] == "matched_test"].set_index("model_id")
        ordered = ["gc_only", "kmer_logistic_regression"]
        ax.bar(x - width / 2, official.loc[ordered, "auroc"], width=width, color="#355070", label="Official")
        ax.bar(x + width / 2, matched.loc[ordered, "auroc"], width=width, color="#6d597a", label="Matched")
        ax.set_title(TASK_LABELS[task_name])
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[name] for name in ordered], rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("AUROC")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "genomecf_matched_negative_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_mitigation_summary(mitigation: pd.DataFrame) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.0), sharex=True)
    tasks = FOCAL_TASKS
    condition_order = ["standard", "temperature_scaled", "gc_balanced", "matched_negative_retraining"]
    colors = {
        "standard": "#355070",
        "temperature_scaled": "#2a9d8f",
        "gc_balanced": "#e07a5f",
        "matched_negative_retraining": "#8d99ae",
    }
    metrics = [("ece", "Expected calibration error"), ("mono_positive_prob_drop", "Mononucleotide-shuffle drop")]
    for row_idx, (metric, ylabel) in enumerate(metrics):
        for col_idx, task_name in enumerate(tasks):
            ax = axes[row_idx, col_idx]
            task = mitigation[mitigation["task_id"] == task_name].copy()
            x = np.arange(2)
            width = 0.24
            for offset_idx, condition in enumerate(condition_order):
                subset = task[task["condition"] == condition].set_index("model_id")
                ordered = ["small_cnn", "small_cnn_rc_aug"]
                values = subset.reindex(ordered)[metric].to_numpy()
                ax.bar(x + (offset_idx - 1) * width, values, width=width, color=colors[condition], label=condition if row_idx == 0 else None)
            ax.set_title(TASK_LABELS[task_name] if row_idx == 0 else "")
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS[name] for name in ["small_cnn", "small_cnn_rc_aug"]], rotation=15, ha="right")
            ax.grid(axis="y", alpha=0.18)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
    axes[0, 0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.22), ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "genomecf_mitigation_summary.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    build_release_registry()
    summary = pd.read_csv(RESULTS_ROOT / "benchmark_summary.csv")
    registry = pd.read_csv(RESULTS_ROOT / "benchmark_registry.csv")
    if DISABLE_LOCAL_RUNTIME_ASSETS:
        package_summary = summary.copy()
        print("Using committed release summary for upgrade artifacts because local runtime assets are disabled.")
    else:
        package_registry = pd.read_csv(CURRENT_REGISTRY) if CURRENT_REGISTRY.exists() else pd.DataFrame()
        package_summary = summarize_release_registry(package_registry) if not package_registry.empty else summary.copy()

    cv_summary, cv_folds = build_chromosome_cv_tables(summary, registry)
    matched_confounders, matched_models = build_matched_negative_tables(summary, package_summary)
    mitigation = build_mitigation_table(package_summary)
    motif = build_real_motif_table(package_summary)
    foundation_status = build_foundation_loader_status(package_summary)

    cv_summary.to_csv(RESULTS_ROOT / "chromosome_cv_summary.csv", index=False)
    cv_folds.to_csv(RESULTS_ROOT / "chromosome_cv_fold_metrics.csv", index=False)
    matched_confounders.to_csv(RESULTS_ROOT / "matched_negative_confounders.csv", index=False)
    matched_models.to_csv(RESULTS_ROOT / "matched_negative_model_summary.csv", index=False)
    mitigation.to_csv(RESULTS_ROOT / "mitigation_summary.csv", index=False)
    motif.to_csv(RESULTS_ROOT / "real_task_motif_disruption.csv", index=False)
    foundation_status.to_csv(RESULTS_ROOT / "foundation_loader_status.csv", index=False)

    plot_cv_summary(summary, cv_summary)
    plot_matched_negative_summary(matched_models)
    plot_mitigation_summary(mitigation)

    manifest = {
        "tables": [
            "chromosome_cv_summary.csv",
            "chromosome_cv_fold_metrics.csv",
            "matched_negative_confounders.csv",
            "matched_negative_model_summary.csv",
            "mitigation_summary.csv",
            "real_task_motif_disruption.csv",
            "foundation_loader_status.csv",
        ],
        "figures": [
            "genomecf_chromosome_cv_summary.png",
            "genomecf_matched_negative_summary.png",
            "genomecf_mitigation_summary.png",
        ],
    }
    (RESULTS_ROOT / "upgrade_artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
