from __future__ import annotations

import json
import sys
from pathlib import Path

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


PUBLICATION_ROOT = PROJECT_ROOT / "results" / "publication"
RELEASE_ROOT = PROJECT_ROOT / "results" / "release"
FIGURES_ROOT = PROJECT_ROOT / "figures"
SOURCE_DATA_ROOT = PROJECT_ROOT / "source_data"
SEED = 2026

CORE_TASKS = [
    "human_nontata_promoters",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ocr_ensembl",
]
SCREENING_TASKS = [
    "dummy_mouse_enhancers_ensembl",
    "drosophila_enhancers_stark",
]
REAL_TASKS = CORE_TASKS + SCREENING_TASKS
FOCAL_TASKS = ["human_nontata_promoters", "human_enhancers_cohn"]
MATCHED_TASKS = ["human_nontata_promoters", "human_enhancers_cohn", "human_enhancers_ensembl"]
EXTERNAL_TASKS = [
    "gue_human_tf_0",
    "gue_human_tf_1",
    "gue_emp_h3k4me3",
    "gue_emp_h3k14ac",
    "mpra_bcl11a_enhancer",
    "mpra_f9_promoter",
    "mpra_hbb_promoter",
    "mpra_ldlr_promoter",
    "mpra_myc_enhancer",
]
PAPER_MODELS = [
    "kmer_logistic_regression",
    "small_cnn",
    "small_cnn_rc_aug",
    "dnabert2",
    "caduceus_ph",
]
CV_MODELS = [
    "kmer_logistic_regression",
    "small_cnn",
    "small_cnn_rc_aug",
    "dnabert2",
    "caduceus_ph",
]
SYNTH_MODELS = [
    "kmer_logistic_regression",
    "small_cnn",
    "small_cnn_rc_aug",
    "dnabert2",
    "caduceus_ph",
]
APPENDIX_REAL_MODELS = [
    "gc_only",
    "cpg_only",
    "length_only",
    "repeat_only",
    "kmer_logistic_regression",
    "small_cnn",
    "small_cnn_rc_aug",
    "dnabert2",
    "caduceus_ph",
    "nucleotide_transformer_v2",
]
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
MODEL_FAMILY_EXPORT = {
    "gc_only": "gc_fraction_only_logistic_regression",
    "cpg_only": "cpg_fraction_only_logistic_regression",
    "length_only": "length_only_logistic_regression",
    "repeat_only": "repeat_fraction_only_logistic_regression",
    "kmer_logistic_regression": "kmer_logistic_regression",
    "small_cnn": "small_cnn",
    "small_cnn_rc_aug": "small_cnn_rc_aug",
    "dnabert2": "dnabert2",
    "caduceus_ph": "caduceus_ph",
    "nucleotide_transformer_v2": "nucleotide_transformer_v2",
}
MODEL_MARKERS = {
    "kmer_logistic_regression": "s",
    "small_cnn": "o",
    "small_cnn_rc_aug": "^",
    "dnabert2": "D",
    "caduceus_ph": "X",
    "nucleotide_transformer_v2": "P",
}
MODEL_COLORS = {
    "kmer_logistic_regression": "#184e77",
    "small_cnn": "#2a9d8f",
    "small_cnn_rc_aug": "#e76f51",
    "dnabert2": "#7b2cbf",
    "caduceus_ph": "#4d908e",
    "nucleotide_transformer_v2": "#6a4c93",
}
TASK_LABELS = {
    "human_nontata_promoters": "Promoters",
    "human_enhancers_cohn": "Enhancers (Cohn)",
    "human_enhancers_ensembl": "Enhancers (Ensembl)",
    "human_ocr_ensembl": "Open chromatin",
    "dummy_mouse_enhancers_ensembl": "Mouse enhancers",
    "drosophila_enhancers_stark": "Fly enhancers",
    "gue_human_tf_0": "External TF binding (human_tf_0)",
    "gue_human_tf_1": "External TF binding (human_tf_1)",
    "gue_emp_h3k4me3": "External histone mark (H3K4me3)",
    "gue_emp_h3k14ac": "External histone mark (H3K14ac)",
    "mpra_bcl11a_enhancer": "MPRA variant effect (BCL11A enhancer)",
    "mpra_f9_promoter": "MPRA variant effect (F9 promoter)",
    "mpra_hbb_promoter": "MPRA variant effect (HBB promoter)",
    "mpra_ldlr_promoter": "MPRA variant effect (LDLR promoter)",
    "mpra_myc_enhancer": "MPRA variant effect (MYC enhancer)",
    "gc_correlated": "GC correlated",
    "gc_matched": "GC matched",
    "gc_conflict": "GC shortcut conflict",
    "two_motif_grammar": "Two-motif grammar",
    "motif_position_conflict": "Motif identity vs position shortcut",
}
TASK_SHORT_LABELS = {
    "human_nontata_promoters": "Pr",
    "human_enhancers_cohn": "EC",
    "human_enhancers_ensembl": "EE",
    "human_ocr_ensembl": "OCR",
}
LATEX_TASK_LABELS = {
    "Promoters": r"\shortstack[l]{Promoters}",
    "Enhancers (Cohn)": r"\shortstack[l]{Enhancers\\(Cohn)}",
    "Enhancers (Ensembl)": r"\shortstack[l]{Enhancers\\(Ensembl)}",
    "Open chromatin": r"\shortstack[l]{Open\\chromatin}",
    "Mouse enhancers": r"\shortstack[l]{Mouse\\enhancers}",
    "Fly enhancers": r"\shortstack[l]{Fly\\enhancers}",
    "External TF binding (human_tf_0)": r"\shortstack[l]{External TF\\(human\_tf\_0)}",
    "External TF binding (human_tf_1)": r"\shortstack[l]{External TF\\(human\_tf\_1)}",
    "External histone mark (H3K4me3)": r"\shortstack[l]{External H3K4me3}",
    "External histone mark (H3K14ac)": r"\shortstack[l]{External H3K14ac}",
    "MPRA variant effect (BCL11A enhancer)": r"\shortstack[l]{MPRA BCL11A\\enhancer}",
    "MPRA variant effect (F9 promoter)": r"\shortstack[l]{MPRA F9\\promoter}",
    "MPRA variant effect (HBB promoter)": r"\shortstack[l]{MPRA HBB\\promoter}",
    "MPRA variant effect (LDLR promoter)": r"\shortstack[l]{MPRA LDLR\\promoter}",
    "MPRA variant effect (MYC enhancer)": r"\shortstack[l]{MPRA MYC\\enhancer}",
    "GC correlated": r"\shortstack[l]{GC\\correlated}",
    "GC matched": r"\shortstack[l]{GC\\matched}",
    "GC shortcut conflict": r"\shortstack[l]{GC shortcut\\conflict}",
    "Two-motif grammar": r"\shortstack[l]{Two-motif\\grammar}",
    "Motif identity vs position shortcut": r"\shortstack[l]{Motif identity\\vs position}",
}
LATEX_MODEL_LABELS = {
    "GC-only logistic regression": r"\shortstack[l]{GC-only logistic\\regression}",
    "CpG-only logistic regression": r"\shortstack[l]{CpG-only logistic\\regression}",
    "Length-only logistic regression": r"\shortstack[l]{Length-only logistic\\regression}",
    "Repeat-only logistic regression": r"\shortstack[l]{Repeat-only logistic\\regression}",
    "6-mer logistic regression": r"\shortstack[l]{6-mer logistic\\regression}",
    "CNN": r"\shortstack[l]{CNN}",
    "RC-aug CNN": r"\shortstack[l]{RC-aug CNN}",
    "DNABERT-2": r"\shortstack[l]{DNABERT-2}",
    "Caduceus-Ph": r"\shortstack[l]{Caduceus-Ph}",
    "Nucleotide Transformer v2": r"\shortstack[l]{Nucleotide\\Transformer v2}",
}
LATEX_INTERVENTION_LABELS = {
    "Temperature scaling": r"\shortstack[l]{Temperature\\scaling}",
    "GC-balanced training": r"\shortstack[l]{GC-balanced\\training}",
    "Matched-negative retraining": r"\shortstack[l]{Matched-negative\\retraining}",
}


def latex_escape(text: object) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def fmt_count(value: int | float | None) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{int(value):,}"


def task_label_tex(text: object) -> str:
    label = str(text)
    return LATEX_TASK_LABELS.get(label, latex_escape(label))


def model_label_tex(text: object) -> str:
    label = str(text)
    return LATEX_MODEL_LABELS.get(label, latex_escape(label))


def intervention_label_tex(text: object) -> str:
    label = str(text)
    return LATEX_INTERVENTION_LABELS.get(label, latex_escape(label))


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


def load_release_inputs() -> dict[str, pd.DataFrame]:
    paths = {
        "registry": RELEASE_ROOT / "benchmark_registry.csv",
        "summary": RELEASE_ROOT / "benchmark_summary.csv",
        "cv_summary": RELEASE_ROOT / "chromosome_cv_summary.csv",
        "cv_folds": RELEASE_ROOT / "chromosome_cv_fold_metrics.csv",
        "matched_confounders": RELEASE_ROOT / "matched_negative_confounders.csv",
        "matched_models": RELEASE_ROOT / "matched_negative_model_summary.csv",
        "mitigation": RELEASE_ROOT / "mitigation_summary.csv",
        "motif": RELEASE_ROOT / "real_task_motif_disruption.csv",
        "motif_probe_summary": RELEASE_ROOT / "real_motif_probe_summary.csv",
        "gc_bin_summary": RELEASE_ROOT / "gc_bin_summary.csv",
        "gc_bin_by_bin": RELEASE_ROOT / "gc_bin_by_bin.csv",
        "external_gc_bin_summary": RELEASE_ROOT / "external_gc_bin_summary.csv",
        "external_gc_bin_by_bin": RELEASE_ROOT / "external_gc_bin_by_bin.csv",
        "external_validation_summary": RELEASE_ROOT / "external_validation_summary.csv",
        "external_validation_family_summary": RELEASE_ROOT / "external_validation_family_summary.csv",
        "external_transfer_prediction": RELEASE_ROOT / "external_transfer_prediction.csv",
        "external_prediction_robustness": RELEASE_ROOT / "external_prediction_robustness.csv",
        "external_case_study": RELEASE_ROOT / "biological_case_study.csv",
        "synthetic_extended": RELEASE_ROOT / "synthetic_extended_summary.csv",
        "foundation_status": RELEASE_ROOT / "foundation_loader_status.csv",
    }
    return {name: pd.read_csv(path) for name, path in paths.items()}


def official_standard_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        (summary["split_id"] == "official")
        & (summary["calibration_method"] == "none")
        & (summary["intervention_id"] == "standard")
    ].copy()


def build_task_overview_table(summary: pd.DataFrame) -> pd.DataFrame:
    official_spec = get_split_spec("official")
    official = official_standard_rows(summary)
    rows: list[dict[str, str]] = []
    for task_name in REAL_TASKS:
        task_spec = get_task_spec(task_name)
        frame = load_task_frame(task_spec, PROJECT_ROOT)
        splits = build_split_frames(frame, official_spec, seed=SEED)
        full_train = int((frame["orig_split"] == "train").sum())
        full_test = int((frame["orig_split"] == "test").sum())
        eval_test = splits["test"]
        pos = int((eval_test["label"] == 1).sum())
        neg = int((eval_test["label"] == 0).sum())
        length_range = task_spec.sequence_length_range or [int(frame["length"].min()), int(frame["length"].max())]
        length_text = (
            f"Fixed ({length_range[0]})"
            if len(length_range) == 2 and length_range[0] == length_range[1]
            else f"Variable ({length_range[0]}-{length_range[1]})"
        )
        coverage_models = set(official[(official["task_id"] == task_name) & (official["model_id"].isin(PAPER_MODELS))]["model_id"].tolist())
        if task_name in CORE_TASKS and "caduceus_ph" in coverage_models:
            coverage = "Diag., 6-mer, CNNs, DNABERT-2, Caduceus"
        else:
            coverage = "6-mer"
        rows.append(
            {
                "tier": "Core tasks" if task_name in CORE_TASKS else "Extended screening tasks",
                "task_id": task_name,
                "task_label": TASK_LABELS[task_name],
                "species": task_spec.species.title(),
                "length_type": length_text,
                "full_counts": f"{fmt_count(full_train)} / {fmt_count(full_test)}",
                "eval_counts": f"{fmt_count(len(splits['train']))} / {fmt_count(len(splits['validation']))} / {fmt_count(len(splits['test']))}",
                "class_balance": f"{pos}:{neg}",
                "chrom_meta": "Yes" if task_spec.species == "human" else "No",
                "coverage": coverage,
            }
        )
    return pd.DataFrame(rows)


def build_main_results_table(summary: pd.DataFrame) -> pd.DataFrame:
    official = official_standard_rows(summary)
    subset = official[
        (official["task_id"].isin(CORE_TASKS))
        & (official["model_id"].isin(PAPER_MODELS))
    ].copy()
    subset["task_label"] = subset["task_id"].map(TASK_LABELS)
    subset["model_label"] = subset["model_id"].map(MODEL_LABELS)
    subset["model_family"] = subset["model_id"].map(MODEL_FAMILY_EXPORT)
    subset = subset[
        [
            "task_id",
            "task_label",
            "model_id",
            "model_family",
            "model_label",
            "auroc",
            "ece",
            "brier",
            "rc_mean_abs_delta",
            "rc_flip_rate",
            "mono_positive_prob_drop",
            "dinuc_positive_prob_drop",
        ]
    ].rename(
        columns={
            "task_id": "dataset",
            "auroc": "ensemble_auroc",
            "ece": "ensemble_ece",
            "brier": "ensemble_brier",
            "rc_mean_abs_delta": "ensemble_reverse_complement_mean_abs_delta",
            "rc_flip_rate": "ensemble_reverse_complement_flip_rate",
            "mono_positive_prob_drop": "ensemble_mono_shuffle_positive_prob_drop",
            "dinuc_positive_prob_drop": "ensemble_dinuc_shuffle_positive_prob_drop",
        }
    )
    subset["order_task"] = subset["dataset"].map({name: idx for idx, name in enumerate(CORE_TASKS)})
    subset["order_model"] = subset["model_id"].map({name: idx for idx, name in enumerate(PAPER_MODELS)})
    subset = subset.sort_values(["order_task", "order_model"]).drop(columns=["order_task", "order_model", "model_id"])
    return subset


def build_cv_main_table(summary: pd.DataFrame, cv_summary: pd.DataFrame) -> pd.DataFrame:
    official = official_standard_rows(summary)
    official = official[
        ((official["task_id"].isin(CORE_TASKS)) | (official["task_id"].isin(FOCAL_TASKS)))
        & (official["model_id"].isin(CV_MODELS))
    ][["task_id", "model_id", "auroc"]].rename(columns={"auroc": "official_auroc"})
    merged = official.merge(cv_summary, on=["task_id", "model_id"], how="inner")
    merged["task_label"] = merged["task_id"].map(TASK_LABELS)
    merged["model_label"] = merged["model_id"].map(MODEL_LABELS)
    return merged[
        [
            "task_id",
            "task_label",
            "model_id",
            "model_label",
            "official_auroc",
            "auroc_mean",
            "auroc_std",
            "ece_mean",
            "brier_mean",
            "rc_mean_abs_delta_mean",
            "mono_positive_prob_drop_mean",
            "dinuc_positive_prob_drop_mean",
        ]
    ].sort_values(["task_id", "model_id"])


def build_appendix_real_results(summary: pd.DataFrame) -> pd.DataFrame:
    official = official_standard_rows(summary)
    subset = official[
        (official["task_id"].isin(REAL_TASKS))
        & (official["model_id"].isin(APPENDIX_REAL_MODELS))
    ].copy()
    subset["task_label"] = subset["task_id"].map(TASK_LABELS)
    subset["model_label"] = subset["model_id"].map(MODEL_LABELS)
    subset["tier"] = np.where(subset["task_id"].isin(CORE_TASKS), "Core", "Screening")
    subset["order_task"] = subset["task_id"].map({name: idx for idx, name in enumerate(REAL_TASKS)})
    subset["order_model"] = subset["model_id"].map({name: idx for idx, name in enumerate(APPENDIX_REAL_MODELS)})
    return subset.sort_values(["order_task", "order_model"]).drop(columns=["order_task", "order_model"])


def build_synthetic_appendix(summary: pd.DataFrame) -> pd.DataFrame:
    subset = summary[(summary["tier"] == "synthetic") & (summary["model_id"].isin(SYNTH_MODELS))].copy()
    subset["condition"] = subset["task_id"].map(TASK_LABELS)
    subset["model_label"] = subset["model_id"].map(MODEL_LABELS)
    subset["order_task"] = subset["task_id"].map({"gc_correlated": 0, "gc_matched": 1})
    subset["order_model"] = subset["model_id"].map({name: idx for idx, name in enumerate(SYNTH_MODELS)})
    return subset.sort_values(["order_task", "order_model"]).drop(columns=["order_task", "order_model"])


def build_official_confounder_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    official_spec = get_split_spec("official")
    confounder_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for task_name in REAL_TASKS:
        frame = load_task_frame(get_task_spec(task_name), PROJECT_ROOT)
        test_df = build_split_frames(frame, official_spec, seed=SEED)["test"].copy()
        row: dict[str, object] = {"task_id": task_name, "task_label": TASK_LABELS[task_name]}
        for label_value, prefix in [(0, "negative"), (1, "positive")]:
            subset = test_df[test_df["label"] == label_value]
            row[f"{prefix}_n"] = int(len(subset))
            for column in ["gc_fraction", "cpg_oe", "length", "repeat_fraction", "n_fraction"]:
                row[f"{prefix}_{column}_mean"] = float(subset[column].mean())
        confounder_rows.append(row)
        if get_task_spec(task_name).species == "human":
            counts = test_df["chromosome_fold"].value_counts().to_dict()
            fold_rows.append(
                {
                    "task_id": task_name,
                    "task_label": TASK_LABELS[task_name],
                    "A": int(counts.get("A", 0)),
                    "B": int(counts.get("B", 0)),
                    "C": int(counts.get("C", 0)),
                    "D": int(counts.get("D", 0)),
                    "E": int(counts.get("E", 0)),
                }
            )
    return pd.DataFrame(confounder_rows), pd.DataFrame(fold_rows)


def build_matched_negative_results(matched_models: pd.DataFrame, matched_confounders: pd.DataFrame) -> pd.DataFrame:
    requested_models = ["gc_only", "kmer_logistic_regression", "small_cnn", "small_cnn_rc_aug", "dnabert2", "caduceus_ph", "nucleotide_transformer_v2"]
    rows: list[dict[str, object]] = []
    for task_name in ["human_nontata_promoters", "human_enhancers_cohn", "human_enhancers_ensembl"]:
        gc_lookup = matched_confounders[matched_confounders["task_id"] == task_name].iloc[0]
        for model_id in requested_models:
            official_subset = matched_models[(matched_models["task_id"] == task_name) & (matched_models["model_id"] == model_id) & (matched_models["split_id"] == "official")]
            matched_subset = matched_models[(matched_models["task_id"] == task_name) & (matched_models["model_id"] == model_id) & (matched_models["split_id"] == "matched_test")]
            if official_subset.empty or matched_subset.empty:
                continue
            official = official_subset.iloc[0]
            matched = matched_subset.iloc[0]
            rows.append(
                {
                    "task_id": task_name,
                    "task_label": TASK_LABELS[task_name],
                    "model_id": model_id,
                    "model_label": MODEL_LABELS[model_id],
                    "original_auroc": float(official["auroc"]),
                    "matched_auroc": float(matched["auroc"]),
                    "original_gc_only_auroc": float(gc_lookup["gc_only_auroc_official"]),
                    "matched_gc_only_auroc": float(gc_lookup["gc_only_auroc_matched"]),
                    "original_ece": float(official["ece"]),
                    "matched_ece": float(matched["ece"]),
                    "original_mono_drop": float(official["mono_positive_prob_drop"]),
                    "matched_mono_drop": float(matched["mono_positive_prob_drop"]),
                    "original_dinuc_drop": float(official["dinuc_positive_prob_drop"]),
                    "matched_dinuc_drop": float(matched["dinuc_positive_prob_drop"]),
                }
            )
    return pd.DataFrame(rows)


def build_matched_negative_confounder_table(matched_confounders: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in matched_confounders.to_dict(orient="records"):
        rows.append(
            {
                "task_id": row["task_id"],
                "task_label": row["task_label"],
                "gc_before_after": f"{row['official_negative_gc_fraction_mean']:.3f}/{row['official_positive_gc_fraction_mean']:.3f} -> {row['matched_negative_gc_fraction_mean']:.3f}/{row['matched_positive_gc_fraction_mean']:.3f}",
                "cpg_before_after": f"{row['official_negative_cpg_oe_mean']:.3f}/{row['official_positive_cpg_oe_mean']:.3f} -> {row['matched_negative_cpg_oe_mean']:.3f}/{row['matched_positive_cpg_oe_mean']:.3f}",
                "length_before_after": f"{row['official_negative_length_mean']:.1f}/{row['official_positive_length_mean']:.1f} -> {row['matched_negative_length_mean']:.1f}/{row['matched_positive_length_mean']:.1f}",
                "repeat_before_after": f"{row['official_negative_repeat_fraction_mean']:.3f}/{row['official_positive_repeat_fraction_mean']:.3f} -> {row['matched_negative_repeat_fraction_mean']:.3f}/{row['matched_positive_repeat_fraction_mean']:.3f}",
                "n_before_after": f"{row['official_negative_n_fraction_mean']:.3f}/{row['official_positive_n_fraction_mean']:.3f} -> {row['matched_negative_n_fraction_mean']:.3f}/{row['matched_positive_n_fraction_mean']:.3f}",
                "gc_only_shift": f"{row['gc_only_auroc_official']:.3f} -> {row['gc_only_auroc_matched']:.3f}",
            }
        )
    return pd.DataFrame(rows)


def build_mitigation_results(mitigation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task_name in FOCAL_TASKS + ["human_enhancers_ensembl"]:
        for model_id in ["small_cnn", "small_cnn_rc_aug", "dnabert2", "caduceus_ph"]:
            subset = mitigation[(mitigation["task_id"] == task_name) & (mitigation["model_id"] == model_id)].copy()
            if subset.empty or "standard" not in set(subset["condition"]):
                continue
            baseline = subset[subset["condition"] == "standard"].iloc[0]
            for condition in ["temperature_scaled", "gc_balanced", "matched_negative_retraining"]:
                if condition not in set(subset["condition"]):
                    continue
                variant = subset[subset["condition"] == condition].iloc[0]
                rows.append(
                    {
                        "task_id": task_name,
                        "task_label": TASK_LABELS[task_name],
                        "model_id": model_id,
                        "model_label": MODEL_LABELS[model_id],
                        "intervention": (
                            "Temperature scaling"
                            if condition == "temperature_scaled"
                            else ("GC-balanced training" if condition == "gc_balanced" else "Matched-negative retraining")
                        ),
                        "auroc_before": float(baseline["auroc"]),
                        "auroc_after": float(variant["auroc"]),
                        "ece_before": float(baseline["ece"]),
                        "ece_after": float(variant["ece"]),
                        "brier_before": float(baseline["brier"]),
                        "brier_after": float(variant["brier"]),
                        "matched_auroc_before": float(baseline.get("matched_auroc", np.nan)),
                        "matched_auroc_after": float(variant.get("matched_auroc", np.nan)),
                        "matched_ece_before": float(baseline.get("matched_ece", np.nan)),
                        "matched_ece_after": float(variant.get("matched_ece", np.nan)),
                        "matched_brier_before": float(baseline.get("matched_brier", np.nan)),
                        "matched_brier_after": float(variant.get("matched_brier", np.nan)),
                        "rc_before": float(baseline["rc_mean_abs_delta"]),
                        "rc_after": float(variant["rc_mean_abs_delta"]),
                        "mono_before": float(baseline["mono_positive_prob_drop"]),
                        "mono_after": float(variant["mono_positive_prob_drop"]),
                        "dinuc_before": float(baseline["dinuc_positive_prob_drop"]),
                        "dinuc_after": float(variant["dinuc_positive_prob_drop"]),
                    }
                )
    return pd.DataFrame(rows)


def build_real_motif_results(motif_probe_summary: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    official = official_standard_rows(summary)[
        ["task_id", "model_id", "mono_positive_prob_drop", "dinuc_positive_prob_drop"]
    ].copy()
    subset = motif_probe_summary[
        motif_probe_summary["task_id"].isin(["human_nontata_promoters", "human_enhancers_cohn", "human_enhancers_ensembl"])
        & motif_probe_summary["model_id"].isin(["kmer_logistic_regression", "small_cnn", "small_cnn_rc_aug", "dnabert2", "caduceus_ph"])
    ].copy()
    subset = subset.merge(official, on=["task_id", "model_id"], how="left")
    subset["task_label"] = subset["task_id"].map(TASK_LABELS)
    subset["model_label"] = subset["model_id"].map(MODEL_LABELS)
    subset["order_task"] = subset["task_id"].map({name: idx for idx, name in enumerate(["human_nontata_promoters", "human_enhancers_cohn", "human_enhancers_ensembl"])})
    subset["order_model"] = subset["model_id"].map({"kmer_logistic_regression": 0, "small_cnn": 1, "small_cnn_rc_aug": 2, "dnabert2": 3, "caduceus_ph": 4})
    return subset.sort_values(["order_task", "order_model"]).drop(columns=["order_task", "order_model"])


def build_gc_bin_results(gc_bin_summary: pd.DataFrame) -> pd.DataFrame:
    subset = gc_bin_summary[
        gc_bin_summary["task_id"].isin(CORE_TASKS)
        & gc_bin_summary["model_id"].isin(PAPER_MODELS)
    ].copy()
    subset["task_label"] = subset["task_id"].map(TASK_LABELS)
    subset["model_label"] = subset["model_id"].map(MODEL_LABELS)
    subset["order_task"] = subset["task_id"].map({name: idx for idx, name in enumerate(CORE_TASKS)})
    subset["order_model"] = subset["model_id"].map({name: idx for idx, name in enumerate(PAPER_MODELS)})
    return subset.sort_values(["order_task", "order_model"]).drop(columns=["order_task", "order_model"])


def build_synthetic_extended_results(synthetic_extended: pd.DataFrame) -> pd.DataFrame:
    subset = synthetic_extended[
        synthetic_extended["model_id"].isin(["kmer_logistic_regression", "small_cnn", "small_cnn_rc_aug", "dnabert2", "caduceus_ph"])
    ].copy()
    subset["task_label"] = subset["task_id"].map(TASK_LABELS)
    subset["model_label"] = subset["model_id"].map(MODEL_LABELS)
    subset["order_task"] = subset["task_id"].map({name: idx for idx, name in enumerate(["gc_conflict", "two_motif_grammar", "motif_position_conflict"])})
    subset["order_model"] = subset["model_id"].map({"kmer_logistic_regression": 0, "small_cnn": 1, "small_cnn_rc_aug": 2, "dnabert2": 3, "caduceus_ph": 4})
    return subset.sort_values(["order_task", "order_model"]).drop(columns=["order_task", "order_model"])


def build_external_validation_results(external_family_summary: pd.DataFrame) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product(
        [["TF binding", "Histone marks", "Variant effect"], PAPER_MODELS],
        names=["external_family", "model_id"],
    ).to_frame(index=False)
    subset = grid.merge(external_family_summary, on=["external_family", "model_id"], how="left")
    subset["model_label"] = subset["model_id"].map(MODEL_LABELS)
    subset["family_label"] = subset["external_family"]
    subset["order_family"] = subset["external_family"].map({"TF binding": 0, "Histone marks": 1, "Variant effect": 2})
    subset["order_model"] = subset["model_id"].map({name: idx for idx, name in enumerate(PAPER_MODELS)})
    return subset.sort_values(["order_family", "order_model"]).drop(columns=["order_family", "order_model"])


def build_external_prediction_summary(transfer_points: pd.DataFrame, stats_payload: dict[str, object]) -> pd.DataFrame:
    regression = pd.DataFrame(stats_payload.get("regression", []))
    lofo = pd.DataFrame(stats_payload.get("leave_one_family_out", []))
    family_stratified = pd.DataFrame(stats_payload.get("family_stratified_regression", []))
    predictor_labels = {
        "core_mean_auroc": "Held-out core AUROC model",
        "core_mean_shortcut_score": "GenomeCF Shortcut Score model",
        "core_mean_auroc+core_mean_shortcut_score": "Combined AUROC + Shortcut Score model",
        "core_mean_auroc+core_mean_rc_delta+core_mean_ece+core_matched_negative_shift+core_gc_bin_auroc_gap": "Full GenomeCF profile model",
    }
    rows = [
        {
            "analysis": "Core AUROC vs external biological reliability",
            "metric": "Spearman",
            "value": float(stats_payload["auroc_vs_external_reliability"]["value"]),
            "ci_low": float(stats_payload["auroc_vs_external_reliability"]["ci_low"]),
            "ci_high": float(stats_payload["auroc_vs_external_reliability"]["ci_high"]),
            "n": int(stats_payload["auroc_vs_external_reliability"]["n"]),
        },
        {
            "analysis": "GenomeCF Shortcut Score vs external biological reliability",
            "metric": "Spearman",
            "value": float(stats_payload["shortcut_vs_external_reliability"]["value"]),
            "ci_low": float(stats_payload["shortcut_vs_external_reliability"]["ci_low"]),
            "ci_high": float(stats_payload["shortcut_vs_external_reliability"]["ci_high"]),
            "n": int(stats_payload["shortcut_vs_external_reliability"]["n"]),
        },
        {
            "analysis": "Core AUROC vs external reliability risk",
            "metric": "Spearman",
            "value": float(stats_payload["auroc_vs_external_risk"]["value"]),
            "ci_low": float(stats_payload["auroc_vs_external_risk"]["ci_low"]),
            "ci_high": float(stats_payload["auroc_vs_external_risk"]["ci_high"]),
            "n": int(stats_payload["auroc_vs_external_risk"]["n"]),
        },
        {
            "analysis": "GenomeCF Shortcut Score vs external reliability risk",
            "metric": "Spearman",
            "value": float(stats_payload["shortcut_vs_external_risk"]["value"]),
            "ci_low": float(stats_payload["shortcut_vs_external_risk"]["ci_low"]),
            "ci_high": float(stats_payload["shortcut_vs_external_risk"]["ci_high"]),
            "n": int(stats_payload["shortcut_vs_external_risk"]["n"]),
        },
        {
            "analysis": "Core AUROC vs external matched-negative shift",
            "metric": "Pearson",
            "value": float(stats_payload["auroc_vs_external_matched_shift"]["value"]),
            "ci_low": float(stats_payload["auroc_vs_external_matched_shift"]["ci_low"]),
            "ci_high": float(stats_payload["auroc_vs_external_matched_shift"]["ci_high"]),
            "n": int(stats_payload["auroc_vs_external_matched_shift"]["n"]),
        },
        {
            "analysis": "GenomeCF Shortcut Score vs external matched-negative shift",
            "metric": "Pearson",
            "value": float(stats_payload["shortcut_vs_external_matched_shift"]["value"]),
            "ci_low": float(stats_payload["shortcut_vs_external_matched_shift"]["ci_low"]),
            "ci_high": float(stats_payload["shortcut_vs_external_matched_shift"]["ci_high"]),
            "n": int(stats_payload["shortcut_vs_external_matched_shift"]["n"]),
        },
    ]
    for _, row in regression.iterrows():
        rows.append(
            {
                "analysis": predictor_labels.get(row["predictors"], str(row["predictors"])),
                "metric": "R^2",
                "value": float(row["r2"]),
                "ci_low": float(row.get("ci_low", np.nan)),
                "ci_high": float(row.get("ci_high", np.nan)),
                "n": int(row["n"]),
            }
        )
    for _, row in family_stratified.iterrows():
        rows.append(
            {
                "analysis": f"{row['external_family']} family: {predictor_labels.get(row['predictors'], str(row['predictors']))}",
                "metric": "Family-stratified R^2",
                "value": float(row["r2"]),
                "ci_low": float(row.get("ci_low", np.nan)),
                "ci_high": float(row.get("ci_high", np.nan)),
                "n": int(row["n"]),
            }
        )
    for _, row in lofo.iterrows():
        rows.append(
            {
                "analysis": predictor_labels.get(row["predictors"], str(row["predictors"])),
                "metric": "Leave-one-family-out R^2",
                "value": float(row["cv_r2"]),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n": int(sum(item["n"] for item in row["folds"])),
            }
        )
    full_profile_advantage = stats_payload.get("full_profile_advantage", {})
    rows.append(
        {
            "analysis": "Full GenomeCF profile vs AUROC model",
            "metric": "Delta R^2",
            "value": float(full_profile_advantage.get("observed_delta", np.nan)),
            "ci_low": float(full_profile_advantage.get("ci_low", np.nan)),
            "ci_high": float(full_profile_advantage.get("ci_high", np.nan)),
            "n": int(stats_payload.get("pair_count", len(transfer_points))),
        }
    )
    shortcut_permutation = stats_payload.get("shortcut_permutation", {})
    rows.append(
        {
            "analysis": "Shortcut Score vs AUROC leave-one-family-out advantage",
            "metric": "Permutation p",
            "value": float(shortcut_permutation.get("p_value", np.nan)),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": int(shortcut_permutation.get("n_perm", 0)),
        }
    )
    full_profile_permutation = stats_payload.get("permutation", {})
    rows.append(
        {
            "analysis": "Full GenomeCF profile vs AUROC leave-one-family-out advantage",
            "metric": "Permutation p",
            "value": float(full_profile_permutation.get("p_value", np.nan)),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": int(full_profile_permutation.get("n_perm", 0)),
        }
    )
    return pd.DataFrame(rows)


def build_case_study_results(case_study: pd.DataFrame) -> pd.DataFrame:
    subset = case_study.copy()
    subset["model_label"] = subset["model_id"].map(MODEL_LABELS)
    subset["order_case"] = subset["case_study_id"].map({"Case A": 0, "Case B": 1}).fillna(9)
    subset["order_model"] = subset["model_id"].map({"small_cnn": 0, "dnabert2": 1, "kmer_logistic_regression": 2}).fillna(9)
    subset["order_variant"] = subset["condition_label"].map({"Standard": 0, "Temperature-scaled": 1, "Matched-negative head": 2}).fillna(9)
    return subset.sort_values(["order_case", "order_model", "order_variant"]).drop(columns=["order_case", "order_model", "order_variant"])


def build_external_appendix_results(external_validation_summary: pd.DataFrame) -> pd.DataFrame:
    subset = external_validation_summary[external_validation_summary["model_id"].isin(PAPER_MODELS)].copy()
    subset["task_label"] = subset["task_id"].map(TASK_LABELS)
    subset["model_label"] = subset["model_id"].map(MODEL_LABELS)
    subset["order_task"] = subset["task_id"].map({name: idx for idx, name in enumerate(EXTERNAL_TASKS)})
    subset["order_model"] = subset["model_id"].map({name: idx for idx, name in enumerate(PAPER_MODELS)})
    return subset.sort_values(["order_task", "order_model"]).drop(columns=["order_task", "order_model"])


def build_shortcut_score_results(
    summary: pd.DataFrame,
    matched_results: pd.DataFrame,
    gc_bin_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    official = official_standard_rows(summary)
    subset = official[
        official["task_id"].isin(MATCHED_TASKS)
        & official["model_id"].isin(PAPER_MODELS)
    ][
        [
            "task_id",
            "model_id",
            "auroc",
            "rc_mean_abs_delta",
            "rc_flip_rate",
            "mono_positive_prob_drop",
            "dinuc_positive_prob_drop",
            "mono_calibration_shift",
            "gc_only_explainability_ratio",
        ]
    ].copy()
    matched_drop = matched_results[["task_id", "model_id", "original_auroc", "matched_auroc"]].copy()
    matched_drop["matched_auroc_drop"] = matched_drop["original_auroc"] - matched_drop["matched_auroc"]
    subset = subset.merge(
        matched_drop[["task_id", "model_id", "matched_auroc_drop"]],
        on=["task_id", "model_id"],
        how="left",
    ).merge(
        gc_bin_results[["task_id", "model_id", "gc_bin_auroc_gap"]],
        on=["task_id", "model_id"],
        how="left",
    )
    subset["mono_badness"] = -subset["mono_positive_prob_drop"]
    subset["dinuc_badness"] = -subset["dinuc_positive_prob_drop"]
    component_map = {
        "rc_rank": "rc_mean_abs_delta",
        "rc_flip_rank": "rc_flip_rate",
        "mono_rank": "mono_badness",
        "dinuc_rank": "dinuc_badness",
        "calibration_rank": "mono_calibration_shift",
        "gc_bin_rank": "gc_bin_auroc_gap",
        "matched_rank": "matched_auroc_drop",
        "gc_ratio_rank": "gc_only_explainability_ratio",
    }
    ranked_frames: list[pd.DataFrame] = []
    for task_id, group in subset.groupby("task_id", as_index=False):
        task = group.copy()
        for rank_name, column in component_map.items():
            task[rank_name] = task[column].rank(method="average", ascending=True)
        task["shortcut_score"] = task[list(component_map.keys())].mean(axis=1)
        task["auroc_rank"] = task["auroc"].rank(method="average", ascending=True)
        ranked_frames.append(task)
    ranked = pd.concat(ranked_frames, ignore_index=True)
    ranked["task_label"] = ranked["task_id"].map(TASK_LABELS)
    ranked["model_label"] = ranked["model_id"].map(MODEL_LABELS)
    model_summary = (
        ranked.groupby(["model_id", "model_label"], as_index=False)
        .agg(
            mean_auroc=("auroc", "mean"),
            mean_auroc_rank=("auroc_rank", "mean"),
            mean_shortcut_score=("shortcut_score", "mean"),
        )
        .sort_values("mean_auroc_rank", ascending=False)
    )
    return ranked, model_summary


def save_dataframe(frame: pd.DataFrame, stem: str) -> None:
    PUBLICATION_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PUBLICATION_ROOT / f"{stem}.csv", index=False)


def write_tabular(path: Path, columns: list[str], rows: list[list[str]], alignment: str) -> None:
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\toprule", " & ".join(columns) + " \\\\", "\\midrule"]
    lines.extend(" & ".join(row) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def write_task_overview_tex(frame: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{@{}llllll@{}}",
        "\\toprule",
        "Task & Species & Length & Eval train/val/test & Chr. meta & Main coverage \\\\",
        "\\midrule",
    ]
    current_tier = None
    for row in frame.to_dict(orient="records"):
        if row["tier"] != current_tier:
            lines.append(f"\\multicolumn{{6}}{{l}}{{\\textbf{{{latex_escape(row['tier'])}}}}} \\\\")
            current_tier = row["tier"]
        task_cell = latex_escape(row["task_label"])
        coverage_cell = "All main models" if "DNABERT-2, Caduceus" in str(row["coverage"]) else "6-mer only"
        lines.append(
            " & ".join(
                [
                    task_cell,
                    latex_escape(row["species"]),
                    latex_escape(row["length_type"]),
                    latex_escape(row["eval_counts"]),
                    latex_escape(row["chrom_meta"]),
                    coverage_cell,
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (PUBLICATION_ROOT / "table1_task_overview.tex").write_text("\n".join(lines) + "\n")


def write_main_results_tex(frame: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            [
                task_label_tex(row["task_label"]),
                model_label_tex(row["model_label"]),
                fmt(float(row["ensemble_auroc"])),
                fmt(float(row["ensemble_ece"])),
                fmt(float(row["ensemble_brier"])),
                fmt(float(row["ensemble_reverse_complement_mean_abs_delta"])),
                fmt(float(row["ensemble_reverse_complement_flip_rate"])),
                fmt(float(row["ensemble_mono_shuffle_positive_prob_drop"])),
                fmt(float(row["ensemble_dinuc_shuffle_positive_prob_drop"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "table2_main_results.tex",
        ["Task", "Model", "AUROC", "ECE", "Brier", "RC $\\Delta$", "RC flip", "Mono drop", "Dinuc drop"],
        rows,
        "p{3.0cm}p{2.2cm}rrrrrrr",
    )


def write_cv_main_tex(frame: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            [
                task_label_tex(row["task_label"]),
                model_label_tex(row["model_label"]),
                fmt(float(row["official_auroc"])),
                fmt(float(row["auroc_mean"])),
                fmt(float(row["auroc_std"])),
                fmt(float(row["ece_mean"])),
                fmt(float(row["brier_mean"])),
                fmt(float(row["rc_mean_abs_delta_mean"])),
                fmt(float(row["mono_positive_prob_drop_mean"])),
                fmt(float(row["dinuc_positive_prob_drop_mean"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "table3_cv_summary.tex",
        ["Task", "Model", "Official", "CV mean", "CV SD", "ECE", "Brier", "RC $\\Delta$", "Mono drop", "Dinuc drop"],
        rows,
        "p{2.3cm}p{2.1cm}rrrrrrrr",
    )


def write_matched_negative_main_tex(frame: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            [
                task_label_tex(row["task_label"]),
                model_label_tex(row["model_label"]),
                fmt(float(row["original_auroc"])),
                fmt(float(row["matched_auroc"])),
                fmt(float(row["original_gc_only_auroc"])),
                fmt(float(row["matched_gc_only_auroc"])),
                fmt(float(row["original_mono_drop"])),
                fmt(float(row["matched_mono_drop"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "table4_matched_negative_summary.tex",
        ["Task", "Model", "Orig. AUROC", "Matched AUROC", "Orig. GC-only", "Matched GC-only", "Orig. mono", "Matched mono"],
        rows,
        "p{2.15cm}p{2.0cm}rrrrrr",
    )


def write_mitigation_main_tex(frame: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            [
                task_label_tex(row["task_label"]),
                model_label_tex(row["model_label"]),
                intervention_label_tex(row["intervention"]),
                fmt(float(row["auroc_before"])),
                fmt(float(row["auroc_after"])),
                fmt(float(row["ece_before"])),
                fmt(float(row["ece_after"])),
                fmt(float(row["brier_before"])),
                fmt(float(row["brier_after"])),
                fmt(float(row["mono_before"])),
                fmt(float(row["mono_after"])),
                fmt(float(row["matched_auroc_before"])),
                fmt(float(row["matched_auroc_after"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "table5_mitigation_summary.tex",
        ["Task", "Model", "Intervention", "AUROC before", "AUROC after", "ECE before", "ECE after", "Brier before", "Brier after", "Mono before", "Mono after", "Matched AUROC before", "Matched AUROC after"],
        rows,
        "p{1.8cm}p{1.7cm}p{1.9cm}rrrrrrrrrr",
    )


def write_gc_bin_main_tex(frame: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            [
                task_label_tex(row["task_label"]),
                model_label_tex(row["model_label"]),
                fmt(float(row["overall_auroc"])),
                fmt(float(row["worst_bin_auroc"])),
                fmt(float(row["gc_bin_auroc_gap"])),
                fmt(float(row["overall_ece"])),
                fmt(float(row["worst_bin_ece"])),
                fmt(float(row["gc_bin_ece_gap"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "table6_gc_bin_summary.tex",
        ["Task", "Model", "Overall AUROC", "Worst-bin AUROC", "AUROC gap", "Overall ECE", "Worst-bin ECE", "ECE gap"],
        rows,
        "p{2.15cm}p{2.0cm}rrrrrr",
    )


def write_motif_main_tex(frame: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            [
                task_label_tex(row["task_label"]),
                model_label_tex(row["model_label"]),
                fmt(int(row["evaluated_count"]), digits=0),
                fmt(float(row["motif_drop"])),
                fmt(float(row["gc_preserving_motif_drop"])),
                fmt(float(row["random_edit_drop"])),
                fmt(float(row["motif_minus_random"])),
                fmt(float(row["mono_positive_prob_drop"])),
                fmt(float(row["dinuc_positive_prob_drop"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "table7_motif_summary.tex",
        ["Task", "Model", "n", "Motif drop", "GC-preserving", "Random edit", "Motif$-$random", "Mono drop", "Dinuc drop"],
        rows,
        "p{2.1cm}p{1.9cm}crrrrrr",
    )


def write_external_validation_main_tex(frame: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            [
                latex_escape(row["family_label"]),
                model_label_tex(row["model_label"]),
                fmt(float(row["official_auroc"])),
                fmt(float(row["official_auprc"])),
                fmt(float(row["spearman_abs_effect"])),
                fmt(float(row["topk_enrichment"])),
                fmt(float(row["worst_bin_auroc"])),
                fmt(float(row["gc_bin_auroc_gap"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "table8_external_validation_summary.tex",
        ["External family", "Model", "Official AUROC", "AUPRC", "Spearman$|\\Delta|$", "Top-k enrich.", "Worst-bin AUROC", "GC-bin gap"],
        rows,
        "p{2.0cm}p{2.0cm}rrrrrr",
    )


def write_external_prediction_main_tex(frame: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for row in frame.to_dict(orient="records"):
        value_cell = fmt(float(row["value"]))
        if pd.notna(row["ci_low"]) and pd.notna(row["ci_high"]):
            value_cell = f"{value_cell} [{fmt(float(row['ci_low']))}, {fmt(float(row['ci_high']))}]"
        metric_text = latex_escape(row["metric"]).replace("R^2", "R$^2$")
        rows.append(
            [
                latex_escape(row["analysis"]),
                metric_text,
                value_cell,
                fmt_count(row["n"]),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "table9_external_prediction_summary.tex",
        ["Analysis", "Statistic", "Estimate (95\\% CI)", "n"],
        rows,
        "p{5.5cm}p{1.2cm}p{3.5cm}c",
    )


def write_case_study_main_tex(frame: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            [
                latex_escape(row["case_study_id"]),
                task_label_tex(row["task_label"]),
                model_label_tex(row["model_label"]),
                latex_escape(row["condition_label"]),
                latex_escape(row["decision_role"]),
                fmt(float(row["core_mean_ece"])),
                fmt(float(row["auroc"])),
                fmt(float(row["auprc"])),
                fmt(float(row["topk_enrichment"])),
                fmt(float(row["spearman_abs_effect"])),
                fmt(float(row["worst_bin_auroc"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "table10_case_study_summary.tex",
        ["Case", "Task", "Model", "Config.", "Decision role", "Core ECE", "External AUROC", "External AUPRC", "Top-k enrich.", "Spearman", "Worst-bin AUROC"],
        rows,
        "p{0.85cm}p{2.0cm}p{1.6cm}p{1.8cm}p{3.2cm}rrrrrr",
    )


def write_appendix_real_tex(frame: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            [
                latex_escape(row["tier"]),
                task_label_tex(row["task_label"]),
                model_label_tex(MODEL_LABELS[row["model_id"]]),
                fmt(int(row["seeds"]), digits=0),
                fmt(float(row["auroc"])),
                fmt(float(row["ece"])),
                fmt(float(row["brier"])),
                fmt(float(row["rc_mean_abs_delta"])),
                fmt(float(row["rc_flip_rate"])),
                fmt(float(row["mono_positive_prob_drop"])),
                fmt(float(row["dinuc_positive_prob_drop"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_real_results.tex",
        ["Tier", "Task", "Model", "Seeds", "AUROC", "ECE", "Brier", "RC $\\Delta$", "RC flip", "Mono drop", "Dinuc drop"],
        rows,
        "p{0.9cm}p{2.3cm}p{2.2cm}crrrrrrr",
    )


def write_synthetic_appendix_tex(frame: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            [
                latex_escape(row["condition"]),
                model_label_tex(row["model_label"]),
                fmt(int(row["seeds"]), digits=0),
                fmt(float(row["auroc"])),
                fmt(float(row["ece"])),
                fmt(float(row["brier"])),
                fmt(float(row["mono_positive_prob_drop"])),
                fmt(float(row["dinuc_positive_prob_drop"])),
                fmt(float(row["motif_positive_prob_drop"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_synthetic_results.tex",
        ["Condition", "Model", "Seeds", "AUROC", "ECE", "Brier", "Mono drop", "Dinuc drop", "Motif drop"],
        rows,
        "p{2.2cm}p{2.2cm}crrrrrr",
    )


def write_gc_only_tex(frame: pd.DataFrame) -> None:
    subset = frame[
        (frame["task_id"].isin(FOCAL_TASKS))
        & (frame["model_id"] == "gc_only")
        & (frame["split_id"] == "official")
    ].copy()
    rows = [
        [task_label_tex(TASK_LABELS[row["task_id"]]), fmt(float(row["auroc"])), fmt(float(row["ece"])), fmt(float(row["brier"]))]
        for row in subset.to_dict(orient="records")
    ]
    write_tabular(
        PUBLICATION_ROOT / "appendix_gc_only.tex",
        ["Task", "AUROC", "ECE", "Brier"],
        rows,
        "p{2.7cm}rrr",
    )


def write_confounder_tex(confounders: pd.DataFrame, chrom_folds: pd.DataFrame) -> None:
    confounder_rows = []
    for row in confounders.to_dict(orient="records"):
        confounder_rows.append(
            [
                task_label_tex(row["task_label"]),
                f"{row['negative_gc_fraction_mean']:.3f} / {row['positive_gc_fraction_mean']:.3f}",
                f"{row['negative_cpg_oe_mean']:.3f} / {row['positive_cpg_oe_mean']:.3f}",
                f"{row['negative_length_mean']:.1f} / {row['positive_length_mean']:.1f}",
                f"{row['negative_repeat_fraction_mean']:.3f} / {row['positive_repeat_fraction_mean']:.3f}",
                f"{row['negative_n_fraction_mean']:.3f} / {row['positive_n_fraction_mean']:.3f}",
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_confounders.tex",
        ["Task", "GC (neg/pos)", "CpG O/E", "Length", "Repeat", "N frac."],
        confounder_rows,
        "p{2.5cm}p{1.8cm}p{1.7cm}p{1.5cm}p{1.5cm}p{1.4cm}",
    )

    fold_rows = []
    for row in chrom_folds.to_dict(orient="records"):
        fold_rows.append(
            [
                task_label_tex(row["task_label"]),
                fmt_count(row["A"]),
                fmt_count(row["B"]),
                fmt_count(row["C"]),
                fmt_count(row["D"]),
                fmt_count(row["E"]),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_chrom_folds.tex",
        ["Task", "Fold A", "Fold B", "Fold C", "Fold D", "Fold E"],
        fold_rows,
        "p{2.7cm}rrrrr",
    )


def write_cv_tex(cv_summary: pd.DataFrame, cv_folds: pd.DataFrame) -> None:
    summary_rows = []
    for row in cv_summary.to_dict(orient="records"):
        summary_rows.append(
            [
                task_label_tex(row["task_label"]),
                model_label_tex(row["model_label"]),
                fmt(float(row["auroc_mean"])),
                fmt(float(row["auroc_std"])),
                fmt(float(row["ece_mean"])),
                fmt(float(row["brier_mean"])),
                fmt(float(row["rc_mean_abs_delta_mean"])),
                fmt(float(row["mono_positive_prob_drop_mean"])),
                fmt(float(row["dinuc_positive_prob_drop_mean"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_chromosome_cv_summary.tex",
        ["Task", "Model", "Mean AUROC", "SD", "ECE", "Brier", "RC $\\Delta$", "Mono drop", "Dinuc drop"],
        summary_rows,
        "p{2.5cm}p{2.2cm}rrrrrrr",
    )

    fold_rows = []
    for row in cv_folds.to_dict(orient="records"):
        fold_rows.append(
            [
                task_label_tex(TASK_LABELS[row["task_id"]]),
                model_label_tex(MODEL_LABELS[row["model_id"]]),
                latex_escape(str(row["split_fold"])),
                fmt(float(row["auroc"])),
                fmt(float(row["ece"])),
                fmt(float(row["brier"])),
                fmt(float(row["rc_mean_abs_delta"])),
                fmt(float(row["mono_positive_prob_drop"])),
                fmt(float(row["dinuc_positive_prob_drop"])),
                f"{fmt_count(row['positive_count'])}:{fmt_count(row['negative_count'])}",
                fmt_count(row["test_count"]),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_chromosome_cv_folds.tex",
        ["Task", "Model", "Fold", "AUROC", "ECE", "Brier", "RC $\\Delta$", "Mono drop", "Dinuc drop", "P:N", "Test n"],
        fold_rows,
        "p{2.15cm}p{2.1cm}crrrrrrrc",
    )


def write_matched_negative_tex(results: pd.DataFrame, confounders: pd.DataFrame) -> None:
    result_rows = []
    for row in results.to_dict(orient="records"):
        result_rows.append(
            [
                task_label_tex(row["task_label"]),
                model_label_tex(row["model_label"]),
                fmt(float(row["original_auroc"])),
                fmt(float(row["matched_auroc"])),
                fmt(float(row["original_gc_only_auroc"])),
                fmt(float(row["matched_gc_only_auroc"])),
                fmt(float(row["original_ece"])),
                fmt(float(row["matched_ece"])),
                fmt(float(row["original_mono_drop"])),
                fmt(float(row["matched_mono_drop"])),
                fmt(float(row["original_dinuc_drop"])),
                fmt(float(row["matched_dinuc_drop"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_matched_negative_results.tex",
        ["Task", "Model", "Orig. AUROC", "Matched AUROC", "Orig. GC-only", "Matched GC-only", "Orig. ECE", "Matched ECE", "Orig. mono", "Matched mono", "Orig. dinuc", "Matched dinuc"],
        result_rows,
        "p{2.3cm}p{2.2cm}rrrrrrrrrr",
    )

    confounder_rows = []
    for row in confounders.to_dict(orient="records"):
        confounder_rows.append(
            [
                task_label_tex(row["task_label"]),
                latex_escape(row["gc_before_after"]),
                latex_escape(row["cpg_before_after"]),
                latex_escape(row["length_before_after"]),
                latex_escape(row["repeat_before_after"]),
                latex_escape(row["n_before_after"]),
                latex_escape(row["gc_only_shift"]),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_matched_negative_confounders.tex",
        ["Task", "GC neg/pos", "CpG neg/pos", "Length neg/pos", "Repeat neg/pos", "N neg/pos", "GC-only AUROC"],
        confounder_rows,
        "p{2.2cm}p{2.6cm}p{2.6cm}p{2.7cm}p{2.6cm}p{2.5cm}p{1.6cm}",
    )


def write_mitigation_tex(results: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for row in results.to_dict(orient="records"):
        rows.append(
            [
                task_label_tex(row["task_label"]),
                model_label_tex(row["model_label"]),
                intervention_label_tex(row["intervention"]),
                fmt(float(row["auroc_before"])),
                fmt(float(row["auroc_after"])),
                fmt(float(row["ece_before"])),
                fmt(float(row["ece_after"])),
                fmt(float(row["brier_before"])),
                fmt(float(row["brier_after"])),
                fmt(float(row["rc_before"])),
                fmt(float(row["rc_after"])),
                fmt(float(row["mono_before"])),
                fmt(float(row["mono_after"])),
                fmt(float(row["dinuc_before"])),
                fmt(float(row["dinuc_after"])),
                fmt(float(row["matched_auroc_before"])),
                fmt(float(row["matched_auroc_after"])),
                fmt(float(row["matched_ece_before"])),
                fmt(float(row["matched_ece_after"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_mitigation_results.tex",
        ["Task", "Model", "Intervention", "AUROC before", "AUROC after", "ECE before", "ECE after", "Brier before", "Brier after", "RC before", "RC after", "Mono before", "Mono after", "Dinuc before", "Dinuc after", "Matched AUROC before", "Matched AUROC after", "Matched ECE before", "Matched ECE after"],
        rows,
        "p{2.0cm}p{1.7cm}p{1.9cm}rrrrrrrrrrrrrrrr",
    )


def write_motif_tex(results: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for row in results.to_dict(orient="records"):
        rows.append(
            [
                task_label_tex(row["task_label"]),
                model_label_tex(row["model_label"]),
                fmt(int(row["evaluated_count"]), digits=0),
                fmt(float(row["official_auroc"])),
                fmt(float(row["motif_drop"])),
                fmt(float(row["gc_preserving_motif_drop"])),
                fmt(float(row["random_edit_drop"])),
                fmt(float(row["motif_minus_random"])),
                fmt(float(row["mono_positive_prob_drop"])),
                fmt(float(row["dinuc_positive_prob_drop"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_real_motif_results.tex",
        ["Task", "Model", "n", "AUROC", "Motif drop", "GC-preserving", "Random edit", "Motif$-$random", "Mono drop", "Dinuc drop"],
        rows,
        "p{2.2cm}p{1.9cm}crrrrrrr",
    )


def write_gc_bin_tex(results: pd.DataFrame, per_bin: pd.DataFrame) -> None:
    summary_rows: list[list[str]] = []
    for row in results.to_dict(orient="records"):
        summary_rows.append(
            [
                task_label_tex(row["task_label"]),
                model_label_tex(row["model_label"]),
                fmt(float(row["overall_auroc"])),
                fmt(float(row["worst_bin_auroc"])),
                fmt(float(row["gc_bin_auroc_gap"])),
                fmt(float(row["overall_ece"])),
                fmt(float(row["worst_bin_ece"])),
                fmt(float(row["gc_bin_ece_gap"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_gc_bin_summary.tex",
        ["Task", "Model", "Overall AUROC", "Worst-bin AUROC", "AUROC gap", "Overall ECE", "Worst-bin ECE", "ECE gap"],
        summary_rows,
        "p{2.2cm}p{2.0cm}rrrrrr",
    )

    bin_rows: list[list[str]] = []
    for row in per_bin.to_dict(orient="records"):
        bin_rows.append(
            [
                task_label_tex(row["task_label"]),
                model_label_tex(row["model_label"]),
                latex_escape(str(int(row["gc_bin"]))),
                fmt(float(row["gc_mean"])),
                fmt_count(row["count"]),
                f"{fmt_count(row['positive_count'])}:{fmt_count(row['negative_count'])}",
                fmt(float(row["auroc"])),
                fmt(float(row["ece"])),
                fmt(float(row["brier"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_gc_bin_by_bin.tex",
        ["Task", "Model", "GC bin", "GC mean", "n", "P:N", "AUROC", "ECE", "Brier"],
        bin_rows,
        "p{2.2cm}p{1.9cm}crcrrrr",
    )


def write_synthetic_extended_tex(results: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for row in results.to_dict(orient="records"):
        rows.append(
            [
                task_label_tex(row["task_label"]),
                model_label_tex(row["model_label"]),
                fmt(float(row["auroc"])),
                fmt(float(row["ece"])),
                fmt(float(row["brier"])),
                fmt(float(row["mono_positive_prob_drop"])),
                fmt(float(row["dinuc_positive_prob_drop"])),
                fmt(float(row["motif_positive_prob_drop"])),
                fmt(float(row["rule_following_rate"])),
                fmt(float(row["shortcut_following_rate"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_synthetic_extended.tex",
        ["Task", "Model", "AUROC", "ECE", "Brier", "Mono drop", "Dinuc drop", "Motif drop", "Rule rate", "Shortcut rate"],
        rows,
        "p{2.8cm}p{1.9cm}rrrrrrrr",
    )


def write_shortcut_score_tex(task_rows: pd.DataFrame, model_rows: pd.DataFrame) -> None:
    task_table = []
    for row in task_rows.to_dict(orient="records"):
        task_table.append(
            [
                task_label_tex(row["task_label"]),
                model_label_tex(row["model_label"]),
                fmt(float(row["auroc"])),
                fmt(float(row["shortcut_score"])),
                fmt(float(row["gc_bin_auroc_gap"])),
                fmt(float(row["matched_auroc_drop"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_shortcut_score_by_task.tex",
        ["Task", "Model", "AUROC", "Shortcut score", "GC-bin gap", "Matched drop"],
        task_table,
        "p{2.3cm}p{1.9cm}rrrr",
    )

    model_table = []
    for row in model_rows.to_dict(orient="records"):
        model_table.append(
            [
                model_label_tex(row["model_label"]),
                fmt(float(row["mean_auroc"])),
                fmt(float(row["mean_auroc_rank"])),
                fmt(float(row["mean_shortcut_score"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_shortcut_score_summary.tex",
        ["Model", "Mean AUROC", "Mean AUROC rank", "Mean Shortcut Score"],
        model_table,
        "p{2.2cm}rrr",
    )


def write_external_validation_tex(frame: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            [
                task_label_tex(row["task_label"]),
                model_label_tex(row["model_label"]),
                fmt(float(row["auroc"])),
                fmt(float(row["auprc"])),
                fmt(float(row["ece"])),
                fmt(float(row["brier"])),
                fmt(float(row["matched_auroc"])),
                fmt(float(row["matched_negative_shift"])),
                fmt(float(row["worst_bin_auroc"])),
                fmt(float(row["gc_bin_auroc_gap"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_external_validation_results.tex",
        ["External task", "Model", "Official AUROC", "AUPRC", "ECE", "Brier", "Matched AUROC", "Matched shift", "Worst-bin AUROC", "GC-bin gap"],
        rows,
        "p{2.4cm}p{1.9cm}rrrrrrrr",
    )


def write_external_prediction_tex(frame: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for row in frame.to_dict(orient="records"):
        value_cell = fmt(float(row["value"]))
        if pd.notna(row["ci_low"]) and pd.notna(row["ci_high"]):
            value_cell = f"{value_cell} [{fmt(float(row['ci_low']))}, {fmt(float(row['ci_high']))}]"
        metric_text = latex_escape(row["metric"]).replace("R^2", "R$^2$")
        rows.append(
            [
                latex_escape(row["analysis"]),
                metric_text,
                value_cell,
                fmt_count(row["n"]),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_external_prediction_summary.tex",
        ["Analysis", "Statistic", "Estimate (95\\% CI)", "n"],
        rows,
        "p{6.1cm}p{1.2cm}p{3.6cm}c",
    )


def build_external_prediction_robustness_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    order = [
        "in-sample (pooled)",
        "leave-one-task-out",
        "leave-one-family-out",
        "within-family leave-one-task-out (Variant effect)",
        "in-sample model-family stratified (k-mer baseline)",
        "in-sample model-family stratified (CNN)",
        "in-sample model-family stratified (foundation model)",
    ]
    subset = frame.copy()
    subset["order"] = subset["analysis_type"].map({name: idx for idx, name in enumerate(order)}).fillna(9999)
    return subset.sort_values(["order", "analysis_type"]).drop(columns=["order"]).reset_index(drop=True)


def write_external_prediction_robustness_tex(frame: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for row in frame.to_dict(orient="records"):
        delta_cell = fmt(float(row["delta_full_vs_auroc"])) if pd.notna(row["delta_full_vs_auroc"]) else "--"
        if pd.notna(row.get("delta_ci_low")) and pd.notna(row.get("delta_ci_high")):
            delta_cell = f"{delta_cell} [{fmt(float(row['delta_ci_low']))}, {fmt(float(row['delta_ci_high']))}]"
        perm_cell = "--"
        if pd.notna(row.get("permutation_p")):
            perm_cell = latex_escape(f"{float(row['permutation_p']):.4f}")
        rows.append(
            [
                latex_escape(row["analysis_type"]),
                fmt_count(row["n"]),
                fmt(float(row["auroc_only_r2"])) if pd.notna(row["auroc_only_r2"]) else "--",
                fmt(float(row["shortcut_only_r2"])) if pd.notna(row["shortcut_only_r2"]) else "--",
                fmt(float(row["full_profile_r2"])) if pd.notna(row["full_profile_r2"]) else "--",
                delta_cell,
                perm_cell,
                latex_escape(str(row.get("interpretation", ""))),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_external_prediction_robustness.tex",
        [
            "Analysis",
            "n",
            "AUROC-only R$^2$",
            "Shortcut-only R$^2$",
            "Full profile R$^2$",
            "$\\Delta R^2$ (95\\% CI)",
            "Perm. $p$",
            "Notes",
        ],
        rows,
        "p{4.0cm}c r r r p{2.7cm} c p{4.1cm}",
    )


def write_case_study_tex(frame: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            [
                latex_escape(row["case_study_id"]),
                task_label_tex(row["task_label"]),
                model_label_tex(row["model_label"]),
                latex_escape(row["condition_label"]),
                latex_escape(row["decision_role"]),
                fmt(float(row["core_mean_auroc"])),
                fmt(float(row["core_mean_ece"])),
                fmt(float(row["core_mean_shortcut_score"])),
                fmt(float(row["auroc"])),
                fmt(float(row["auprc"])),
                fmt(float(row["spearman_abs_effect"])),
                fmt(float(row["topk_enrichment"])),
                fmt(float(row["ece"])),
                fmt(float(row["brier"])),
                fmt(float(row["worst_bin_auroc"])),
                fmt(float(row["gc_bin_auroc_gap"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_case_study_results.tex",
        ["Case", "Task", "Model", "Config.", "Decision role", "Core AUROC", "Core ECE", "Core score", "External AUROC", "External AUPRC", "Spearman$|\\Delta|$", "Top-k enrich.", "ECE", "Brier", "Worst-bin", "GC gap"],
        rows,
        "p{0.85cm}p{2.0cm}p{1.6cm}p{1.7cm}p{2.9cm}rrrrrrrrrrrr",
    )


def write_external_gc_bin_tex(summary_rows: pd.DataFrame, bin_rows_frame: pd.DataFrame) -> None:
    summary_rows_out: list[list[str]] = []
    for row in summary_rows.to_dict(orient="records"):
        summary_rows_out.append(
            [
                task_label_tex(TASK_LABELS[row["task_id"]]),
                model_label_tex(MODEL_LABELS[row["model_id"]]),
                fmt(float(row["overall_auroc"])),
                fmt(float(row["worst_bin_auroc"])),
                fmt(float(row["gc_bin_auroc_gap"])),
                fmt(float(row["overall_ece"])),
                fmt(float(row["worst_bin_ece"])),
                fmt(float(row["gc_bin_ece_gap"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_external_gc_bin_summary.tex",
        ["External task", "Model", "Overall AUROC", "Worst-bin AUROC", "AUROC gap", "Overall ECE", "Worst-bin ECE", "ECE gap"],
        summary_rows_out,
        "p{2.7cm}p{1.9cm}rrrrrr",
    )

    bin_rows: list[list[str]] = []
    for row in bin_rows_frame.to_dict(orient="records"):
        bin_rows.append(
            [
                task_label_tex(TASK_LABELS[row["task_id"]]),
                model_label_tex(MODEL_LABELS[row["model_id"]]),
                latex_escape(str(int(row["gc_bin"]))),
                fmt(float(row["gc_mean"])),
                fmt_count(row["count"]),
                f"{fmt_count(row['positive_count'])}:{fmt_count(row['negative_count'])}",
                fmt(float(row["auroc"])),
                fmt(float(row["ece"])),
                fmt(float(row["brier"])),
            ]
        )
    write_tabular(
        PUBLICATION_ROOT / "appendix_external_gc_bin_by_bin.tex",
        ["External task", "Model", "GC bin", "GC mean", "n", "P:N", "AUROC", "ECE", "Brier"],
        bin_rows,
        "p{2.7cm}p{1.9cm}crcrrrr",
    )


def plot_tradeoff(summary: pd.DataFrame) -> None:
    configure_matplotlib()
    subset = official_standard_rows(summary)
    subset = subset[(subset["task_id"].isin(CORE_TASKS)) & (subset["model_id"].isin(PAPER_MODELS))].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.8))
    panel_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    legend_ax = axes[1, 1]
    panels = [
        ("rc_mean_abs_delta", "AUROC vs reverse-complement instability", "Reverse-complement instability (lower is better)"),
        ("mono_positive_prob_drop", "AUROC vs mononucleotide-shuffle sensitivity", "Positive-probability drop (higher is better)"),
        ("ece", "AUROC vs expected calibration error", "Expected calibration error (lower is better)"),
    ]
    for ax, (metric, title, ylabel) in zip(panel_axes, panels):
        if metric == "mono_positive_prob_drop":
            ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
        for model_id in PAPER_MODELS:
            group = subset[subset["model_id"] == model_id]
            if group.empty:
                continue
            ax.scatter(
                group["auroc"],
                group[metric],
                s=92,
                marker=MODEL_MARKERS[model_id],
                color=MODEL_COLORS[model_id],
                edgecolor="white",
                linewidth=0.8,
                label=MODEL_LABELS[model_id],
                zorder=3,
            )
            for _, row in group.iterrows():
                ax.annotate(
                    TASK_SHORT_LABELS[row["task_id"]],
                    (row["auroc"], row[metric]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=9.2,
                    color="#333333",
                )
        ax.set_xlabel("AUROC")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.18, linewidth=0.6)
    legend_ax.axis("off")
    handles, labels = panel_axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, frameon=False, loc="upper left", ncol=1, fontsize=10)
    legend_ax.text(
        0.0,
        0.48,
        "Task labels:\nPr = Promoters\nEC = Enhancers (Cohn)\nEE = Enhancers (Ensembl)\nOCR = Open chromatin",
        fontsize=10,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "genomecf_tradeoff_publication.png", bbox_inches="tight")
    plt.close(fig)


def plot_calibration(registry: pd.DataFrame) -> None:
    configure_matplotlib()
    subset = registry[
        (registry["task_id"].isin(FOCAL_TASKS))
        & (registry["model_id"].isin(PAPER_MODELS))
        & (registry["split_id"] == "official")
        & (registry["calibration_method"] == "none")
        & (registry["intervention_id"] == "standard")
        & (registry["perturbation_id"].isin(["original", "k1_shuffle"]))
    ].copy()
    subset = (
        subset.groupby(["task_id", "model_id", "perturbation_id"], as_index=False)[["ece", "brier"]]
        .mean()
        .sort_values(["task_id", "model_id", "perturbation_id"])
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 6.9), sharex="col")
    x = np.arange(len(PAPER_MODELS))
    width = 0.36
    for col_idx, task_name in enumerate(FOCAL_TASKS):
        task = subset[subset["task_id"] == task_name]
        original = task[task["perturbation_id"] == "original"].set_index("model_id").reindex(PAPER_MODELS)
        shuffled = task[task["perturbation_id"] == "k1_shuffle"].set_index("model_id").reindex(PAPER_MODELS)
        original_ece = original["ece"].to_numpy()
        shuffle_ece = shuffled["ece"].to_numpy()
        original_brier = original["brier"].to_numpy()
        mono_brier = shuffled["brier"].to_numpy()
        ece_ax = axes[0, col_idx]
        brier_ax = axes[1, col_idx]
        ece_ax.bar(x - width / 2, original_ece, width=width, color="#355070", label="Original")
        ece_ax.bar(x + width / 2, shuffle_ece, width=width, color="#e07a5f", label="Mononucleotide shuffle")
        brier_ax.bar(x - width / 2, original_brier, width=width, color="#355070")
        brier_ax.bar(x + width / 2, mono_brier, width=width, color="#e07a5f")
        for ax, left_vals, right_vals in [(ece_ax, original_ece, shuffle_ece), (brier_ax, original_brier, mono_brier)]:
            for xpos, value in zip(x - width / 2, left_vals):
                ax.text(xpos, value + 0.004, f"{value:.02f}", ha="center", va="bottom", fontsize=8.5, rotation=90)
            for xpos, value in zip(x + width / 2, right_vals):
                ax.text(xpos, value + 0.004, f"{value:.02f}", ha="center", va="bottom", fontsize=8.5, rotation=90)
            ax.grid(axis="y", alpha=0.18, linewidth=0.6)
        ece_ax.set_title(TASK_LABELS[task_name])
        brier_ax.set_xticks(x)
        brier_ax.set_xticklabels([MODEL_LABELS[name] for name in PAPER_MODELS], rotation=15, ha="right")
    axes[0, 0].set_ylabel("Expected calibration error")
    axes[1, 0].set_ylabel("Brier score")
    axes[0, 0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.25), ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "genomecf_calibration_publication.png", bbox_inches="tight")
    plt.close(fig)


def plot_generalization_gap(summary: pd.DataFrame, cv_summary: pd.DataFrame) -> None:
    configure_matplotlib()
    short_labels = {
        "kmer_logistic_regression": "6-mer",
        "small_cnn": "CNN",
        "small_cnn_rc_aug": "RC-aug",
        "dnabert2": "DNABERT-2",
        "caduceus_ph": "Caduceus",
    }
    official = official_standard_rows(summary)
    official = official[(official["task_id"].isin(CORE_TASKS)) & (official["model_id"].isin(CV_MODELS))][["task_id", "model_id", "auroc"]].rename(columns={"auroc": "official_auroc"})
    merged = official.merge(cv_summary[["task_id", "model_id", "auroc_mean", "auroc_std"]], on=["task_id", "model_id"], how="inner")
    merged["gap"] = merged["official_auroc"] - merged["auroc_mean"]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.0), sharey=True)
    axes = axes.ravel()
    for ax, task_name in zip(axes, CORE_TASKS):
        available = [model_id for model_id in CV_MODELS if model_id in set(merged[merged["task_id"] == task_name]["model_id"])]
        subset = merged[merged["task_id"] == task_name].set_index("model_id").loc[available]
        x = np.arange(len(available))
        values = subset["gap"].to_numpy()
        errors = subset["auroc_std"].to_numpy()
        bars = ax.bar(
            x,
            values,
            yerr=errors,
            color=[MODEL_COLORS[name] for name in available],
            capsize=3,
        )
        ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
        ax.set_title(TASK_LABELS[task_name])
        ax.set_xticks(x)
        ax.set_xticklabels([short_labels.get(name, MODEL_LABELS[name]) for name in available], rotation=12, ha="right")
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)
        for bar, value in zip(bars, values):
            offset = 0.006 if value >= 0 else -0.012
            ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.03f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8.5)
    fig.supylabel("Official AUROC minus five-fold CV mean AUROC", x=0.04, fontsize=11)
    fig.tight_layout(pad=1.0, w_pad=1.2, h_pad=1.4)
    fig.subplots_adjust(left=0.12, bottom=0.12)
    fig.savefig(FIGURES_ROOT / "genomecf_generalization_gap.png", bbox_inches="tight")
    plt.close(fig)


def plot_gc_bin_robustness(gc_bin_results: pd.DataFrame) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.1))
    panels = [
        ("worst_bin_auroc", "Overall AUROC vs worst-GC-bin AUROC", "Worst-GC-bin AUROC"),
        ("gc_bin_auroc_gap", "Overall AUROC vs GC-bin AUROC gap", "Best-minus-worst GC-bin AUROC"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, panels):
        for model_id in PAPER_MODELS:
            subset = gc_bin_results[gc_bin_results["model_id"] == model_id]
            if subset.empty:
                continue
            ax.scatter(
                subset["overall_auroc"],
                subset[metric],
                s=78,
                marker=MODEL_MARKERS[model_id],
                color=MODEL_COLORS[model_id],
                edgecolor="white",
                linewidth=0.8,
                label=MODEL_LABELS[model_id],
                zorder=3,
            )
            for _, row in subset.iterrows():
                ax.annotate(
                    TASK_SHORT_LABELS[row["task_id"]],
                    (row["overall_auroc"], row[metric]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8.5,
                    color="#333333",
                )
        ax.set_xlabel("Overall AUROC")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.18, linewidth=0.6)
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.24), ncol=3)
    fig.text(0.5, -0.03, "Task labels: Pr = Promoters, EC = Enhancers (Cohn), EE = Enhancers (Ensembl), OCR = Open chromatin.", ha="center", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "genomecf_gc_bin_robustness.png", bbox_inches="tight")
    plt.close(fig)


def plot_synthetic(synthetic_extended: pd.DataFrame) -> None:
    configure_matplotlib()
    synth = synthetic_extended.copy()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.0))
    width = 0.14
    top_conditions = ["gc_correlated", "gc_matched"]
    x = np.arange(len(top_conditions))
    available_models = [model for model in SYNTH_MODELS if set(top_conditions).issubset(set(synth[synth["model_id"] == model]["task_id"]))]
    for ax, metric, title in [
        (axes[0, 0], "auroc", "Planted-motif pair: AUROC"),
        (axes[0, 1], "motif_positive_prob_drop", "Planted-motif pair: motif-disruption drop"),
    ]:
        for offset_idx, model_id in enumerate(available_models):
            subset = synth[synth["model_id"] == model_id].set_index("task_id").loc[top_conditions]
            vals = subset[metric].to_numpy()
            xpos = x + (offset_idx - (len(available_models) - 1) / 2) * width
            ax.bar(xpos, vals, width=width, color=MODEL_COLORS[model_id], label=MODEL_LABELS[model_id] if metric == "auroc" else None)
        if metric != "auroc":
            ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(["GC correlated", "GC matched"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)

    for ax, task_id, title in [
        (axes[1, 0], "gc_conflict", "GC conflict: rule vs shortcut"),
        (axes[1, 1], "motif_position_conflict", "Position conflict: rule vs shortcut"),
    ]:
        subset = synth[synth["task_id"] == task_id].set_index("model_id")
        models = [model for model in SYNTH_MODELS if model in subset.index]
        xpos = np.arange(len(models))
        rule = subset.loc[models, "rule_following_rate"].to_numpy()
        shortcut = subset.loc[models, "shortcut_following_rate"].to_numpy()
        ax.bar(xpos - 0.18, rule, width=0.36, color="#355070", label="Rule-following rate" if task_id == "gc_conflict" else None)
        ax.bar(xpos + 0.18, shortcut, width=0.36, color="#e07a5f", label="Shortcut-following rate" if task_id == "gc_conflict" else None)
        ax.set_xticks(xpos)
        ax.set_xticklabels([MODEL_LABELS[name] for name in models], rotation=15, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    axes[0, 0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.24), ncol=3)
    axes[1, 0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.20), ncol=2)
    fig.suptitle("Synthetic GenomeCF tasks reveal when high AUROC still reflects shortcut following.", y=0.99, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "genomecf_synthetic_publication.png", bbox_inches="tight")
    plt.close(fig)


def plot_shortcut_score(model_summary: pd.DataFrame) -> None:
    configure_matplotlib()
    ordered = model_summary.sort_values("mean_auroc_rank", ascending=False).copy()
    x = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    axes[0].bar(
        x,
        ordered["mean_auroc_rank"].to_numpy(),
        color=[MODEL_COLORS[row["model_id"]] for _, row in ordered.iterrows()],
    )
    axes[1].bar(
        x,
        ordered["mean_shortcut_score"].to_numpy(),
        color=[MODEL_COLORS[row["model_id"]] for _, row in ordered.iterrows()],
    )
    axes[0].set_title("Higher AUROC rank is better")
    axes[1].set_title("Higher Shortcut Score is worse")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(ordered["model_label"].tolist(), rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    axes[0].set_ylabel("Average within-task rank")
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "genomecf_shortcut_score.png", bbox_inches="tight")
    plt.close(fig)


def plot_foundation_comparison(summary: pd.DataFrame, mitigation_results: pd.DataFrame) -> None:
    configure_matplotlib()
    focal_tasks = ["human_nontata_promoters", "human_enhancers_cohn"]
    models = ["dnabert2", "caduceus_ph"]
    official = official_standard_rows(summary)
    official = official[
        official["task_id"].isin(focal_tasks) & official["model_id"].isin(models)
    ].copy()
    official["task_label"] = official["task_id"].map(TASK_LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.1))
    width = 0.18
    x = np.arange(len(focal_tasks))

    for ax, metric, title in [
        (axes[0, 0], "auroc", "Frozen foundation models: focal-task AUROC"),
        (axes[0, 1], "rc_mean_abs_delta", "Frozen foundation models: reverse-complement instability"),
    ]:
        for offset_idx, model_id in enumerate(models):
            subset = official[official["model_id"] == model_id].set_index("task_id").reindex(focal_tasks)
            xpos = x + (offset_idx - 0.5) * width
            ax.bar(xpos, subset[metric].to_numpy(), width=width, color=MODEL_COLORS[model_id], label=MODEL_LABELS[model_id])
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS[name] for name in focal_tasks])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    axes[0, 0].legend(frameon=False, loc="upper left")
    axes[0, 0].set_ylabel("AUROC")
    axes[0, 1].set_ylabel("RC instability")

    intervention_map = {
        "Temperature scaling": "ece_after",
        "Matched-negative retraining": "matched_auroc_after",
    }
    baseline_map = {
        "Temperature scaling": "ece_before",
        "Matched-negative retraining": "matched_auroc_before",
    }
    panels = [
        (axes[1, 0], "Temperature scaling", "Temperature scaling lowers ECE"),
        (axes[1, 1], "Matched-negative retraining", "Matched-negative heads change matched-test AUROC"),
    ]
    for ax, intervention, title in panels:
        panel = mitigation_results[
            mitigation_results["task_id"].isin(focal_tasks)
            & mitigation_results["model_id"].isin(models)
            & (mitigation_results["intervention"] == intervention)
        ].copy()
        labels = []
        before = []
        after = []
        colors = []
        for task_name in focal_tasks:
            for model_id in models:
                subset = panel[(panel["task_id"] == task_name) & (panel["model_id"] == model_id)]
                if subset.empty:
                    continue
                row = subset.iloc[0]
                labels.append(f"{TASK_SHORT_LABELS[task_name]}-{MODEL_LABELS[model_id]}")
                before.append(float(row[baseline_map[intervention]]))
                after.append(float(row[intervention_map[intervention]]))
                colors.append(MODEL_COLORS[model_id])
        xpos = np.arange(len(labels))
        ax.bar(xpos - 0.16, before, width=0.32, color="#b0bec5", label="Before" if intervention == "Temperature scaling" else None)
        ax.bar(xpos + 0.16, after, width=0.32, color=colors, label="After" if intervention == "Temperature scaling" else None)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    axes[1, 0].set_ylabel("ECE")
    axes[1, 1].set_ylabel("Matched-negative AUROC")
    axes[1, 0].legend(frameon=False, loc="upper left")
    fig.text(0.5, -0.02, "Task abbreviations: Pr = Promoters, EC = Enhancers (Cohn).", ha="center", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "genomecf_foundation_comparison.png", bbox_inches="tight")
    plt.close(fig)


def write_source_data(
    summary: pd.DataFrame,
    main_results: pd.DataFrame,
    mitigation_results: pd.DataFrame,
    external_validation_results: pd.DataFrame,
    external_prediction_results: pd.DataFrame,
    external_prediction_robustness_table: pd.DataFrame,
    case_study_results: pd.DataFrame,
    synthetic_extended_results: pd.DataFrame,
) -> None:
    SOURCE_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    fig1_payload = {
        "figure": "Fig. 1",
        "title": "GenomeCF resource overview",
        "blocks": [
            {"group": "inputs", "label": "Core benchmark", "details": "4 human tasks; official split and chromosome CV"},
            {"group": "inputs", "label": "External biological validation", "details": "TF binding, histone marks, MPRA variant effect"},
            {"group": "inputs", "label": "GenomeCF-Synth", "details": "shortcut conflict, motif grammar, position conflict"},
            {"group": "metrics", "label": "Counterfactual metrics", "details": "AUROC, AUPRC, ECE, Brier, RC, shuffle, matched-negative, GC-bin"},
            {"group": "outputs", "label": "Software resource", "details": "package, CLI, registry, website, checklist"},
            {"group": "outputs", "label": "Use cases", "details": "model reliability profiles, variant prioritization, reporting standard"},
        ],
        "source_artifacts": [
            "results/publication/table1_task_overview.csv",
            "docs/reporting_checklist.yaml",
            "results/release/benchmark_registry.csv",
        ],
    }
    (SOURCE_DATA_ROOT / "Fig1_source_data.json").write_text(json.dumps(fig1_payload, indent=2))

    main_results.to_csv(SOURCE_DATA_ROOT / "Fig2_source_data.csv", index=False)

    official = official_standard_rows(summary)
    foundation_official = official[
        official["task_id"].isin(FOCAL_TASKS) & official["model_id"].isin(["dnabert2", "caduceus_ph"])
    ][["task_id", "model_id", "auroc", "rc_mean_abs_delta"]].copy()
    foundation_official["task_label"] = foundation_official["task_id"].map(TASK_LABELS)
    foundation_official["model_label"] = foundation_official["model_id"].map(MODEL_LABELS)
    foundation_long = foundation_official.melt(
        id_vars=["task_id", "task_label", "model_id", "model_label"],
        value_vars=["auroc", "rc_mean_abs_delta"],
        var_name="metric",
        value_name="value",
    )
    foundation_long["panel"] = "frozen_foundation_models"
    mitigation_long = mitigation_results[
        mitigation_results["task_id"].isin(FOCAL_TASKS)
        & mitigation_results["model_id"].isin(["dnabert2", "caduceus_ph"])
        & mitigation_results["intervention"].isin(["Temperature scaling", "Matched-negative retraining"])
    ].copy()
    intervention_rows: list[dict[str, object]] = []
    for _, row in mitigation_long.iterrows():
        if row["intervention"] == "Temperature scaling":
            pairs = [("ece_before", "before"), ("ece_after", "after")]
            metric_name = "ece"
        else:
            pairs = [("matched_auroc_before", "before"), ("matched_auroc_after", "after")]
            metric_name = "matched_auroc"
        for field, phase in pairs:
            intervention_rows.append(
                {
                    "panel": row["intervention"],
                    "task_id": row["task_id"],
                    "task_label": TASK_LABELS[row["task_id"]],
                    "model_id": row["model_id"],
                    "model_label": MODEL_LABELS[row["model_id"]],
                    "metric": metric_name,
                    "phase": phase,
                    "value": row[field],
                }
            )
    foundation_source = pd.concat([foundation_long, pd.DataFrame(intervention_rows)], ignore_index=True, sort=False)
    foundation_source.to_csv(SOURCE_DATA_ROOT / "Fig3_source_data.csv", index=False)

    external_validation_source = external_validation_results.copy()
    external_validation_source["panel"] = "assay_family_performance"
    external_prediction_source = external_prediction_results.copy()
    external_prediction_source["panel"] = "prediction_summary"
    external_prediction_robustness_source = external_prediction_robustness_table.copy()
    external_prediction_robustness_source["panel"] = "prediction_robustness"
    external_source = pd.concat(
        [external_validation_source, external_prediction_source, external_prediction_robustness_source],
        ignore_index=True,
        sort=False,
    )
    external_source.to_csv(SOURCE_DATA_ROOT / "Fig4_source_data.csv", index=False)

    case_study_results.to_csv(SOURCE_DATA_ROOT / "Fig5_source_data.csv", index=False)
    synthetic_extended_results.to_csv(SOURCE_DATA_ROOT / "Fig6_source_data.csv", index=False)

    manifest = {
        "Fig1": {"path": "source_data/Fig1_source_data.json", "figure": "genomecf overview schematic"},
        "Fig2": {"path": "source_data/Fig2_source_data.csv", "figure": "tradeoff scatter panels"},
        "Fig3": {"path": "source_data/Fig3_source_data.csv", "figure": "foundation-model comparison"},
        "Fig4": {"path": "source_data/Fig4_source_data.csv", "figure": "external validation and prediction"},
        "Fig5": {"path": "source_data/Fig5_source_data.csv", "figure": "MPRA case studies"},
        "Fig6": {"path": "source_data/Fig6_source_data.csv", "figure": "GenomeCF-Synth"},
    }
    (SOURCE_DATA_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (SOURCE_DATA_ROOT / "README.md").write_text(
        "\n".join(
            [
                "# GenomeCF source data",
                "",
                "This directory contains the source-data files for the six main display items in the Nature Methods Resource manuscript.",
                "",
                "- `Fig1_source_data.json`: schematic block and artifact mapping for the resource overview.",
                "- `Fig2_source_data.csv`: core-task AUROC and counterfactual metrics used in the tradeoff figure.",
                "- `Fig3_source_data.csv`: focal-task foundation-model metrics and adaptation summaries.",
                "- `Fig4_source_data.csv`: external assay-family results and external-prediction summaries.",
                "- `Fig5_source_data.csv`: MPRA case-study results for BCL11A and MYC.",
                "- `Fig6_source_data.csv`: GenomeCF-Synth results used in the synthetic mechanism figure.",
            ]
        )
    )


def write_key_numbers(
    summary: pd.DataFrame,
    matched_results: pd.DataFrame,
    mitigation_results: pd.DataFrame,
    gc_bin_results: pd.DataFrame,
    synthetic_extended: pd.DataFrame,
    external_prediction_summary: pd.DataFrame,
    case_study_results: pd.DataFrame,
    external_stats: dict[str, object],
) -> None:
    official = official_standard_rows(summary)

    def row(task_id: str, model_id: str) -> pd.Series:
        return official[(official["task_id"] == task_id) & (official["model_id"] == model_id)].iloc[0]

    promoter_kmer = row("human_nontata_promoters", "kmer_logistic_regression")
    promoter_rc = row("human_nontata_promoters", "small_cnn_rc_aug")
    dnabert_enh = row("human_enhancers_cohn", "dnabert2")
    caduceus_prom = row("human_nontata_promoters", "caduceus_ph")
    caduceus_enh = row("human_enhancers_cohn", "caduceus_ph")
    synth_corr = synthetic_extended[(synthetic_extended["task_id"] == "gc_correlated") & (synthetic_extended["model_id"] == "kmer_logistic_regression")].iloc[0]
    synth_match = synthetic_extended[(synthetic_extended["task_id"] == "gc_matched") & (synthetic_extended["model_id"] == "kmer_logistic_regression")].iloc[0]
    synth_conflict = synthetic_extended[(synthetic_extended["task_id"] == "gc_conflict") & (synthetic_extended["model_id"] == "dnabert2")].iloc[0]
    gcbin_promoter = gc_bin_results[(gc_bin_results["task_id"] == "human_nontata_promoters") & (gc_bin_results["model_id"] == "dnabert2")].iloc[0]
    promoter_gc_matched = matched_results[(matched_results["task_id"] == "human_nontata_promoters") & (matched_results["model_id"] == "gc_only")].iloc[0]
    promoter_temp = mitigation_results[
        (mitigation_results["task_id"] == "human_nontata_promoters")
        & (mitigation_results["model_id"] == "small_cnn_rc_aug")
        & (mitigation_results["intervention"] == "Temperature scaling")
    ].iloc[0]
    shortcut_external = external_prediction_summary[
        (external_prediction_summary["analysis"] == "GenomeCF Shortcut Score vs external biological reliability")
        & (external_prediction_summary["metric"] == "Spearman")
    ].iloc[0]
    auroc_external = external_prediction_summary[
        (external_prediction_summary["analysis"] == "Core AUROC vs external biological reliability")
        & (external_prediction_summary["metric"] == "Spearman")
    ].iloc[0]
    full_profile_r2 = external_prediction_summary[
        (external_prediction_summary["analysis"] == "Full GenomeCF profile model")
        & (external_prediction_summary["metric"] == "R^2")
    ].iloc[0]
    full_profile_advantage = external_prediction_summary[
        (external_prediction_summary["analysis"] == "Full GenomeCF profile vs AUROC model")
        & (external_prediction_summary["metric"] == "Delta R^2")
    ].iloc[0]
    case_temp = case_study_results[
        (case_study_results["model_id"] == "small_cnn")
        & (case_study_results["task_id"] == "mpra_bcl11a_enhancer")
        & (case_study_results["condition_label"] == "Temperature-scaled")
    ].iloc[0]
    case_myc = case_study_results[
        (case_study_results["model_id"] == "dnabert2")
        & (case_study_results["task_id"] == "mpra_myc_enhancer")
        & (case_study_results["condition_label"] == "Standard")
    ].iloc[0]
    payload = {
        "promoter_kmer_auroc": float(promoter_kmer["auroc"]),
        "promoter_kmer_rc_instability": float(promoter_kmer["rc_mean_abs_delta"]),
        "promoter_rc_aug_rc_instability": float(promoter_rc["rc_mean_abs_delta"]),
        "dnabert_enhancer_auroc": float(dnabert_enh["auroc"]),
        "dnabert_enhancer_mono_shuffle_drop": float(dnabert_enh["mono_positive_prob_drop"]),
        "caduceus_promoter_auroc": float(caduceus_prom["auroc"]),
        "caduceus_promoter_rc_instability": float(caduceus_prom["rc_mean_abs_delta"]),
        "caduceus_enhancer_auroc": float(caduceus_enh["auroc"]),
        "matched_promoter_gc_only_auroc": float(promoter_gc_matched["matched_auroc"]),
        "official_promoter_gc_only_auroc": float(promoter_gc_matched["original_auroc"]),
        "promoter_rc_aug_temperature_ece_before": float(promoter_temp["ece_before"]),
        "promoter_rc_aug_temperature_ece_after": float(promoter_temp["ece_after"]),
        "synthetic_gc_correlated_auroc": float(synth_corr["auroc"]),
        "synthetic_gc_correlated_motif_drop": float(synth_corr["motif_positive_prob_drop"]),
        "synthetic_gc_matched_mono_drop": float(synth_match["mono_positive_prob_drop"]),
        "synthetic_gc_matched_motif_drop": float(synth_match["motif_positive_prob_drop"]),
        "synthetic_gc_conflict_dnabert_rule_rate": float(synth_conflict["rule_following_rate"]),
        "synthetic_gc_conflict_dnabert_shortcut_rate": float(synth_conflict["shortcut_following_rate"]),
        "promoter_dnabert_worst_gc_bin_auroc": float(gcbin_promoter["worst_bin_auroc"]),
        "promoter_dnabert_gc_gap": float(gcbin_promoter["gc_bin_auroc_gap"]),
        "external_shortcut_vs_reliability_spearman": float(shortcut_external["value"]),
        "external_shortcut_vs_reliability_spearman_ci_low": float(shortcut_external["ci_low"]),
        "external_shortcut_vs_reliability_spearman_ci_high": float(shortcut_external["ci_high"]),
        "external_auroc_vs_reliability_spearman": float(auroc_external["value"]),
        "external_pair_count": int(external_stats.get("pair_count", 0)),
        "external_full_profile_r2": float(full_profile_r2["value"]),
        "external_full_profile_r2_ci_low": float(full_profile_r2["ci_low"]),
        "external_full_profile_r2_ci_high": float(full_profile_r2["ci_high"]),
        "external_full_profile_advantage_delta_r2": float(full_profile_advantage["value"]),
        "external_full_profile_advantage_ci_low": float(full_profile_advantage["ci_low"]),
        "external_full_profile_advantage_ci_high": float(full_profile_advantage["ci_high"]),
        "case_bcl11a_temp_auroc": float(case_temp["auroc"]),
        "case_bcl11a_temp_auprc": float(case_temp["auprc"]),
        "case_bcl11a_temp_topk_enrichment": float(case_temp["topk_enrichment"]),
        "case_myc_dnabert_topk_enrichment": float(case_myc["topk_enrichment"]),
        "case_myc_dnabert_spearman": float(case_myc["spearman_abs_effect"]),
    }
    (PUBLICATION_ROOT / "key_numbers.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    PUBLICATION_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    release = load_release_inputs()
    summary = release["summary"]
    cv_summary = release["cv_summary"]
    cv_folds = release["cv_folds"]
    matched_confounders = release["matched_confounders"]
    registry = release["registry"]
    matched_models = release["matched_models"]
    mitigation = release["mitigation"]
    motif = release["motif"]
    motif_probe_summary = release["motif_probe_summary"]
    gc_bin_summary = release["gc_bin_summary"]
    gc_bin_by_bin = release["gc_bin_by_bin"]
    external_gc_bin_summary = release["external_gc_bin_summary"]
    external_gc_bin_by_bin = release["external_gc_bin_by_bin"]
    external_validation_summary = release["external_validation_summary"]
    external_validation_family_summary = release["external_validation_family_summary"]
    external_transfer_prediction = release["external_transfer_prediction"]
    external_prediction_robustness = release["external_prediction_robustness"]
    external_case_study = release["external_case_study"]
    synthetic_extended = release["synthetic_extended"]
    external_stats = json.loads((RELEASE_ROOT / "external_transfer_stats.json").read_text())

    task_overview = build_task_overview_table(summary)
    main_results = build_main_results_table(summary)
    cv_main = build_cv_main_table(summary, cv_summary)
    appendix_real = build_appendix_real_results(summary)
    appendix_synth = build_synthetic_appendix(summary)
    appendix_gc_only = official_standard_rows(summary)[
        (official_standard_rows(summary)["task_id"].isin(FOCAL_TASKS))
        & (official_standard_rows(summary)["model_id"] == "gc_only")
    ].copy()
    confounders, chrom_folds = build_official_confounder_summary()
    matched_results = build_matched_negative_results(matched_models, matched_confounders)
    matched_confounder_table = build_matched_negative_confounder_table(matched_confounders)
    mitigation_results = build_mitigation_results(mitigation)
    motif_results = build_real_motif_results(motif_probe_summary, summary)
    gc_bin_results = build_gc_bin_results(gc_bin_summary)
    synthetic_extended_results = build_synthetic_extended_results(synthetic_extended)
    external_validation_results = build_external_validation_results(external_validation_family_summary)
    external_prediction_results = build_external_prediction_summary(external_transfer_prediction, external_stats)
    external_prediction_robustness_table = build_external_prediction_robustness_table(external_prediction_robustness)
    case_study_results = build_case_study_results(external_case_study)
    appendix_external_results = build_external_appendix_results(external_validation_summary)
    shortcut_task_results, shortcut_model_results = build_shortcut_score_results(summary, matched_results, gc_bin_results)

    for stem, frame in [
        ("table1_task_overview", task_overview),
        ("table2_main_results", main_results),
        ("table3_cv_summary", cv_main),
        ("table4_matched_negative_summary", matched_results),
        ("table5_mitigation_summary", mitigation_results),
        ("table6_gc_bin_summary", gc_bin_results),
        ("table7_motif_summary", motif_results),
        ("table8_external_validation_summary", external_validation_results),
        ("table9_external_prediction_summary", external_prediction_results),
        ("table10_case_study_summary", case_study_results),
        ("appendix_real_results", appendix_real),
        ("appendix_synthetic_results", appendix_synth),
        ("appendix_synthetic_extended", synthetic_extended_results),
        ("appendix_gc_only", appendix_gc_only),
        ("appendix_confounders", confounders),
        ("appendix_chrom_folds", chrom_folds),
        ("appendix_chromosome_cv_summary", cv_summary),
        ("appendix_chromosome_cv_folds", cv_folds),
        ("appendix_matched_negative_results", matched_results),
        ("appendix_matched_negative_confounders", matched_confounder_table),
        ("appendix_mitigation_results", mitigation_results),
        ("appendix_real_motif_results", motif_results),
        ("appendix_gc_bin_summary", gc_bin_results),
        ("appendix_gc_bin_by_bin", gc_bin_by_bin),
        ("appendix_external_validation_results", appendix_external_results),
        ("appendix_external_prediction_summary", external_prediction_results),
        ("appendix_external_prediction_robustness", external_prediction_robustness_table),
        ("appendix_case_study_results", case_study_results),
        ("appendix_external_gc_bin_summary", external_gc_bin_summary),
        ("appendix_external_gc_bin_by_bin", external_gc_bin_by_bin),
        ("appendix_shortcut_score_by_task", shortcut_task_results),
        ("appendix_shortcut_score_summary", shortcut_model_results),
    ]:
        save_dataframe(frame, stem)

    write_task_overview_tex(task_overview)
    write_main_results_tex(main_results)
    write_cv_main_tex(cv_main)
    write_matched_negative_main_tex(matched_results)
    write_mitigation_main_tex(mitigation_results)
    write_gc_bin_main_tex(gc_bin_results)
    write_motif_main_tex(motif_results)
    write_external_validation_main_tex(external_validation_results)
    write_external_prediction_main_tex(external_prediction_results)
    write_case_study_main_tex(case_study_results)
    write_appendix_real_tex(appendix_real)
    write_synthetic_appendix_tex(appendix_synth)
    write_synthetic_extended_tex(synthetic_extended_results)
    write_gc_only_tex(appendix_gc_only)
    write_confounder_tex(confounders, chrom_folds)
    write_cv_tex(cv_summary, cv_folds)
    write_matched_negative_tex(matched_results, matched_confounder_table)
    write_mitigation_tex(mitigation_results)
    write_motif_tex(motif_results)
    write_gc_bin_tex(gc_bin_results, gc_bin_by_bin)
    write_external_validation_tex(appendix_external_results)
    write_external_prediction_tex(external_prediction_results)
    write_external_prediction_robustness_tex(external_prediction_robustness_table)
    write_case_study_tex(case_study_results)
    write_external_gc_bin_tex(external_gc_bin_summary, external_gc_bin_by_bin)
    write_shortcut_score_tex(shortcut_task_results, shortcut_model_results)

    plot_tradeoff(summary)
    plot_foundation_comparison(summary, mitigation_results)
    plot_calibration(registry)
    plot_generalization_gap(summary, cv_summary)
    plot_gc_bin_robustness(gc_bin_results)
    plot_synthetic(synthetic_extended_results)
    plot_shortcut_score(shortcut_model_results)
    write_source_data(
        summary,
        main_results,
        mitigation_results,
        external_validation_results,
        external_prediction_results,
        external_prediction_robustness_table,
        case_study_results,
        synthetic_extended_results,
    )
    write_key_numbers(
        summary,
        matched_results,
        mitigation_results,
        gc_bin_results,
        synthetic_extended_results,
        external_prediction_results,
        case_study_results,
        external_stats,
    )

    manifest = {
        "generated_tables": [
            "table1_task_overview",
            "table2_main_results",
            "table3_cv_summary",
            "table4_matched_negative_summary",
            "table5_mitigation_summary",
            "table6_gc_bin_summary",
            "table7_motif_summary",
            "table8_external_validation_summary",
            "table9_external_prediction_summary",
            "table10_case_study_summary",
            "appendix_real_results",
            "appendix_synthetic_results",
            "appendix_synthetic_extended",
            "appendix_gc_only",
            "appendix_confounders",
            "appendix_chrom_folds",
            "appendix_chromosome_cv_summary",
            "appendix_chromosome_cv_folds",
            "appendix_matched_negative_results",
            "appendix_matched_negative_confounders",
            "appendix_mitigation_results",
            "appendix_real_motif_results",
            "appendix_gc_bin_summary",
            "appendix_gc_bin_by_bin",
            "appendix_external_validation_results",
            "appendix_external_prediction_summary",
            "appendix_external_prediction_robustness",
            "appendix_case_study_results",
            "appendix_external_gc_bin_summary",
            "appendix_external_gc_bin_by_bin",
            "appendix_shortcut_score_by_task",
            "appendix_shortcut_score_summary",
        ],
        "generated_figures": [
            "genomecf_tradeoff_publication.png",
            "genomecf_foundation_comparison.png",
            "genomecf_calibration_publication.png",
            "genomecf_generalization_gap.png",
            "genomecf_gc_bin_robustness.png",
            "genomecf_synthetic_publication.png",
            "genomecf_shortcut_score.png",
            "genomecf_external_validation.png",
            "genomecf_external_prediction.png",
            "genomecf_biological_case_study.png",
        ],
    }
    (PUBLICATION_ROOT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
