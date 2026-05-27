from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = PROJECT_ROOT / "package_src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from genomecf.release import build_release_registry
from genomecf.variant_tasks import VARIANT_TASKS


RESULTS_ROOT = PROJECT_ROOT / "results" / "release"
FIGURES_ROOT = PROJECT_ROOT / "figures"
PUBLICATION_ROOT = PROJECT_ROOT / "results" / "publication"
SEED = 2026
PAPER_MODELS = [
    "kmer_logistic_regression",
    "small_cnn",
    "small_cnn_rc_aug",
    "dnabert2",
    "caduceus_ph",
]
CORE_TASKS = [
    "human_nontata_promoters",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ocr_ensembl",
]
GUE_EXTERNAL_TASKS = [
    "gue_human_tf_0",
    "gue_human_tf_1",
    "gue_emp_h3k4me3",
    "gue_emp_h3k14ac",
]
VARIANT_TASK_IDS = list(sorted(VARIANT_TASKS))
TASK_LABELS = {
    "gue_human_tf_0": "External TF binding (human_tf_0)",
    "gue_human_tf_1": "External TF binding (human_tf_1)",
    "gue_emp_h3k4me3": "External histone mark (H3K4me3)",
    "gue_emp_h3k14ac": "External histone mark (H3K14ac)",
    "mpra_bcl11a_enhancer": "MPRA variant effect (BCL11A enhancer)",
    "mpra_f9_promoter": "MPRA variant effect (F9 promoter)",
    "mpra_hbb_promoter": "MPRA variant effect (HBB promoter)",
    "mpra_ldlr_promoter": "MPRA variant effect (LDLR promoter)",
    "mpra_myc_enhancer": "MPRA variant effect (MYC enhancer)",
}
FAMILY_MAP = {
    "gue_human_tf_0": "TF binding",
    "gue_human_tf_1": "TF binding",
    "gue_emp_h3k4me3": "Histone marks",
    "gue_emp_h3k14ac": "Histone marks",
    "mpra_bcl11a_enhancer": "Variant effect",
    "mpra_f9_promoter": "Variant effect",
    "mpra_hbb_promoter": "Variant effect",
    "mpra_ldlr_promoter": "Variant effect",
    "mpra_myc_enhancer": "Variant effect",
}
MODEL_LABELS = {
    "kmer_logistic_regression": "6-mer logistic regression",
    "small_cnn": "CNN",
    "small_cnn_rc_aug": "RC-aug CNN",
    "dnabert2": "DNABERT-2",
    "caduceus_ph": "Caduceus-Ph",
}
MODEL_FAMILY = {
    "kmer_logistic_regression": "k-mer baseline",
    "small_cnn": "CNN",
    "small_cnn_rc_aug": "CNN",
    "dnabert2": "foundation model",
    "caduceus_ph": "foundation model",
}
MODEL_COLORS = {
    "kmer_logistic_regression": "#184e77",
    "small_cnn": "#2a9d8f",
    "small_cnn_rc_aug": "#e76f51",
    "dnabert2": "#7b2cbf",
    "caduceus_ph": "#4d908e",
}
FAMILY_COLORS = {"TF binding": "#355070", "Histone marks": "#bc6c25", "Variant effect": "#7b2cbf"}
CASE_STUDY_TASKS = ["mpra_bcl11a_enhancer", "mpra_myc_enhancer"]


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


def bootstrap_ci(
    frame: pd.DataFrame,
    statistic_fn,
    *,
    seed: int = SEED,
    n_boot: int = 1000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    n_rows = len(frame)
    for _ in range(n_boot):
        idx = rng.integers(0, n_rows, size=n_rows)
        sample = frame.iloc[idx].reset_index(drop=True)
        value = statistic_fn(sample)
        if np.isfinite(value):
            estimates.append(float(value))
    if not estimates:
        return float("nan"), float("nan")
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def _rank_pct(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    working = series.astype(float)
    transformed = working if higher_is_better else -working
    return transformed.rank(method="average", pct=True, ascending=True)


def _corr_value(frame: pd.DataFrame, x: str, y: str, method: str) -> float:
    clean = frame[[x, y]].dropna()
    if len(clean) < 3:
        return float("nan")
    return float(clean.corr(method=method).iloc[0, 1])


def _corr_payload(frame: pd.DataFrame, x: str, y: str, *, method: str = "spearman") -> dict[str, float]:
    clean = frame[[x, y]].dropna().reset_index(drop=True)
    value = _corr_value(clean, x, y, method)
    ci_low, ci_high = bootstrap_ci(clean, lambda sample: _corr_value(sample, x, y, method))
    return {
        "value": value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(len(clean)),
        "method": method,
    }


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.allclose(np.std(y_true), 0.0):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def _partial_corr_payload(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    numeric_controls: list[str] | None = None,
    categorical_controls: list[str] | None = None,
) -> dict[str, float]:
    numeric_controls = numeric_controls or []
    categorical_controls = categorical_controls or []
    columns = [x, y] + numeric_controls + categorical_controls
    clean = frame[columns].dropna().reset_index(drop=True)
    if clean.empty:
        return {"value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0, "method": "pearson"}

    def _residualized(sample: pd.DataFrame) -> float:
        control_blocks: list[pd.DataFrame] = []
        if numeric_controls:
            control_blocks.append(sample[numeric_controls].astype(float))
        if categorical_controls:
            control_blocks.append(pd.get_dummies(sample[categorical_controls], drop_first=True, dtype=float))
        if control_blocks:
            controls = pd.concat(control_blocks, axis=1)
            x_model = LinearRegression().fit(controls, sample[x].to_numpy(dtype=float))
            y_model = LinearRegression().fit(controls, sample[y].to_numpy(dtype=float))
            x_resid = sample[x].to_numpy(dtype=float) - x_model.predict(controls)
            y_resid = sample[y].to_numpy(dtype=float) - y_model.predict(controls)
        else:
            x_resid = sample[x].to_numpy(dtype=float)
            y_resid = sample[y].to_numpy(dtype=float)
        if np.std(x_resid) == 0 or np.std(y_resid) == 0:
            return float("nan")
        return float(np.corrcoef(x_resid, y_resid)[0, 1])

    value = _residualized(clean)
    ci_low, ci_high = bootstrap_ci(clean, _residualized)
    return {
        "value": value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(len(clean)),
        "method": "pearson",
    }


def _fit_linear_r2(frame: pd.DataFrame, predictors: list[str], outcome: str) -> float:
    clean = frame[predictors + [outcome]].dropna().reset_index(drop=True)
    if clean.empty:
        return float("nan")
    X = clean[predictors].to_numpy(dtype=float)
    y = clean[outcome].to_numpy(dtype=float)
    model = LinearRegression().fit(X, y)
    return float(r2_score(y, model.predict(X)))


def _linear_fit(frame: pd.DataFrame, predictors: list[str], outcome: str) -> dict[str, object]:
    clean = frame[predictors + [outcome]].dropna().reset_index(drop=True)
    if clean.empty:
        return {
            "predictors": "+".join(predictors),
            "outcome": outcome,
            "r2": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0,
            "coef": [],
            "intercept": float("nan"),
        }

    X = clean[predictors].to_numpy(dtype=float)
    y = clean[outcome].to_numpy(dtype=float)
    model = LinearRegression().fit(X, y)
    ci_low, ci_high = bootstrap_ci(clean, lambda sample: _fit_linear_r2(sample, predictors, outcome))
    return {
        "predictors": "+".join(predictors),
        "outcome": outcome,
        "r2": float(r2_score(y, model.predict(X))),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(len(clean)),
        "coef": [float(value) for value in model.coef_],
        "intercept": float(model.intercept_),
    }


def _leave_one_family_out(frame: pd.DataFrame, predictors: list[str], outcome: str) -> dict[str, object]:
    clean = frame[predictors + [outcome, "external_family"]].dropna().reset_index(drop=True)
    if clean.empty:
        return {
            "predictors": "+".join(predictors),
            "outcome": outcome,
            "cv_r2": float("nan"),
            "folds": [],
        }
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    for family in sorted(clean["external_family"].unique()):
        train = clean[clean["external_family"] != family]
        test = clean[clean["external_family"] == family]
        if train.empty or test.empty:
            continue
        model = LinearRegression().fit(train[predictors].to_numpy(dtype=float), train[outcome].to_numpy(dtype=float))
        pred = model.predict(test[predictors].to_numpy(dtype=float))
        fold_r2 = _safe_r2(test[outcome].to_numpy(dtype=float), pred)
        fold_rows.append({"held_out_family": family, "n": int(len(test)), "r2": fold_r2})
        predictions.append(
            pd.DataFrame(
                {
                    "held_out_family": family,
                    "y_true": test[outcome].to_numpy(dtype=float),
                    "y_pred": pred,
                }
            )
        )
    if not predictions:
        return {"predictors": "+".join(predictors), "outcome": outcome, "cv_r2": float("nan"), "folds": fold_rows}
    pred_frame = pd.concat(predictions, ignore_index=True)
    return {
        "predictors": "+".join(predictors),
        "outcome": outcome,
        "cv_r2": _safe_r2(pred_frame["y_true"].to_numpy(dtype=float), pred_frame["y_pred"].to_numpy(dtype=float)),
        "folds": fold_rows,
    }


def _leave_one_group_out(
    frame: pd.DataFrame,
    group_col: str,
    predictors: list[str],
    outcome: str,
) -> dict[str, object]:
    clean = frame[predictors + [outcome, group_col]].dropna().reset_index(drop=True)
    if clean.empty:
        return {
            "predictors": "+".join(predictors),
            "outcome": outcome,
            "group_col": group_col,
            "cv_r2": float("nan"),
            "folds": [],
        }
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    for group in sorted(clean[group_col].unique()):
        train = clean[clean[group_col] != group]
        test = clean[clean[group_col] == group]
        if train.empty or test.empty:
            continue
        if len(train) < 3:
            continue
        model = LinearRegression().fit(train[predictors].to_numpy(dtype=float), train[outcome].to_numpy(dtype=float))
        pred = model.predict(test[predictors].to_numpy(dtype=float))
        fold_r2 = _safe_r2(test[outcome].to_numpy(dtype=float), pred)
        fold_rows.append({"held_out": group, "n": int(len(test)), "r2": fold_r2})
        predictions.append(
            pd.DataFrame(
                {
                    "held_out": group,
                    "y_true": test[outcome].to_numpy(dtype=float),
                    "y_pred": pred,
                }
            )
        )
    if not predictions:
        return {
            "predictors": "+".join(predictors),
            "outcome": outcome,
            "group_col": group_col,
            "cv_r2": float("nan"),
            "folds": fold_rows,
        }
    pred_frame = pd.concat(predictions, ignore_index=True)
    return {
        "predictors": "+".join(predictors),
        "outcome": outcome,
        "group_col": group_col,
        "cv_r2": _safe_r2(pred_frame["y_true"].to_numpy(dtype=float), pred_frame["y_pred"].to_numpy(dtype=float)),
        "folds": fold_rows,
    }


def _in_sample_permutation_pvalue(
    frame: pd.DataFrame,
    *,
    outcome: str,
    predictors_a: list[str],
    predictors_b: list[str],
    seed: int = SEED,
    n_perm: int = 2000,
) -> dict[str, float]:
    clean = frame[predictors_a + predictors_b + [outcome]].dropna().reset_index(drop=True)
    observed_delta = float(_fit_linear_r2(clean, predictors_b, outcome) - _fit_linear_r2(clean, predictors_a, outcome))
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_perm):
        shuffled = clean.copy()
        shuffled[outcome] = rng.permutation(shuffled[outcome].to_numpy())
        delta = _fit_linear_r2(shuffled, predictors_b, outcome) - _fit_linear_r2(shuffled, predictors_a, outcome)
        if np.isfinite(delta):
            deltas.append(float(delta))
    if not deltas:
        return {"observed_delta": observed_delta, "p_value": float("nan"), "n_perm": 0}
    hits = float((np.abs(deltas) >= abs(observed_delta)).sum())
    p_value = float((hits + 1.0) / (len(deltas) + 1.0))
    return {"observed_delta": observed_delta, "p_value": p_value, "n_perm": int(len(deltas))}


def _permutation_pvalue(
    frame: pd.DataFrame,
    *,
    outcome: str,
    predictors_a: list[str],
    predictors_b: list[str],
    seed: int = SEED,
    n_perm: int = 1000,
) -> dict[str, float]:
    clean = frame[predictors_a + predictors_b + [outcome, "external_family"]].dropna().reset_index(drop=True)
    observed_a = _leave_one_family_out(clean, predictors_a, outcome)["cv_r2"]
    observed_b = _leave_one_family_out(clean, predictors_b, outcome)["cv_r2"]
    observed_delta = float(observed_b - observed_a)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_perm):
        shuffled = clean.copy()
        shuffled[outcome] = rng.permutation(shuffled[outcome].to_numpy())
        a = _leave_one_family_out(shuffled, predictors_a, outcome)["cv_r2"]
        b = _leave_one_family_out(shuffled, predictors_b, outcome)["cv_r2"]
        if np.isfinite(a) and np.isfinite(b):
            deltas.append(float(b - a))
    if not deltas:
        return {"observed_delta": observed_delta, "p_value": float("nan"), "n_perm": 0}
    hits = float((np.abs(deltas) >= abs(observed_delta)).sum())
    p_value = float((hits + 1.0) / (len(deltas) + 1.0))
    return {"observed_delta": observed_delta, "p_value": p_value, "n_perm": int(len(deltas))}


def _family_stratified_regression(frame: pd.DataFrame, predictors: list[str], outcome: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family, subset in frame.groupby("external_family", sort=True):
        fit = _linear_fit(subset, predictors, outcome)
        rows.append(
            {
                "external_family": family,
                "predictors": "+".join(predictors),
                "outcome": outcome,
                "r2": fit["r2"],
                "ci_low": fit["ci_low"],
                "ci_high": fit["ci_high"],
                "n": fit["n"],
            }
        )
    return rows


def _bootstrap_r2_advantage(
    frame: pd.DataFrame,
    *,
    outcome: str,
    predictors_a: list[str],
    predictors_b: list[str],
    seed: int = SEED,
    n_boot: int = 1000,
) -> dict[str, float]:
    clean = frame[predictors_a + predictors_b + [outcome]].dropna().reset_index(drop=True)
    if clean.empty:
        return {"observed_delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
    fit_a = _linear_fit(clean, predictors_a, outcome)
    fit_b = _linear_fit(clean, predictors_b, outcome)
    observed_delta = float(fit_b["r2"] - fit_a["r2"])
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    n_rows = len(clean)
    for _ in range(n_boot):
        idx = rng.integers(0, n_rows, size=n_rows)
        sample = clean.iloc[idx].reset_index(drop=True)
        delta = _fit_linear_r2(sample, predictors_b, outcome) - _fit_linear_r2(sample, predictors_a, outcome)
        if np.isfinite(delta):
            deltas.append(float(delta))
    if not deltas:
        return {"observed_delta": observed_delta, "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
    return {
        "observed_delta": observed_delta,
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "n_boot": int(len(deltas)),
    }


def condition_label(calibration_method: str, intervention_id: str) -> str:
    if intervention_id == "matched_negative_retraining":
        return "Matched-negative head"
    if intervention_id == "gc_balanced":
        return "GC-balanced"
    if calibration_method == "temperature":
        return "Temperature-scaled"
    if calibration_method == "isotonic":
        return "Isotonic"
    return "Standard"


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "summary": pd.read_csv(RESULTS_ROOT / "benchmark_summary.csv"),
        "gc_bin_summary": pd.read_csv(RESULTS_ROOT / "gc_bin_summary.csv"),
        "external_gc_bin_summary": pd.read_csv(RESULTS_ROOT / "external_gc_bin_summary.csv"),
    }


def load_variant_summaries() -> pd.DataFrame:
    files = sorted((RESULTS_ROOT / "variant_effect").glob("*_summary.csv"))
    if not files:
        return pd.DataFrame()
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    frame["task_label"] = frame["task_id"].map(TASK_LABELS)
    frame["external_family"] = frame["task_id"].map(FAMILY_MAP)
    frame["model_label"] = frame["model_id"].map(MODEL_LABELS).fillna(frame["model_id"])
    frame["variant_condition"] = [
        condition_label(calibration, intervention)
        for calibration, intervention in zip(frame["calibration_method"], frame["intervention_id"])
    ]
    return frame


def standard_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[(summary["calibration_method"] == "none") & (summary["intervention_id"] == "standard")].copy()


def build_external_classification_points(summary: pd.DataFrame, external_gc: pd.DataFrame) -> pd.DataFrame:
    official = summary[
        (summary["tier"] == "external")
        & (summary["task_id"].isin(GUE_EXTERNAL_TASKS))
        & (summary["model_id"].isin(PAPER_MODELS))
        & (summary["split_id"] == "official")
    ].copy()
    matched = summary[
        (summary["tier"] == "external")
        & (summary["task_id"].isin(GUE_EXTERNAL_TASKS))
        & (summary["model_id"].isin(PAPER_MODELS))
        & (summary["split_id"] == "matched_test")
    ][
        [
            "task_id",
            "model_id",
            "calibration_method",
            "intervention_id",
            "auroc",
            "auprc",
            "ece",
            "brier",
        ]
    ].rename(
        columns={
            "auroc": "matched_auroc",
            "auprc": "matched_auprc",
            "ece": "matched_ece",
            "brier": "matched_brier",
        }
    )
    points = official.merge(
        matched,
        on=["task_id", "model_id", "calibration_method", "intervention_id"],
        how="left",
    ).merge(
        external_gc[
            [
                "task_id",
                "model_id",
                "overall_auroc",
                "overall_ece",
                "overall_brier",
                "worst_bin_auroc",
                "best_bin_auroc",
                "gc_bin_auroc_gap",
                "worst_bin_ece",
                "gc_bin_ece_gap",
            ]
        ].rename(
            columns={
                "overall_auroc": "gc_overall_auroc",
                "overall_ece": "gc_overall_ece",
                "overall_brier": "gc_overall_brier",
            }
        ),
        on=["task_id", "model_id"],
        how="left",
    )
    points["task_label"] = points["task_id"].map(TASK_LABELS)
    points["external_family"] = points["task_id"].map(FAMILY_MAP)
    points["model_label"] = points["model_id"].map(MODEL_LABELS)
    points["condition_label"] = [
        condition_label(calibration, intervention)
        for calibration, intervention in zip(points["calibration_method"], points["intervention_id"])
    ]
    points["spearman_abs_effect"] = np.nan
    points["pearson_abs_effect"] = np.nan
    points["topk_precision"] = np.nan
    points["topk_enrichment"] = np.nan
    points["matched_negative_shift"] = points["auroc"] - points["matched_auroc"]
    points["matched_negative_abs_shift"] = points["matched_negative_shift"].abs()
    return points


def build_variant_points(summary: pd.DataFrame, variant_summary: pd.DataFrame) -> pd.DataFrame:
    if variant_summary.empty:
        return pd.DataFrame()
    official = summary[
        (summary["task_id"].isin(VARIANT_TASK_IDS))
        & (summary["split_id"] == "official")
        & (summary["model_id"].isin(["kmer_logistic_regression", "small_cnn", "small_cnn_rc_aug", "dnabert2", "caduceus_ph"]))
    ].copy()
    points = official.merge(
        variant_summary[
            [
                "task_id",
                "model_id",
                "split_id",
                "calibration_method",
                "intervention_id",
                "spearman_abs_effect",
                "pearson_abs_effect",
                "topk_precision",
                "topk_enrichment",
                "worst_bin_auroc",
                "best_bin_auroc",
                "gc_bin_auroc_gap",
                "worst_bin_ece",
                "best_bin_ece",
                "gc_bin_ece_gap",
                "embedding_dim",
                "model_checkpoint",
                "tokenizer_name",
            ]
        ],
        on=["task_id", "model_id", "split_id", "calibration_method", "intervention_id"],
        how="left",
        suffixes=("", "_variant"),
    )
    points["task_label"] = points["task_id"].map(TASK_LABELS)
    points["external_family"] = "Variant effect"
    points["model_label"] = points["model_id"].map(MODEL_LABELS).fillna(points["model_id"])
    points["condition_label"] = [
        condition_label(calibration, intervention)
        for calibration, intervention in zip(points["calibration_method"], points["intervention_id"])
    ]
    points["matched_auroc"] = np.nan
    points["matched_auprc"] = np.nan
    points["matched_ece"] = np.nan
    points["matched_brier"] = np.nan
    points["matched_negative_shift"] = np.nan
    points["matched_negative_abs_shift"] = np.nan
    points["gc_overall_auroc"] = points["auroc"]
    points["gc_overall_ece"] = points["ece"]
    points["gc_overall_brier"] = points["brier"]
    return points


def build_core_profiles(summary: pd.DataFrame, gc_bin_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    core = summary[
        (summary["tier"] == "core")
        & (summary["task_id"].isin(CORE_TASKS))
        & (summary["split_id"] == "official")
    ].copy()
    gc_lookup = gc_bin_summary[
        gc_bin_summary["task_id"].isin(CORE_TASKS) & gc_bin_summary["model_id"].isin(PAPER_MODELS)
    ].groupby("model_id", as_index=False).agg(
        core_worst_gc_bin_auroc=("worst_bin_auroc", "mean"),
        core_gc_bin_auroc_gap=("gc_bin_auroc_gap", "mean"),
        core_worst_gc_bin_ece=("worst_bin_ece", "mean"),
    )

    matched_shift_rows: list[dict[str, object]] = []
    matched_subset = core[core["split_id"].isin(["official", "matched_test"])] if "matched_test" in core["split_id"].unique() else summary[
        (summary["tier"] == "core")
        & (summary["task_id"].isin(CORE_TASKS))
        & (summary["split_id"].isin(["official", "matched_test"]))
    ].copy()
    for keys, group in matched_subset.groupby(["task_id", "model_id", "calibration_method", "intervention_id"]):
        official = group[group["split_id"] == "official"]
        matched = group[group["split_id"] == "matched_test"]
        if official.empty or matched.empty:
            continue
        matched_shift_rows.append(
            {
                "task_id": keys[0],
                "model_id": keys[1],
                "calibration_method": keys[2],
                "intervention_id": keys[3],
                "matched_auroc_drop": float(official.iloc[0]["auroc"] - matched.iloc[0]["auroc"]),
            }
        )
    matched_shift = pd.DataFrame(matched_shift_rows)

    profiles = (
        core.groupby(["model_id", "calibration_method", "intervention_id"], as_index=False)
        .agg(
            core_mean_auroc=("auroc", "mean"),
            core_mean_auprc=("auprc", "mean"),
            core_mean_ece=("ece", "mean"),
            core_mean_brier=("brier", "mean"),
            core_mean_rc_delta=("rc_mean_abs_delta", "mean"),
            core_mean_rc_flip=("rc_flip_rate", "mean"),
            core_mean_mono_drop=("mono_positive_prob_drop", "mean"),
            core_mean_dinuc_drop=("dinuc_positive_prob_drop", "mean"),
            core_mean_shortcut_score=("shortcut_score", "mean"),
        )
        .merge(
            matched_shift.groupby(["model_id", "calibration_method", "intervention_id"], as_index=False).agg(
                core_matched_negative_shift=("matched_auroc_drop", "mean")
            ),
            on=["model_id", "calibration_method", "intervention_id"],
            how="left",
        )
        .merge(gc_lookup, on="model_id", how="left")
    )
    standard_profile = profiles[
        (profiles["calibration_method"] == "none") & (profiles["intervention_id"] == "standard")
    ].copy()
    return profiles, standard_profile


def attach_core_profiles(points: pd.DataFrame, profiles: pd.DataFrame, standard_profile: pd.DataFrame) -> pd.DataFrame:
    merged = points.merge(
        profiles,
        on=["model_id", "calibration_method", "intervention_id"],
        how="left",
    )
    fallback = standard_profile.drop(columns=["calibration_method", "intervention_id"]).rename(
        columns={column: f"{column}_fallback" for column in standard_profile.columns if column not in {"model_id"}}
    )
    merged = merged.merge(fallback, on="model_id", how="left")
    for column in [
        "core_mean_auroc",
        "core_mean_auprc",
        "core_mean_ece",
        "core_mean_brier",
        "core_mean_rc_delta",
        "core_mean_rc_flip",
        "core_mean_mono_drop",
        "core_mean_dinuc_drop",
        "core_mean_shortcut_score",
        "core_matched_negative_shift",
        "core_worst_gc_bin_auroc",
        "core_gc_bin_auroc_gap",
        "core_worst_gc_bin_ece",
    ]:
        fallback_column = f"{column}_fallback"
        if fallback_column in merged.columns:
            merged[column] = merged[column].fillna(merged[fallback_column])
            merged = merged.drop(columns=[fallback_column])
    return merged


def add_external_scores(points: pd.DataFrame) -> pd.DataFrame:
    enriched_rows: list[pd.DataFrame] = []
    for task_id, group in points.groupby("task_id", as_index=False):
        task = group.copy()
        performance_components = [
            _rank_pct(task["auroc"], higher_is_better=True),
            _rank_pct(task["auprc"], higher_is_better=True),
            _rank_pct(task["worst_bin_auroc"], higher_is_better=True),
        ]
        if task["spearman_abs_effect"].notna().any():
            performance_components.append(_rank_pct(task["spearman_abs_effect"], higher_is_better=True))
        if task["topk_enrichment"].notna().any():
            performance_components.append(_rank_pct(task["topk_enrichment"], higher_is_better=True))
        if task["matched_auroc"].notna().any():
            performance_components.append(_rank_pct(task["matched_auroc"], higher_is_better=True))
        task["external_biological_reliability"] = pd.concat(performance_components, axis=1).mean(axis=1)

        risk_components = [
            _rank_pct(task["ece"], higher_is_better=False),
            _rank_pct(task["brier"], higher_is_better=False),
            _rank_pct(task["rc_mean_abs_delta"], higher_is_better=False),
            _rank_pct(task["rc_flip_rate"], higher_is_better=False),
            _rank_pct(task["gc_bin_auroc_gap"], higher_is_better=False),
            _rank_pct(task["worst_bin_auroc"], higher_is_better=True),
            _rank_pct(task["mono_positive_prob_drop"], higher_is_better=True),
            _rank_pct(task["dinuc_positive_prob_drop"], higher_is_better=True),
        ]
        if task["matched_negative_abs_shift"].notna().any():
            risk_components.append(_rank_pct(task["matched_negative_abs_shift"], higher_is_better=False))
        task["external_reliability_risk"] = pd.concat(risk_components, axis=1).mean(axis=1)
        enriched_rows.append(task)
    return pd.concat(enriched_rows, ignore_index=True)


def build_external_validation_summary(points: pd.DataFrame) -> pd.DataFrame:
    standard = points[
        (points["condition_label"] == "Standard") & (points["model_id"].isin(PAPER_MODELS))
    ].copy()
    return (
        standard.groupby(["external_family", "model_id", "model_label"], as_index=False)
        .agg(
            official_auroc=("auroc", "mean"),
            official_auprc=("auprc", "mean"),
            official_ece=("ece", "mean"),
            official_brier=("brier", "mean"),
            worst_bin_auroc=("worst_bin_auroc", "mean"),
            gc_bin_auroc_gap=("gc_bin_auroc_gap", "mean"),
            spearman_abs_effect=("spearman_abs_effect", "mean"),
            topk_enrichment=("topk_enrichment", "mean"),
            matched_negative_shift=("matched_negative_shift", "mean"),
            external_biological_reliability=("external_biological_reliability", "mean"),
            external_reliability_risk=("external_reliability_risk", "mean"),
            task_config_pairs=("task_id", "count"),
        )
        .sort_values(["external_family", "model_id"])
        .reset_index(drop=True)
    )


def build_external_prediction_analysis(points: pd.DataFrame) -> dict[str, object]:
    frame = points[
        [
            "task_id",
            "task_label",
            "external_family",
            "model_id",
            "model_label",
            "condition_label",
            "external_biological_reliability",
            "external_reliability_risk",
            "matched_negative_shift",
            "core_mean_auroc",
            "core_mean_shortcut_score",
            "core_mean_rc_delta",
            "core_mean_ece",
            "core_matched_negative_shift",
            "core_gc_bin_auroc_gap",
        ]
    ].copy()
    frame["configuration_label"] = frame["model_label"] + " | " + frame["condition_label"]
    frame["model_family"] = frame["model_id"].map(MODEL_FAMILY).fillna("other")

    auroc_corr = _corr_payload(frame, "core_mean_auroc", "external_biological_reliability", method="spearman")
    shortcut_corr = _corr_payload(frame, "core_mean_shortcut_score", "external_biological_reliability", method="spearman")
    risk_auroc_corr = _corr_payload(frame, "core_mean_auroc", "external_reliability_risk", method="spearman")
    risk_shortcut_corr = _corr_payload(frame, "core_mean_shortcut_score", "external_reliability_risk", method="spearman")
    matched_auroc_corr = _corr_payload(frame, "core_mean_auroc", "matched_negative_shift", method="pearson")
    matched_shortcut_corr = _corr_payload(frame, "core_mean_shortcut_score", "matched_negative_shift", method="pearson")

    partial_shortcut = _partial_corr_payload(
        frame,
        "core_mean_shortcut_score",
        "external_biological_reliability",
        numeric_controls=["core_mean_auroc"],
        categorical_controls=["model_id", "external_family"],
    )
    partial_shortcut_by_family = _partial_corr_payload(
        frame,
        "core_mean_shortcut_score",
        "external_biological_reliability",
        numeric_controls=["core_mean_auroc"],
        categorical_controls=["model_family", "external_family"],
    )
    partial_gc_gap = _partial_corr_payload(
        frame,
        "core_gc_bin_auroc_gap",
        "external_reliability_risk",
        numeric_controls=["core_mean_auroc"],
        categorical_controls=["model_id", "external_family"],
    )

    regression_specs = [
        ["core_mean_auroc"],
        ["core_mean_shortcut_score"],
        ["core_mean_auroc", "core_mean_shortcut_score"],
        [
            "core_mean_auroc",
            "core_mean_rc_delta",
            "core_mean_ece",
            "core_matched_negative_shift",
            "core_gc_bin_auroc_gap",
        ],
    ]
    full_profile_predictors = [
        "core_mean_auroc",
        "core_mean_rc_delta",
        "core_mean_ece",
        "core_matched_negative_shift",
        "core_gc_bin_auroc_gap",
    ]

    regression = [_linear_fit(frame, predictors, "external_biological_reliability") for predictors in regression_specs]
    lofo = [_leave_one_family_out(frame, predictors, "external_biological_reliability") for predictors in regression_specs]
    loto = [_leave_one_group_out(frame, "task_id", predictors, "external_biological_reliability") for predictors in regression_specs]

    within_family_task_cv: list[dict[str, object]] = []
    for family, subset in frame.groupby("external_family", sort=True):
        if subset["task_id"].nunique() < 3:
            continue
        within_family_task_cv.append(
            {
                "external_family": family,
                "results": [
                    _leave_one_group_out(subset, "task_id", predictors, "external_biological_reliability")
                    for predictors in regression_specs
                ],
            }
        )

    family_stratified = [
        * _family_stratified_regression(frame, ["core_mean_auroc"], "external_biological_reliability"),
        * _family_stratified_regression(frame, full_profile_predictors, "external_biological_reliability"),
    ]

    model_family_stratified: list[dict[str, object]] = []
    for model_family, subset in frame.groupby("model_family", sort=True):
        model_family_stratified.append(
            {
                "model_family": model_family,
                "auroc_only": _linear_fit(subset, ["core_mean_auroc"], "external_biological_reliability"),
                "shortcut_only": _linear_fit(subset, ["core_mean_shortcut_score"], "external_biological_reliability"),
                "full_profile": _linear_fit(subset, full_profile_predictors, "external_biological_reliability"),
                "n": int(len(subset.dropna(subset=["external_biological_reliability"]))),
            }
        )

    full_profile_advantage = _bootstrap_r2_advantage(
        frame,
        outcome="external_biological_reliability",
        predictors_a=["core_mean_auroc"],
        predictors_b=full_profile_predictors,
    )
    shortcut_lofo_permutation = _permutation_pvalue(
        frame,
        outcome="external_biological_reliability",
        predictors_a=["core_mean_auroc"],
        predictors_b=["core_mean_shortcut_score"],
    )
    full_profile_lofo_permutation = _permutation_pvalue(
        frame,
        outcome="external_biological_reliability",
        predictors_a=["core_mean_auroc"],
        predictors_b=full_profile_predictors,
    )
    full_profile_in_sample_permutation = _in_sample_permutation_pvalue(
        frame,
        outcome="external_biological_reliability",
        predictors_a=["core_mean_auroc"],
        predictors_b=full_profile_predictors,
    )

    return {
        "frame": frame,
        "stats": {
            "pair_count": int(len(frame)),
            "family_counts": frame["external_family"].value_counts().sort_index().to_dict(),
            "auroc_vs_external_reliability": auroc_corr,
            "shortcut_vs_external_reliability": shortcut_corr,
            "auroc_vs_external_risk": risk_auroc_corr,
            "shortcut_vs_external_risk": risk_shortcut_corr,
            "auroc_vs_external_matched_shift": matched_auroc_corr,
            "shortcut_vs_external_matched_shift": matched_shortcut_corr,
            "partial_shortcut_vs_external_reliability": partial_shortcut,
            "partial_shortcut_vs_external_reliability_controls": partial_shortcut_by_family,
            "partial_gc_gap_vs_external_risk": partial_gc_gap,
            "regression": regression,
            "leave_one_family_out": lofo,
            "leave_one_task_out": loto,
            "within_family_leave_one_task_out": within_family_task_cv,
            "family_stratified_regression": family_stratified,
            "model_family_stratified_regression": model_family_stratified,
            "full_profile_advantage": full_profile_advantage,
            "shortcut_permutation": shortcut_lofo_permutation,
            "permutation": full_profile_lofo_permutation,
            "in_sample_permutation": full_profile_in_sample_permutation,
        },
    }


def build_case_study(points: pd.DataFrame) -> pd.DataFrame:
    subset = points[
        (
            ((points["task_id"] == "mpra_bcl11a_enhancer") & (points["model_id"].isin(["small_cnn", "dnabert2"])))
            | ((points["task_id"] == "mpra_myc_enhancer") & (points["model_id"].isin(["kmer_logistic_regression", "dnabert2"])))
        )
        & (
            ((points["model_id"] == "small_cnn") & (points["condition_label"].isin(["Standard", "Temperature-scaled"])))
            | ((points["task_id"] == "mpra_myc_enhancer") & (points["model_id"].isin(["kmer_logistic_regression", "dnabert2"])) & (points["condition_label"] == "Standard"))
            | ((points["task_id"] == "mpra_bcl11a_enhancer") & (points["model_id"] == "dnabert2") & (points["condition_label"] == "Standard"))
        )
    ].copy()
    subset["decision_role"] = np.select(
        [
            (subset["task_id"] == "mpra_bcl11a_enhancer") & (subset["model_id"] == "small_cnn") & (subset["condition_label"] == "Standard"),
            (subset["task_id"] == "mpra_bcl11a_enhancer") & (subset["model_id"] == "small_cnn") & (subset["condition_label"] == "Temperature-scaled"),
            (subset["task_id"] == "mpra_myc_enhancer") & (subset["model_id"] == "kmer_logistic_regression"),
            (subset["task_id"] == "mpra_myc_enhancer") & (subset["model_id"] == "dnabert2"),
        ],
        [
            "AUROC-only default configuration",
            "GenomeCF-aware configuration",
            "AUROC-only variant-ranking choice",
            "GenomeCF-aware variant-ranking choice",
        ],
        default="Reference comparison",
    )
    subset["case_study_id"] = subset["task_id"].map(
        {
            "mpra_bcl11a_enhancer": "Case A",
            "mpra_myc_enhancer": "Case B",
        }
    )
    subset["case_study_message"] = subset["task_id"].map(
        {
            "mpra_bcl11a_enhancer": "Calibration-aware selection improves variant prioritization on BCL11A MPRA.",
            "mpra_myc_enhancer": "GenomeCF shifts variant-ranking preference from 6-mer AUROC to DNABERT-2 top-k enrichment on MYC MPRA.",
        }
    )
    return subset[
        [
            "case_study_id",
            "case_study_message",
            "task_id",
            "task_label",
            "external_family",
            "model_id",
            "model_label",
            "condition_label",
            "decision_role",
            "core_mean_auroc",
            "core_mean_ece",
            "core_mean_shortcut_score",
            "auroc",
            "auprc",
            "spearman_abs_effect",
            "topk_enrichment",
            "ece",
            "brier",
            "worst_bin_auroc",
            "gc_bin_auroc_gap",
        ]
    ].sort_values(["case_study_id", "task_id", "model_id", "condition_label"])


def plot_external_validation(family_summary: pd.DataFrame) -> None:
    configure_matplotlib()
    families = ["TF binding", "Histone marks", "Variant effect"]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2))
    metrics = [
        ("official_auroc", "External AUROC across assay families", "Official AUROC"),
        ("worst_bin_auroc", "External worst-GC-bin AUROC", "Worst-GC-bin AUROC"),
    ]
    width = 0.15
    x = np.arange(len(families))
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        for offset, model_id in enumerate(PAPER_MODELS):
            subset = family_summary[family_summary["model_id"] == model_id].set_index("external_family").reindex(families)
            ax.bar(
                x + (offset - 2) * width,
                subset[metric].to_numpy(),
                width=width,
                color=MODEL_COLORS[model_id],
                label=MODEL_LABELS[model_id],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(families)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.18)
    axes[0].legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "genomecf_external_validation.png", bbox_inches="tight")
    plt.close(fig)


def build_external_prediction_robustness(stats: dict[str, object]) -> pd.DataFrame:
    def _predictor_key(predictors: list[str]) -> str:
        return "+".join(predictors)

    auroc_key = _predictor_key(["core_mean_auroc"])
    shortcut_key = _predictor_key(["core_mean_shortcut_score"])
    full_key = _predictor_key([
        "core_mean_auroc",
        "core_mean_rc_delta",
        "core_mean_ece",
        "core_matched_negative_shift",
        "core_gc_bin_auroc_gap",
    ])

    reg = pd.DataFrame(stats.get("regression", []))
    lofo = pd.DataFrame(stats.get("leave_one_family_out", []))
    loto = pd.DataFrame(stats.get("leave_one_task_out", []))

    def _r2_from(df: pd.DataFrame, key: str, col: str) -> float:
        if df.empty:
            return float("nan")
        subset = df[df["predictors"] == key]
        if subset.empty:
            return float("nan")
        return float(subset.iloc[0][col])

    in_sample = {
        "analysis_type": "in-sample (pooled)",
        "n": int(stats.get("pair_count", 0)),
        "auroc_only_r2": _r2_from(reg, auroc_key, "r2"),
        "shortcut_only_r2": _r2_from(reg, shortcut_key, "r2"),
        "full_profile_r2": _r2_from(reg, full_key, "r2"),
    }
    in_sample["delta_full_vs_auroc"] = float(in_sample["full_profile_r2"] - in_sample["auroc_only_r2"])
    advantage = stats.get("full_profile_advantage", {})
    in_sample["delta_ci_low"] = float(advantage.get("ci_low", float("nan")))
    in_sample["delta_ci_high"] = float(advantage.get("ci_high", float("nan")))
    in_perm = stats.get("in_sample_permutation", {})
    in_sample["permutation_p"] = float(in_perm.get("p_value", float("nan")))
    in_sample["interpretation"] = "Primary analysis: full GenomeCF profile vs AUROC-only (pooled)."

    lofo_row = {
        "analysis_type": "leave-one-family-out",
        "n": int(stats.get("pair_count", 0)),
        "auroc_only_r2": _r2_from(lofo, auroc_key, "cv_r2"),
        "shortcut_only_r2": _r2_from(lofo, shortcut_key, "cv_r2"),
        "full_profile_r2": _r2_from(lofo, full_key, "cv_r2"),
    }
    lofo_row["delta_full_vs_auroc"] = float(lofo_row["full_profile_r2"] - lofo_row["auroc_only_r2"])
    lofo_row["delta_ci_low"] = float("nan")
    lofo_row["delta_ci_high"] = float("nan")
    lofo_perm = stats.get("permutation", {})
    lofo_row["permutation_p"] = float(lofo_perm.get("p_value", float("nan")))
    lofo_row["interpretation"] = "Hard setting: extrapolation to a held-out assay family."

    loto_row = {
        "analysis_type": "leave-one-task-out",
        "n": int(stats.get("pair_count", 0)),
        "auroc_only_r2": _r2_from(loto, auroc_key, "cv_r2"),
        "shortcut_only_r2": _r2_from(loto, shortcut_key, "cv_r2"),
        "full_profile_r2": _r2_from(loto, full_key, "cv_r2"),
    }
    loto_row["delta_full_vs_auroc"] = float(loto_row["full_profile_r2"] - loto_row["auroc_only_r2"])
    loto_row["delta_ci_low"] = float("nan")
    loto_row["delta_ci_high"] = float("nan")
    loto_row["permutation_p"] = float("nan")
    loto_row["interpretation"] = "Task-held-out prediction within the same mix of assay families."

    rows: list[dict[str, object]] = [in_sample, lofo_row, loto_row]

    within = stats.get("within_family_leave_one_task_out", [])
    for payload in within:
        family = payload.get("external_family", "")
        results = pd.DataFrame(payload.get("results", []))
        row = {
            "analysis_type": f"within-family leave-one-task-out ({family})",
            "n": int(stats.get("family_counts", {}).get(family, 0)),
            "auroc_only_r2": _r2_from(results, auroc_key, "cv_r2"),
            "shortcut_only_r2": _r2_from(results, shortcut_key, "cv_r2"),
            "full_profile_r2": _r2_from(results, full_key, "cv_r2"),
        }
        row["delta_full_vs_auroc"] = float(row["full_profile_r2"] - row["auroc_only_r2"])
        row["delta_ci_low"] = float("nan")
        row["delta_ci_high"] = float("nan")
        row["permutation_p"] = float("nan")
        row["interpretation"] = "Within-family robustness (assay-family specific)."
        rows.append(row)

    model_stratified = stats.get("model_family_stratified_regression", [])
    for payload in model_stratified:
        mfam = payload.get("model_family", "")
        row = {
            "analysis_type": f"in-sample model-family stratified ({mfam})",
            "n": int(payload.get("n", 0)),
            "auroc_only_r2": float(payload.get("auroc_only", {}).get("r2", float("nan"))),
            "shortcut_only_r2": float(payload.get("shortcut_only", {}).get("r2", float("nan"))),
            "full_profile_r2": float(payload.get("full_profile", {}).get("r2", float("nan"))),
        }
        row["delta_full_vs_auroc"] = float(row["full_profile_r2"] - row["auroc_only_r2"])
        row["delta_ci_low"] = float("nan")
        row["delta_ci_high"] = float("nan")
        row["permutation_p"] = float("nan")
        row["interpretation"] = "Checks robustness when restricting to a model family."
        rows.append(row)

    return pd.DataFrame(rows)


def plot_external_prediction(frame: pd.DataFrame, stats: dict[str, object]) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2))
    panels = [
        ("core_mean_auroc", "Held-out core AUROC"),
        ("core_mean_shortcut_score", "GenomeCF Shortcut Score"),
    ]
    for ax, (xcol, xlabel) in zip(axes[:2], panels):
        for family, subset in frame.groupby("external_family"):
            ax.scatter(
                subset[xcol],
                subset["external_biological_reliability"],
                s=64,
                alpha=0.85,
                label=family,
                color=FAMILY_COLORS[family],
                edgecolor="white",
                linewidth=0.6,
            )
        clean = frame[[xcol, "external_biological_reliability"]].dropna().sort_values(xcol)
        if len(clean) >= 2:
            coef = np.polyfit(clean[xcol].to_numpy(), clean["external_biological_reliability"].to_numpy(), deg=1)
            line = np.poly1d(coef)
            ax.plot(clean[xcol], line(clean[xcol]), color="#333333", linewidth=1.4, alpha=0.85)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("External biological reliability (higher is better)")
        metric_key = "auroc_vs_external_reliability" if xcol == "core_mean_auroc" else "shortcut_vs_external_reliability"
        payload = stats[metric_key]
        ax.set_title(
            f"{xlabel}\nSpearman {payload['value']:.2f} [{payload['ci_low']:.2f}, {payload['ci_high']:.2f}]"
        )
        ax.grid(alpha=0.18)
    axes[1].legend(frameon=False, loc="upper left")

    lofo = pd.DataFrame(stats["leave_one_family_out"])
    x = np.arange(len(lofo))
    axes[2].bar(
        x,
        lofo["cv_r2"].to_numpy(),
        color=["#355070", "#7b2cbf", "#577590", "#4d908e"],
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(
        ["AUROC", "Shortcut Score", "AUROC + Shortcut", "AUROC + components"],
        rotation=18,
        ha="right",
    )
    axes[2].set_ylabel("Leave-one-family-out $R^2$")
    axes[2].set_title("Cross-family prediction of external reliability")
    axes[2].grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "genomecf_external_prediction.png", bbox_inches="tight")
    plt.close(fig)


def plot_case_study(case_study: pd.DataFrame) -> None:
    configure_matplotlib()
    tasks = CASE_STUDY_TASKS
    metrics = [
        ("auprc", "External AUPRC"),
        ("topk_enrichment", "Top-k enrichment"),
        ("spearman_abs_effect", "Spearman |effect|"),
        ("worst_bin_auroc", "Worst-GC-bin AUROC"),
    ]
    fig, axes = plt.subplots(len(tasks), len(metrics), figsize=(15.5, 7.8), sharey="col")
    if len(tasks) == 1:
        axes = np.asarray([axes])
    for row_idx, task_id in enumerate(tasks):
        subset = case_study[case_study["task_id"] == task_id].copy()
        labels = [f"{row.model_label}\n{row.condition_label}" for row in subset.itertuples()]
        x = np.arange(len(subset))
        colors = [MODEL_COLORS[row.model_id] for row in subset.itertuples()]
        for col_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            ax.bar(x, subset[metric].to_numpy(dtype=float), color=colors)
            if row_idx == 0:
                ax.set_title(ylabel)
            if col_idx == 0:
                ax.set_ylabel(TASK_LABELS[task_id].replace("MPRA variant effect ", "").replace(" (", "\n("))
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=18, ha="right")
            ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "genomecf_biological_case_study.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    PUBLICATION_ROOT.mkdir(parents=True, exist_ok=True)
    build_release_registry()
    release = load_inputs()
    summary = release["summary"]
    external_gc = release["external_gc_bin_summary"]
    variant_summary = load_variant_summaries()

    classification_points = build_external_classification_points(summary, external_gc)
    variant_points = build_variant_points(summary, variant_summary)
    external_points = pd.concat([classification_points, variant_points], ignore_index=True, sort=False)
    profiles, standard_profile = build_core_profiles(summary, release["gc_bin_summary"])
    external_points = attach_core_profiles(external_points, profiles, standard_profile)
    external_points = add_external_scores(external_points)
    family_summary = build_external_validation_summary(external_points)
    prediction_bundle = build_external_prediction_analysis(external_points)
    transfer = prediction_bundle["frame"]
    stats = prediction_bundle["stats"]
    robustness = build_external_prediction_robustness(stats)
    case_study = build_case_study(external_points)

    external_points.to_csv(RESULTS_ROOT / "external_validation_summary.csv", index=False)
    family_summary.to_csv(RESULTS_ROOT / "external_validation_family_summary.csv", index=False)
    profiles.to_csv(RESULTS_ROOT / "external_core_profile.csv", index=False)
    transfer.to_csv(RESULTS_ROOT / "external_transfer_prediction.csv", index=False)
    robustness.to_csv(RESULTS_ROOT / "external_prediction_robustness.csv", index=False)
    case_study.to_csv(RESULTS_ROOT / "biological_case_study.csv", index=False)
    (RESULTS_ROOT / "external_transfer_stats.json").write_text(json.dumps(stats, indent=2))

    plot_external_validation(family_summary)
    plot_external_prediction(transfer, stats)
    plot_case_study(case_study)

    manifest = {
        "tables": [
            "external_validation_summary.csv",
            "external_validation_family_summary.csv",
            "external_core_profile.csv",
            "external_transfer_prediction.csv",
            "external_prediction_robustness.csv",
            "biological_case_study.csv",
        ],
        "figures": [
            "genomecf_external_validation.png",
            "genomecf_external_prediction.png",
            "genomecf_biological_case_study.png",
        ],
        "stats": "external_transfer_stats.json",
    }
    (RESULTS_ROOT / "nature_methods_artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
