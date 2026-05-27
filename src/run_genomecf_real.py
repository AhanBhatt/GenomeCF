from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from genomecf import (
    DEFAULT_BINARY_TASKS,
    DNABERT2Embedder,
    HOLDOUT_CHROMS,
    HUMAN_TASKS,
    build_kmer_logistic_regression,
    create_counterfactuals,
    evaluate_prediction_set,
    load_task_frame,
    predict_cnn_probabilities,
    save_json,
    train_cnn_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
REPO_ROOT = PROJECT_ROOT / "external" / "genomic_benchmarks"
RESULTS_ROOT = PROJECT_ROOT / "results" / "genomecf_real"
FIGURES_ROOT = PROJECT_ROOT / "figures"
DNABERT_DIR = PROJECT_ROOT / "external" / "dnabert2_local"

OFFICIAL_TRAIN_PER_LABEL = 3000
OFFICIAL_TEST_PER_LABEL = 1000
DNABERT_TRAIN_PER_LABEL = 1000
DNABERT_TEST_PER_LABEL = 500
CNN_TASKS = [
    "human_nontata_promoters",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ocr_ensembl",
]
DNABERT_TASKS = [
    "human_nontata_promoters",
    "human_enhancers_cohn",
]
SEEDS = [11, 23, 37, 49, 61]


def task_cache_key(dataset_name: str, split_scheme: str, train_per_label: int, test_per_label: int, seed: int) -> str:
    return f"{dataset_name}__{split_scheme}__tr{train_per_label}__te{test_per_label}__seed{seed}"


def get_task_frames() -> dict[str, object]:
    cache: dict[str, object] = {}
    for dataset_name in DEFAULT_BINARY_TASKS:
        cache[task_cache_key(dataset_name, "official", OFFICIAL_TRAIN_PER_LABEL, OFFICIAL_TEST_PER_LABEL, 2026)] = load_task_frame(
            dataset_name,
            DATA_ROOT,
            REPO_ROOT,
            split_scheme="official",
            seed=2026,
            train_per_label=OFFICIAL_TRAIN_PER_LABEL,
            test_per_label=OFFICIAL_TEST_PER_LABEL,
        )
    for dataset_name in HUMAN_TASKS:
        cache[task_cache_key(dataset_name, "chromosome_holdout", OFFICIAL_TRAIN_PER_LABEL, OFFICIAL_TEST_PER_LABEL, 2026)] = load_task_frame(
            dataset_name,
            DATA_ROOT,
            REPO_ROOT,
            split_scheme="chromosome_holdout",
            seed=2026,
            train_per_label=OFFICIAL_TRAIN_PER_LABEL,
            test_per_label=OFFICIAL_TEST_PER_LABEL,
        )
    for dataset_name in DNABERT_TASKS:
        cache[task_cache_key(dataset_name, "official", DNABERT_TRAIN_PER_LABEL, DNABERT_TEST_PER_LABEL, 2026)] = load_task_frame(
            dataset_name,
            DATA_ROOT,
            REPO_ROOT,
            split_scheme="official",
            seed=2026,
            train_per_label=DNABERT_TRAIN_PER_LABEL,
            test_per_label=DNABERT_TEST_PER_LABEL,
        )
    return cache


def run_kmer_suite(task_frames: dict[str, object]) -> tuple[list[dict], dict[tuple[str, str], dict[str, np.ndarray]]]:
    rows: list[dict] = []
    prediction_store: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for dataset_name in DEFAULT_BINARY_TASKS:
        for split_scheme in ["official"] + (["chromosome_holdout"] if dataset_name in HUMAN_TASKS else []):
            task = task_frames[task_cache_key(dataset_name, split_scheme, OFFICIAL_TRAIN_PER_LABEL, OFFICIAL_TEST_PER_LABEL, 2026)]
            model = build_kmer_logistic_regression(k=6, seed=42)
            model.fit(task.train["sequence"].tolist(), task.train["label"].tolist())

            original_probs = model.predict_proba(task.test["sequence"].tolist())[:, 1]
            counterfactuals = create_counterfactuals(task.test["sequence"].tolist(), seed=42)
            cf_probs = {name: model.predict_proba(seqs)[:, 1] for name, seqs in counterfactuals.items()}
            metrics = evaluate_prediction_set(task.test["label"].to_numpy(), original_probs, cf_probs, seed=42)
            row = {
                "dataset": dataset_name,
                "split_scheme": split_scheme,
                "model_family": "kmer_logistic_regression",
                "seed": 42,
                **metrics,
            }
            rows.append(row)
            prediction_store[(dataset_name, f"{split_scheme}::kmer_logistic_regression")] = {
                "original": original_probs,
                **cf_probs,
                "labels": task.test["label"].to_numpy(),
            }
            save_json(row, RESULTS_ROOT / "per_run" / f"{dataset_name}__{split_scheme}__kmer_logistic_regression.json")
            print("kmer", dataset_name, split_scheme, metrics["auroc"])

    return rows, prediction_store


def run_cnn_suite(task_frames: dict[str, object]) -> tuple[list[dict], dict[tuple[str, str], list[dict[str, np.ndarray]]], list[dict]]:
    rows: list[dict] = []
    prediction_store: dict[tuple[str, str], list[dict[str, np.ndarray]]] = defaultdict(list)
    history_rows: list[dict] = []

    for dataset_name in CNN_TASKS:
        task = task_frames[task_cache_key(dataset_name, "official", OFFICIAL_TRAIN_PER_LABEL, OFFICIAL_TEST_PER_LABEL, 2026)]
        for rc_augment, model_family in [(False, "small_cnn"), (True, "small_cnn_rc_aug")]:
            for seed in SEEDS:
                model, history = train_cnn_model(
                    task.train["sequence"].tolist(),
                    task.train["label"].tolist(),
                    seed=seed,
                    rc_augment=rc_augment,
                )
                for epoch_row in history:
                    history_rows.append(
                        {
                            "dataset": dataset_name,
                            "model_family": model_family,
                            "seed": seed,
                            **epoch_row,
                        }
                    )

                original_probs = predict_cnn_probabilities(model, task.test["sequence"].tolist())
                counterfactuals = create_counterfactuals(task.test["sequence"].tolist(), seed=seed)
                cf_probs = {name: predict_cnn_probabilities(model, seqs) for name, seqs in counterfactuals.items()}
                metrics = evaluate_prediction_set(task.test["label"].to_numpy(), original_probs, cf_probs, seed=seed)
                row = {
                    "dataset": dataset_name,
                    "split_scheme": "official",
                    "model_family": model_family,
                    "seed": seed,
                    **metrics,
                }
                rows.append(row)
                prediction_store[(dataset_name, model_family)].append(
                    {
                        "labels": task.test["label"].to_numpy(),
                        "original": original_probs,
                        **cf_probs,
                    }
                )
                save_json(
                    row,
                    RESULTS_ROOT / "per_run" / f"{dataset_name}__official__{model_family}__seed{seed}.json",
                )
                save_json(
                    {"history": history},
                    RESULTS_ROOT / "per_run" / f"{dataset_name}__official__{model_family}__seed{seed}__history.json",
                )
                print("cnn", dataset_name, model_family, seed, metrics["auroc"])

    return rows, prediction_store, history_rows


def run_dnabert_suite(task_frames: dict[str, object]) -> tuple[list[dict], dict[tuple[str, str], dict[str, np.ndarray]]]:
    rows: list[dict] = []
    prediction_store: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    embedder = DNABERT2Embedder(DNABERT_DIR)

    for dataset_name in DNABERT_TASKS:
        task = task_frames[task_cache_key(dataset_name, "official", DNABERT_TRAIN_PER_LABEL, DNABERT_TEST_PER_LABEL, 2026)]
        x_train = embedder.encode(task.train["sequence"].tolist(), batch_size=16)
        x_test = embedder.encode(task.test["sequence"].tolist(), batch_size=16)
        model = LogisticRegression(max_iter=400, solver="liblinear", random_state=42)
        model.fit(x_train, task.train["label"].to_numpy())

        original_probs = model.predict_proba(x_test)[:, 1]
        counterfactuals = create_counterfactuals(task.test["sequence"].tolist(), seed=42)
        cf_embeddings = {name: embedder.encode(seqs, batch_size=16) for name, seqs in counterfactuals.items()}
        cf_probs = {name: model.predict_proba(emb)[:, 1] for name, emb in cf_embeddings.items()}
        metrics = evaluate_prediction_set(task.test["label"].to_numpy(), original_probs, cf_probs, seed=42)
        row = {
            "dataset": dataset_name,
            "split_scheme": "official",
            "model_family": "dnabert2_embedding_logistic",
            "seed": 42,
            **metrics,
        }
        rows.append(row)
        prediction_store[(dataset_name, "official::dnabert2_embedding_logistic")] = {
            "labels": task.test["label"].to_numpy(),
            "original": original_probs,
            **cf_probs,
        }
        save_json(row, RESULTS_ROOT / "per_run" / f"{dataset_name}__official__dnabert2_embedding_logistic.json")
        print("dnabert2", dataset_name, metrics["auroc"])

    return rows, prediction_store


def summarize_seed_runs(rows: list[dict], prediction_store: dict[tuple[str, str], list[dict[str, np.ndarray]]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    summaries: list[dict] = []
    metric_columns = [col for col in frame.columns if col not in {"dataset", "split_scheme", "model_family", "seed"}]

    for (dataset_name, model_family), group in frame.groupby(["dataset", "model_family"]):
        summary = {
            "dataset": dataset_name,
            "split_scheme": group["split_scheme"].iloc[0],
            "model_family": model_family,
            "n_seeds": int(len(group)),
        }
        for metric in metric_columns:
            summary[f"{metric}_mean"] = float(group[metric].mean())
            summary[f"{metric}_std"] = float(group[metric].std(ddof=0))

        if (dataset_name, model_family) in prediction_store:
            preds = prediction_store[(dataset_name, model_family)]
            labels = preds[0]["labels"]
            ensemble_original = np.mean([item["original"] for item in preds], axis=0)
            ensemble_counterfactuals = {
                name: np.mean([item[name] for item in preds], axis=0)
                for name in ["reverse_complement", "mono_shuffle", "dinuc_shuffle"]
            }
            ensemble_metrics = evaluate_prediction_set(labels, ensemble_original, ensemble_counterfactuals, seed=2026)
            for metric, value in ensemble_metrics.items():
                summary[f"ensemble_{metric}"] = float(value)
        summaries.append(summary)
    return pd.DataFrame(summaries)


def summarize_deterministic_rows(frame: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict] = []
    metric_columns = [col for col in frame.columns if col not in {"dataset", "split_scheme", "model_family", "seed"}]
    for _, row in frame.iterrows():
        summary = {
            "dataset": row["dataset"],
            "split_scheme": row["split_scheme"],
            "model_family": row["model_family"],
            "n_seeds": 1,
        }
        for metric in metric_columns:
            summary[f"{metric}_mean"] = float(row[metric])
            summary[f"{metric}_std"] = 0.0
            summary[f"ensemble_{metric}"] = float(row[metric])
        summaries.append(summary)
    return pd.DataFrame(summaries)


def plot_benchmark_overview(summary_frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("ensemble_auroc", "Ensemble AUROC"),
        ("ensemble_reverse_complement_mean_abs_delta", "RC mean abs delta"),
        ("ensemble_mono_shuffle_positive_prob_drop", "Mono-shuffle positive drop"),
        ("ensemble_dinuc_shuffle_positive_prob_drop", "Dinuc-shuffle positive drop"),
    ]
    model_order = ["kmer_logistic_regression", "small_cnn", "small_cnn_rc_aug", "dnabert2_embedding_logistic"]
    colors = {
        "kmer_logistic_regression": "#204b87",
        "small_cnn": "#4c9f70",
        "small_cnn_rc_aug": "#d4842d",
        "dnabert2_embedding_logistic": "#8a3fb0",
    }

    for ax, (metric, title) in zip(axes.ravel(), metrics):
        subset = summary_frame[summary_frame["model_family"].isin(model_order) & (summary_frame["split_scheme"] == "official")]
        for idx, model_family in enumerate(model_order):
            model_subset = subset[subset["model_family"] == model_family]
            if model_subset.empty:
                continue
            ax.scatter(
                [idx] * len(model_subset),
                model_subset[metric],
                s=50,
                color=colors[model_family],
                label=model_family if title == metrics[0][1] else None,
            )
        ax.set_title(title)
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels(["k-mer", "CNN", "RC-aug CNN", "DNABERT-2"], rotation=15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, frameon=False, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    task_frames = get_task_frames()
    kmer_rows, _ = run_kmer_suite(task_frames)
    cnn_rows, cnn_prediction_store, history_rows = run_cnn_suite(task_frames)
    dnabert_rows, _ = run_dnabert_suite(task_frames)

    all_rows = kmer_rows + cnn_rows + dnabert_rows
    per_run_frame = pd.DataFrame(all_rows)
    per_run_frame.to_csv(RESULTS_ROOT / "per_run_metrics.csv", index=False)
    pd.DataFrame(history_rows).to_csv(RESULTS_ROOT / "cnn_history.csv", index=False)

    seed_summary = summarize_seed_runs(cnn_rows, cnn_prediction_store)
    deterministic_rows = per_run_frame[per_run_frame["model_family"].isin(["kmer_logistic_regression", "dnabert2_embedding_logistic"])]
    deterministic_summary = summarize_deterministic_rows(deterministic_rows)

    summary_frame = pd.concat([seed_summary, deterministic_summary], ignore_index=True, sort=False)
    summary_frame.to_csv(RESULTS_ROOT / "summary_metrics.csv", index=False)
    plot_benchmark_overview(summary_frame, FIGURES_ROOT / "genomecf_overview.png")

    summary_payload = {
        "tasks_official": DEFAULT_BINARY_TASKS,
        "tasks_chromosome_holdout": sorted(HUMAN_TASKS),
        "holdout_chromosomes": list(HOLDOUT_CHROMS),
        "cnn_seeds": SEEDS,
    }
    save_json(summary_payload, RESULTS_ROOT / "run_config.json")
    print(summary_frame[["dataset", "split_scheme", "model_family", "n_seeds"] + [c for c in summary_frame.columns if c.startswith("ensemble_auroc")][:1]].to_string(index=False))


if __name__ == "__main__":
    main()
