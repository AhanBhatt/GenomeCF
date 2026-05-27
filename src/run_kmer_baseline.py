from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from project_pipeline import (
    build_kmer_logistic_regression,
    bundle_to_frame,
    create_counterfactuals,
    evaluate_binary_classifier,
    load_dataset,
    save_json,
    summarize_counterfactual_behavior,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results" / "kmer_baseline"
FIGURES_ROOT = PROJECT_ROOT / "figures"
DATASETS = ["human_nontata_promoters", "human_enhancers_cohn"]


def plot_dataset_summary(summary_frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for idx, metric in enumerate(["length", "gc_fraction"]):
        ax = axes[idx]
        for dataset_name, group in summary_frame.groupby("dataset"):
            ax.hist(group[metric], bins=30, alpha=0.55, label=dataset_name)
        ax.set_title(metric.replace("_", " ").title())
        ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_counterfactual_summary(counterfactual_frame: pd.DataFrame, output_path: Path) -> None:
    metric_order = [
        "reverse_complement_mean_abs_delta",
        "reverse_complement_prediction_flip_rate",
        "positive_gc_shuffle_prob_drop",
    ]
    pretty_names = {
        "reverse_complement_mean_abs_delta": "RC mean abs delta",
        "reverse_complement_prediction_flip_rate": "RC flip rate",
        "positive_gc_shuffle_prob_drop": "Shuffle prob drop",
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, metric in zip(axes, metric_order):
        values = counterfactual_frame.set_index("dataset")[metric]
        ax.bar(values.index, values.values, color=["#204b87", "#4c9f70"])
        ax.set_title(pretty_names[metric])
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    dataset_rows: list[dict] = []
    model_rows: list[dict] = []
    counterfactual_rows: list[dict] = []

    for dataset_name in DATASETS:
        bundle = load_dataset(dataset_name, DATA_ROOT)
        frame = bundle_to_frame(bundle)
        dataset_rows.extend(frame.to_dict(orient="records"))

        model = build_kmer_logistic_regression(k=6)
        model.fit(bundle.train_sequences, bundle.train_labels)

        test_metrics = evaluate_binary_classifier(model, bundle.test_sequences, bundle.test_labels)
        counterfactuals = create_counterfactuals(bundle.test_sequences, seed=42)
        cf_metrics = summarize_counterfactual_behavior(
            model,
            bundle.test_sequences,
            bundle.test_labels,
            counterfactuals,
        )

        model_payload = {
            "dataset": dataset_name,
            "model": "kmer_logistic_regression",
            "kmer_size": 6,
            "test_auroc": test_metrics["auroc"],
            "test_accuracy": test_metrics["accuracy"],
        }
        model_rows.append(model_payload)
        counterfactual_rows.append({"dataset": dataset_name, **cf_metrics})

        save_json(
            {
                **model_payload,
                **cf_metrics,
            },
            RESULTS_ROOT / f"{dataset_name}_metrics.json",
        )

    dataset_frame = pd.DataFrame(dataset_rows)
    model_frame = pd.DataFrame(model_rows)
    counterfactual_frame = pd.DataFrame(counterfactual_rows)

    dataset_summary = (
        dataset_frame.groupby(["dataset", "split", "label"])[["length", "gc_fraction"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    dataset_summary.to_csv(RESULTS_ROOT / "dataset_summary.csv", index=False)
    model_frame.to_csv(RESULTS_ROOT / "model_metrics.csv", index=False)
    counterfactual_frame.to_csv(RESULTS_ROOT / "counterfactual_metrics.csv", index=False)

    plot_dataset_summary(
        dataset_frame[dataset_frame["split"] == "train"],
        FIGURES_ROOT / "dataset_length_gc_summary.png",
    )
    plot_counterfactual_summary(
        counterfactual_frame,
        FIGURES_ROOT / "kmer_counterfactual_summary.png",
    )

    print("Saved baseline metrics to", RESULTS_ROOT)
    print(model_frame.to_string(index=False))
    print(counterfactual_frame.to_string(index=False))


if __name__ == "__main__":
    main()
