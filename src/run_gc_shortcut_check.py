from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from project_pipeline import bundle_to_frame, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results" / "gc_shortcut_check"
FIGURES_ROOT = PROJECT_ROOT / "figures"
DATASETS = ["human_nontata_promoters", "human_enhancers_cohn"]


def fit_gc_baseline(train_gc: np.ndarray, train_labels: list[int], test_gc: np.ndarray, test_labels: list[int]) -> dict[str, float]:
    model = LogisticRegression(random_state=42)
    model.fit(train_gc.reshape(-1, 1), train_labels)
    probs = model.predict_proba(test_gc.reshape(-1, 1))[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "test_auroc": float(roc_auc_score(test_labels, probs)),
        "test_accuracy": float(accuracy_score(test_labels, preds)),
        "coef": float(model.coef_[0][0]),
        "intercept": float(model.intercept_[0]),
    }


def plot_gc_distributions(frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, dataset_name in zip(axes, DATASETS):
        subset = frame[(frame["dataset"] == dataset_name) & (frame["split"] == "test")]
        for label, color, label_name in [(0, "#204b87", "Negative"), (1, "#b95c2e", "Positive")]:
            values = subset[subset["label"] == label]["gc_fraction"]
            ax.hist(values, bins=30, alpha=0.55, label=label_name, color=color)
        ax.set_title(dataset_name)
        ax.set_xlabel("GC fraction")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Count")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | str]] = []

    for dataset_name in DATASETS:
        bundle = load_dataset(dataset_name, DATA_ROOT)
        frame = bundle_to_frame(bundle)
        all_frames.append(frame)

        train_gc = np.array([(seq.count("G") + seq.count("C")) / len(seq) for seq in bundle.train_sequences])
        test_gc = np.array([(seq.count("G") + seq.count("C")) / len(seq) for seq in bundle.test_sequences])
        metrics = fit_gc_baseline(train_gc, bundle.train_labels, test_gc, bundle.test_labels)

        test_frame = frame[frame["split"] == "test"]
        class_summary = (
            test_frame.groupby("label")["gc_fraction"]
            .agg(["mean", "std"])
            .rename(index={0: "negative", 1: "positive"})
        )
        class_summary.to_csv(RESULTS_ROOT / f"{dataset_name}_gc_by_label.csv")
        metric_rows.append(
            {
                "dataset": dataset_name,
                "model": "gc_fraction_only_logistic_regression",
                **metrics,
                "negative_gc_mean": float(class_summary.loc["negative", "mean"]),
                "positive_gc_mean": float(class_summary.loc["positive", "mean"]),
            }
        )

    combined_frame = pd.concat(all_frames, ignore_index=True)
    plot_gc_distributions(combined_frame, FIGURES_ROOT / "gc_fraction_by_class.png")
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(RESULTS_ROOT / "gc_baseline_metrics.csv", index=False)
    print(metrics_frame.to_string(index=False))


if __name__ == "__main__":
    main()
