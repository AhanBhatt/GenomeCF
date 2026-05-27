from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_RESULTS = PROJECT_ROOT / "results" / "genomecf_real" / "summary_metrics.csv"
FIGURES_ROOT = PROJECT_ROOT / "figures"


def main() -> None:
    frame = pd.read_csv(REAL_RESULTS)
    subset = frame[
        (frame["split_scheme"] == "official")
        & (frame["dataset"].isin(["human_nontata_promoters", "human_enhancers_cohn"]))
        & (frame["model_family"].isin(["kmer_logistic_regression", "small_cnn_rc_aug", "dnabert2_embedding_logistic"]))
    ].copy()

    model_order = ["kmer_logistic_regression", "small_cnn_rc_aug", "dnabert2_embedding_logistic"]
    pretty_model = {
        "kmer_logistic_regression": "k-mer",
        "small_cnn_rc_aug": "RC-aug CNN",
        "dnabert2_embedding_logistic": "DNABERT-2",
    }
    colors = {"original": "#204b87", "mono_shuffle": "#d4842d"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
    for ax, dataset_name in zip(axes, ["human_nontata_promoters", "human_enhancers_cohn"]):
        ds = subset[subset["dataset"] == dataset_name].set_index("model_family")
        x = range(len(model_order))
        width = 0.34
        original = [ds.loc[m, "ensemble_ece"] for m in model_order]
        shuffle = [ds.loc[m, "ensemble_mono_shuffle_ece"] for m in model_order]
        ax.bar([i - width / 2 for i in x], original, width=width, color=colors["original"], label="Original")
        ax.bar([i + width / 2 for i in x], shuffle, width=width, color=colors["mono_shuffle"], label="Mono-shuffle")
        ax.set_xticks(list(x))
        ax.set_xticklabels([pretty_model[m] for m in model_order], rotation=15)
        ax.set_title(dataset_name)
    axes[0].set_ylabel("Expected calibration error")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    path = FIGURES_ROOT / "genomecf_calibration.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
