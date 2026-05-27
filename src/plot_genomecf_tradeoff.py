from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_RESULTS = PROJECT_ROOT / "results" / "genomecf_real" / "summary_metrics.csv"
FIGURES_ROOT = PROJECT_ROOT / "figures"


def main() -> None:
    frame = pd.read_csv(REAL_RESULTS)
    frame = frame[frame["split_scheme"] == "official"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    color_map = {
        "kmer_logistic_regression": "#204b87",
        "small_cnn": "#4c9f70",
        "small_cnn_rc_aug": "#d4842d",
        "dnabert2_embedding_logistic": "#8a3fb0",
    }
    label_map = {
        "kmer_logistic_regression": "k-mer",
        "small_cnn": "CNN",
        "small_cnn_rc_aug": "RC-aug CNN",
        "dnabert2_embedding_logistic": "DNABERT-2",
    }

    for model_family, group in frame.groupby("model_family"):
        axes[0].scatter(
            group["ensemble_auroc"],
            group["ensemble_reverse_complement_mean_abs_delta"],
            s=60,
            color=color_map[model_family],
            label=label_map[model_family],
        )
        axes[1].scatter(
            group["ensemble_auroc"],
            group["ensemble_mono_shuffle_positive_prob_drop"],
            s=60,
            color=color_map[model_family],
            label=label_map[model_family],
        )
        for _, row in group.iterrows():
            short = row["dataset"].replace("human_", "").replace("_ensembl", "").replace("_cohn", "").replace("_stark", "").replace("_nontata", "")
            axes[0].annotate(short, (row["ensemble_auroc"], row["ensemble_reverse_complement_mean_abs_delta"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
            axes[1].annotate(short, (row["ensemble_auroc"], row["ensemble_mono_shuffle_positive_prob_drop"]), fontsize=7, xytext=(3, 3), textcoords="offset points")

    axes[0].set_xlabel("AUROC")
    axes[0].set_ylabel("RC mean abs delta")
    axes[0].set_title("Accuracy vs reverse-complement stability")
    axes[1].set_xlabel("AUROC")
    axes[1].set_ylabel("Mono-shuffle positive drop")
    axes[1].set_title("Accuracy vs shuffle sensitivity")
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    path = FIGURES_ROOT / "genomecf_tradeoff.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
