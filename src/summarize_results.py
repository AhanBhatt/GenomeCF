from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
FIGURES_ROOT = PROJECT_ROOT / "figures"


def load_results() -> pd.DataFrame:
    kmer_model = pd.read_csv(RESULTS_ROOT / "kmer_baseline" / "model_metrics.csv")
    kmer_cf = pd.read_csv(RESULTS_ROOT / "kmer_baseline" / "counterfactual_metrics.csv")
    cnn_model = pd.read_csv(RESULTS_ROOT / "cnn_baseline" / "model_metrics.csv")
    cnn_cf = pd.read_csv(RESULTS_ROOT / "cnn_baseline" / "counterfactual_metrics.csv")

    kmer = kmer_model.merge(kmer_cf, on="dataset")
    cnn = cnn_model.merge(cnn_cf, on="dataset")
    return pd.concat([kmer, cnn], ignore_index=True)


def plot_metric_grid(results: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("test_auroc", "Test AUROC"),
        ("test_accuracy", "Test Accuracy"),
        ("reverse_complement_mean_abs_delta", "RC Mean Abs Delta"),
        ("positive_gc_shuffle_prob_drop", "Positive Shuffle Drop"),
    ]
    datasets = list(results["dataset"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7))
    axes = axes.ravel()

    color_map = {
        "kmer_logistic_regression": "#204b87",
        "small_sequence_cnn": "#4c9f70",
    }
    label_map = {
        "kmer_logistic_regression": "6-mer logistic regression",
        "small_sequence_cnn": "Small CNN",
    }

    for ax, (metric, title) in zip(axes, metrics):
        width = 0.35
        x = range(len(datasets))
        for offset, model_name in [(-width / 2, "kmer_logistic_regression"), (width / 2, "small_sequence_cnn")]:
            subset = results[results["model"] == model_name].set_index("dataset").loc[datasets]
            ax.bar(
                [value + offset for value in x],
                subset[metric].values,
                width=width,
                color=color_map[model_name],
                label=label_map[model_name],
            )
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(datasets, rotation=15)
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(results: pd.DataFrame, output_path: Path) -> None:
    gc_path = RESULTS_ROOT / "gc_shortcut_check" / "gc_baseline_metrics.csv"
    gc_results = pd.read_csv(gc_path) if gc_path.exists() else None
    lines = [
        "# Experimental Summary",
        "",
        "## Main findings",
        "",
    ]

    for dataset_name in results["dataset"].unique():
        subset = results[results["dataset"] == dataset_name].set_index("model")
        kmer = subset.loc["kmer_logistic_regression"]
        cnn = subset.loc["small_sequence_cnn"]
        lines.extend(
            [
                f"### {dataset_name}",
                f"- AUROC: k-mer logistic regression {kmer['test_auroc']:.3f}, small CNN {cnn['test_auroc']:.3f}.",
                f"- Reverse-complement mean absolute probability change: k-mer {kmer['reverse_complement_mean_abs_delta']:.3f}, CNN {cnn['reverse_complement_mean_abs_delta']:.3f}.",
                f"- Positive-sequence GC-shuffle probability drop: k-mer {kmer['positive_gc_shuffle_prob_drop']:.3f}, CNN {cnn['positive_gc_shuffle_prob_drop']:.3f}.",
                "",
            ]
        )

    lines.extend(
        [
            "## Shortcut check",
            "",
        ]
    )

    if gc_results is not None:
        for _, row in gc_results.iterrows():
            lines.append(
                f"- {row['dataset']}: a GC-fraction-only logistic regression reaches AUROC {row['test_auroc']:.3f} "
                f"with negative vs positive mean GC fractions of {row['negative_gc_mean']:.3f} and {row['positive_gc_mean']:.3f}."
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- The promoter dataset is easier than the enhancer dataset for both models.",
            "- The k-mer model reaches the highest raw AUROC on both datasets, especially on promoters.",
            "- The CNN is substantially more stable under reverse-complement perturbations, especially on promoters.",
            "- Both models show shortcut sensitivity on GC-matched shuffled promoters, which suggests they are still using composition-level cues rather than only biologically meaningful motifs.",
            "- On enhancers, the CNN shows better counterfactual behavior than the k-mer model even though its AUROC is slightly lower.",
            "- The GC-only baseline shows that simple composition already carries substantial predictive signal, which helps explain why shuffled sequences can still receive strong positive scores.",
            "",
            "## Files",
            "",
            f"- Combined metrics: `{(RESULTS_ROOT / 'combined_model_comparison.csv').resolve()}`",
            f"- Comparison figure: `{(FIGURES_ROOT / 'model_comparison_grid.png').resolve()}`",
            f"- GC shortcut metrics: `{(RESULTS_ROOT / 'gc_shortcut_check' / 'gc_baseline_metrics.csv').resolve()}`",
            f"- GC distribution figure: `{(FIGURES_ROOT / 'gc_fraction_by_class.png').resolve()}`",
        ]
    )

    output_path.write_text("\n".join(lines))


def main() -> None:
    results = load_results()
    results.to_csv(RESULTS_ROOT / "combined_model_comparison.csv", index=False)
    plot_metric_grid(results, FIGURES_ROOT / "model_comparison_grid.png")
    write_summary(results, RESULTS_ROOT / "summary.md")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
