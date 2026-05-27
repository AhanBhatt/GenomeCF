from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_RESULTS = PROJECT_ROOT / "results" / "genomecf_real" / "summary_metrics.csv"
SYNTH_RESULTS = PROJECT_ROOT / "results" / "genomecf_synthetic" / "summary_metrics.csv"
HOLDOUT_RESULTS = PROJECT_ROOT / "results" / "genomecf_holdout_cnn"
OUTPUT_MD = PROJECT_ROOT / "results" / "genomecf_expanded_summary.md"
FIGURES_ROOT = PROJECT_ROOT / "figures"


def load_holdout_ensembles() -> pd.DataFrame:
    rows = []
    for path in sorted(HOLDOUT_RESULTS.glob("*__ensemble.json")):
        rows.append(json.loads(path.read_text()))
    return pd.DataFrame(rows)


def plot_holdout_comparison(frame: pd.DataFrame, output_path: Path) -> None:
    target_tasks = ["human_nontata_promoters", "human_enhancers_cohn"]
    model_order = ["kmer_logistic_regression", "small_cnn", "small_cnn_rc_aug"]
    pretty_model = {
        "kmer_logistic_regression": "k-mer",
        "small_cnn": "CNN",
        "small_cnn_rc_aug": "RC-aug CNN",
    }
    colors = {
        "official": "#204b87",
        "chromosome_holdout": "#d4842d",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, dataset_name in zip(axes, target_tasks):
        subset = frame[frame["dataset"] == dataset_name]
        x = range(len(model_order))
        width = 0.34
        for offset, split_name in [(-width / 2, "official"), (width / 2, "chromosome_holdout")]:
            vals = []
            for model_name in model_order:
                match = subset[(subset["model_family"] == model_name) & (subset["split_scheme"] == split_name)]
                vals.append(match["auroc"].iloc[0] if not match.empty else float("nan"))
            ax.bar([i + offset for i in x], vals, width=width, color=colors[split_name], label=split_name if dataset_name == target_tasks[0] else None)
        ax.set_xticks(list(x))
        ax.set_xticklabels([pretty_model[m] for m in model_order], rotation=15)
        ax.set_title(dataset_name)
    axes[0].set_ylabel("AUROC")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    real = pd.read_csv(REAL_RESULTS)
    synthetic = pd.read_csv(SYNTH_RESULTS)
    holdout = load_holdout_ensembles()

    kmer_rows = real[real["model_family"] == "kmer_logistic_regression"][
        ["dataset", "split_scheme", "model_family", "ensemble_auroc", "ensemble_reverse_complement_mean_abs_delta", "ensemble_mono_shuffle_positive_prob_drop", "ensemble_dinuc_shuffle_positive_prob_drop", "ensemble_ece"]
    ].rename(
        columns={
            "ensemble_auroc": "auroc",
            "ensemble_reverse_complement_mean_abs_delta": "reverse_complement_mean_abs_delta",
            "ensemble_mono_shuffle_positive_prob_drop": "mono_shuffle_positive_prob_drop",
            "ensemble_dinuc_shuffle_positive_prob_drop": "dinuc_shuffle_positive_prob_drop",
            "ensemble_ece": "ece",
        }
    )
    cnn_rows = holdout[["dataset", "split_scheme", "model_family", "auroc", "reverse_complement_mean_abs_delta", "mono_shuffle_positive_prob_drop", "dinuc_shuffle_positive_prob_drop", "ece"]]
    official_cnn = real[
        real["model_family"].isin(["small_cnn", "small_cnn_rc_aug"])
    ][["dataset", "split_scheme", "model_family", "ensemble_auroc", "ensemble_reverse_complement_mean_abs_delta", "ensemble_mono_shuffle_positive_prob_drop", "ensemble_dinuc_shuffle_positive_prob_drop", "ensemble_ece"]].rename(
        columns={
            "ensemble_auroc": "auroc",
            "ensemble_reverse_complement_mean_abs_delta": "reverse_complement_mean_abs_delta",
            "ensemble_mono_shuffle_positive_prob_drop": "mono_shuffle_positive_prob_drop",
            "ensemble_dinuc_shuffle_positive_prob_drop": "dinuc_shuffle_positive_prob_drop",
            "ensemble_ece": "ece",
        }
    )
    split_comparison = pd.concat([kmer_rows, official_cnn, cnn_rows], ignore_index=True)
    split_comparison.to_csv(PROJECT_ROOT / "results" / "genomecf_split_comparison.csv", index=False)

    plot_holdout_comparison(split_comparison, FIGURES_ROOT / "genomecf_holdout_comparison.png")

    best_official = real[real["split_scheme"] == "official"].sort_values("ensemble_auroc", ascending=False)
    highest_instability = real.sort_values("ensemble_reverse_complement_mean_abs_delta", ascending=False)

    lines = [
        "# GenomeCF Expanded Summary",
        "",
        "## Scope",
        "",
        "- Real benchmark tasks: 6 binary genomic sequence tasks.",
        "- Human chromosome-held-out evaluation: 4 tasks for k-mer logistic regression, plus learned-model holdout runs on the 2 core human tasks.",
        "- Learned models with 5 random seeds: small CNN and reverse-complement-augmented CNN.",
        "- Pretrained model baseline: DNABERT-2 frozen embeddings with logistic regression on the 2 core tasks.",
        "- Perturbations: reverse complement, mononucleotide-preserving shuffle, dinucleotide-preserving shuffle.",
        "- Synthetic benchmark: planted motif tasks with GC-correlated and GC-matched conditions.",
        "",
        "## Key real-data findings",
        "",
        f"- The strongest official-split AUROC in the expanded suite is {best_official.iloc[0]['ensemble_auroc']:.3f} from {best_official.iloc[0]['model_family']} on {best_official.iloc[0]['dataset']}.",
        f"- The largest reverse-complement instability among the real tasks is {highest_instability.iloc[0]['ensemble_reverse_complement_mean_abs_delta']:.3f} from {highest_instability.iloc[0]['model_family']} on {highest_instability.iloc[0]['dataset']} ({highest_instability.iloc[0]['split_scheme']}).",
        "- Reverse-complement augmentation consistently helps on several human tasks. On `human_enhancers_ensembl`, ensemble AUROC rises from 0.839 to 0.884 and mono-shuffle drop rises from 0.178 to 0.320.",
        "- DNABERT-2 embeddings give the best AUROC on `human_enhancers_cohn` at 0.830, but still show negative mononucleotide-shuffle drop (-0.049), which means high accuracy does not prevent shortcut-like behavior.",
        "- On promoters, the k-mer model remains strongest by AUROC (0.898 official) but has much larger reverse-complement instability (0.263) than either CNN ensemble (0.044) or DNABERT-2 (0.078).",
        "",
        "## Split findings",
        "",
        "- On `human_nontata_promoters`, k-mer AUROC drops from 0.898 on the official split to 0.840 on the chromosome-held-out split.",
        "- On the same promoter task, the chromosome-held-out CNN ensemble reaches 0.877 AUROC, which nearly closes the gap to the official k-mer result while keeping reverse-complement instability low at 0.038.",
        "- On `human_enhancers_cohn`, reverse-complement-augmented CNN improves chromosome-held-out AUROC from 0.671 to 0.687 relative to the non-augmented CNN.",
        "",
        "## Synthetic findings",
        "",
        "- In the GC-correlated synthetic condition, every model reaches AUROC about 1.0 while motif-disruption and shuffle drops stay near zero. This means the models can solve the task without relying on the planted motif.",
        "- In the GC-matched synthetic condition, AUROC stays about 1.0 but motif-disruption and shuffle drops jump to about 0.87 to 0.95, showing that once the shortcut is removed, the models track the true motif rule.",
        "- This synthetic result supports the interpretation of the real-data benchmark: high AUROC alone cannot tell whether the model learned the intended biological mechanism.",
        "",
        "## Files",
        "",
        f"- Real benchmark summary: `{REAL_RESULTS.resolve()}`",
        f"- Synthetic benchmark summary: `{SYNTH_RESULTS.resolve()}`",
        f"- Split comparison table: `{(PROJECT_ROOT / 'results' / 'genomecf_split_comparison.csv').resolve()}`",
        f"- Holdout comparison figure: `{(FIGURES_ROOT / 'genomecf_holdout_comparison.png').resolve()}`",
        f"- Official benchmark overview figure: `{(FIGURES_ROOT / 'genomecf_overview.png').resolve()}`",
        f"- Synthetic benchmark figure: `{(FIGURES_ROOT / 'genomecf_synthetic_summary.png').resolve()}`",
    ]
    OUTPUT_MD.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
