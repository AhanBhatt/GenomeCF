from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from genomecf import (
    build_kmer_logistic_regression,
    create_counterfactuals,
    evaluate_prediction_set,
    predict_cnn_probabilities,
    reverse_complement,
    save_json,
    train_cnn_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "genomecf_synthetic"
FIGURES_ROOT = PROJECT_ROOT / "figures"
SEEDS = [11, 23, 37, 49, 61]
MOTIF = "CACGTG"
ALPHABET = ["A", "C", "G", "T"]


def sample_background_sequence(length: int, gc_fraction: float, rng: random.Random) -> str:
    seq = []
    for _ in range(length):
        if rng.random() < gc_fraction:
            seq.append(rng.choice(["G", "C"]))
        else:
            seq.append(rng.choice(["A", "T"]))
    return "".join(seq)


def plant_motif(sequence: str, motif: str, position: int) -> str:
    return sequence[:position] + motif + sequence[position + len(motif) :]


def destroy_motif(sequence: str, motif_start: int) -> str:
    if motif_start < 0:
        return sequence
    motif = list(sequence[motif_start : motif_start + len(MOTIF)])
    motif[2] = "A" if motif[2] != "A" else "T"
    return sequence[:motif_start] + "".join(motif) + sequence[motif_start + len(MOTIF) :]


def sample_negative_without_motif(length: int, gc_fraction: float, rng: random.Random) -> str:
    while True:
        seq = sample_background_sequence(length, gc_fraction, rng)
        if MOTIF not in seq:
            return seq


def generate_synthetic_dataset(
    dataset_name: str,
    train_per_class: int,
    test_per_class: int,
    sequence_length: int,
    positive_gc: float,
    negative_gc: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    rng = random.Random(seed)
    rows = []
    for split_name, n_per_class in [("train", train_per_class), ("test", test_per_class)]:
        for label in [0, 1]:
            gc_fraction = positive_gc if label == 1 else negative_gc
            for idx in range(n_per_class):
                if label == 1:
                    background = sample_background_sequence(sequence_length, gc_fraction, rng)
                    start = rng.randint(15, sequence_length - len(MOTIF) - 15)
                    sequence = plant_motif(background, MOTIF, start)
                else:
                    start = -1
                    sequence = sample_negative_without_motif(sequence_length, gc_fraction, rng)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "split": split_name,
                        "label": label,
                        "sequence": sequence,
                        "motif_start": start,
                        "gc_fraction": (sequence.count("G") + sequence.count("C")) / len(sequence),
                    }
                )
    frame = pd.DataFrame(rows)
    return {
        "train": frame[frame["split"] == "train"].reset_index(drop=True),
        "test": frame[frame["split"] == "test"].reset_index(drop=True),
    }


def create_synthetic_counterfactuals(test_frame: pd.DataFrame, seed: int) -> dict[str, list[str]]:
    standard = create_counterfactuals(test_frame["sequence"].tolist(), seed=seed)
    motif_disrupted = [
        destroy_motif(seq, start) for seq, start in zip(test_frame["sequence"], test_frame["motif_start"])
    ]
    standard["motif_disruption"] = motif_disrupted
    return standard


def run_kmer_condition(train_frame: pd.DataFrame, test_frame: pd.DataFrame, seed: int) -> dict[str, float]:
    model = build_kmer_logistic_regression(k=6, seed=seed)
    model.fit(train_frame["sequence"].tolist(), train_frame["label"].tolist())
    original_probs = model.predict_proba(test_frame["sequence"].tolist())[:, 1]
    counterfactuals = create_synthetic_counterfactuals(test_frame, seed=seed)
    cf_probs = {name: model.predict_proba(seqs)[:, 1] for name, seqs in counterfactuals.items()}
    return {
        **evaluate_prediction_set(test_frame["label"].to_numpy(), original_probs, cf_probs, seed=seed),
        "model_family": "kmer_logistic_regression",
    }


def run_cnn_condition(train_frame: pd.DataFrame, test_frame: pd.DataFrame, seed: int, rc_augment: bool) -> dict[str, float]:
    model, history = train_cnn_model(
        train_frame["sequence"].tolist(),
        train_frame["label"].tolist(),
        seed=seed,
        rc_augment=rc_augment,
    )
    original_probs = predict_cnn_probabilities(model, test_frame["sequence"].tolist())
    counterfactuals = create_synthetic_counterfactuals(test_frame, seed=seed)
    cf_probs = {name: predict_cnn_probabilities(model, seqs) for name, seqs in counterfactuals.items()}
    metrics = evaluate_prediction_set(test_frame["label"].to_numpy(), original_probs, cf_probs, seed=seed)
    metrics["model_family"] = "small_cnn_rc_aug" if rc_augment else "small_cnn"
    metrics["final_val_auroc"] = history[-1]["val_auroc"]
    return metrics


def plot_synthetic_summary(summary_frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    conditions = ["gc_correlated", "gc_matched"]
    model_order = ["kmer_logistic_regression", "small_cnn", "small_cnn_rc_aug"]
    pretty = {
        "kmer_logistic_regression": "k-mer",
        "small_cnn": "CNN",
        "small_cnn_rc_aug": "RC-aug CNN",
    }
    colors = {
        "kmer_logistic_regression": "#204b87",
        "small_cnn": "#4c9f70",
        "small_cnn_rc_aug": "#d4842d",
    }
    metrics = [
        ("auroc_mean", "AUROC"),
        ("mono_shuffle_positive_prob_drop_mean", "Mono-shuffle drop"),
        ("motif_disruption_positive_prob_drop_mean", "Motif disruption drop"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        width = 0.22
        x = np.arange(len(conditions))
        for offset_idx, model_name in enumerate(model_order):
            subset = summary_frame[summary_frame["model_family"] == model_name].set_index("dataset").loc[conditions]
            ax.bar(
                x + (offset_idx - 1) * width,
                subset[metric].values,
                width=width,
                color=colors[model_name],
                label=pretty[model_name] if title == "AUROC" else None,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(["GC correlated", "GC matched"])
        ax.set_title(title)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    conditions = {
        "gc_correlated": {"positive_gc": 0.7, "negative_gc": 0.3},
        "gc_matched": {"positive_gc": 0.5, "negative_gc": 0.5},
    }

    per_run_rows: list[dict] = []
    summary_rows: list[dict] = []

    for dataset_name, gc_config in conditions.items():
        dataset = generate_synthetic_dataset(
            dataset_name=dataset_name,
            train_per_class=3000,
            test_per_class=1000,
            sequence_length=200,
            positive_gc=gc_config["positive_gc"],
            negative_gc=gc_config["negative_gc"],
            seed=2026,
        )

        kmer_metrics = run_kmer_condition(dataset["train"], dataset["test"], seed=42)
        per_run_rows.append({"dataset": dataset_name, "seed": 42, **kmer_metrics})

        for seed in SEEDS:
            for rc_augment in [False, True]:
                metrics = run_cnn_condition(dataset["train"], dataset["test"], seed=seed, rc_augment=rc_augment)
                per_run_rows.append({"dataset": dataset_name, "seed": seed, **metrics})
                print("synthetic", dataset_name, metrics["model_family"], seed, metrics["auroc"])

    per_run_frame = pd.DataFrame(per_run_rows)
    per_run_frame.to_csv(RESULTS_ROOT / "per_run_metrics.csv", index=False)

    for (dataset_name, model_family), group in per_run_frame.groupby(["dataset", "model_family"]):
        summary = {
            "dataset": dataset_name,
            "model_family": model_family,
            "n_runs": len(group),
        }
        metric_columns = [col for col in group.columns if col not in {"dataset", "model_family", "seed"}]
        for metric in metric_columns:
            summary[f"{metric}_mean"] = float(group[metric].mean())
            summary[f"{metric}_std"] = float(group[metric].std(ddof=0))
        summary_rows.append(summary)

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(RESULTS_ROOT / "summary_metrics.csv", index=False)
    plot_synthetic_summary(summary_frame, FIGURES_ROOT / "genomecf_synthetic_summary.png")

    save_json(
        {
            "motif": MOTIF,
            "conditions": conditions,
            "seeds": SEEDS,
        },
        RESULTS_ROOT / "run_config.json",
    )
    print(summary_frame[["dataset", "model_family", "auroc_mean", "mono_shuffle_positive_prob_drop_mean", "motif_disruption_positive_prob_drop_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
