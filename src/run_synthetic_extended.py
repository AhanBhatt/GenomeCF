from __future__ import annotations

from pathlib import Path
import random
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from genomecf.data import annotate_sequence_confounders
from genomecf.metrics import counterfactual_metrics, standard_metrics
from genomecf.models import build_runner
from genomecf.perturbations import disrupt_first_motif, klet_preserving_shuffle, reverse_complement
from genomecf.config import get_model_spec


RESULTS_ROOT = PROJECT_ROOT / "results" / "release"
SEED = 2026
SEQ_LEN = 200
MOTIF_A = "CACGTG"
MOTIF_B = "GGAA"
TASKS = ["gc_correlated", "gc_matched", "gc_conflict", "two_motif_grammar", "motif_position_conflict"]
MODELS = [
    ("kmer_logistic_regression", "classical"),
    ("small_cnn", "supervised"),
    ("small_cnn_rc_aug", "supervised"),
    ("dnabert2", "frozen"),
    ("caduceus_ph", "frozen"),
]
MODEL_LABELS = {
    "kmer_logistic_regression": "6-mer logistic regression",
    "small_cnn": "CNN",
    "small_cnn_rc_aug": "RC-aug CNN",
    "dnabert2": "DNABERT-2",
    "caduceus_ph": "Caduceus-Ph",
}
TASK_LABELS = {
    "gc_correlated": "GC correlated",
    "gc_matched": "GC matched",
    "gc_conflict": "GC shortcut conflict",
    "two_motif_grammar": "Two-motif grammar",
    "motif_position_conflict": "Motif identity vs position shortcut",
}


def sample_background(length: int, gc_fraction: float, rng: random.Random) -> str:
    return "".join(rng.choice(["G", "C"]) if rng.random() < gc_fraction else rng.choice(["A", "T"]) for _ in range(length))


def plant(sequence: str, motif: str, start: int) -> str:
    return sequence[:start] + motif + sequence[start + len(motif) :]


def sample_without_motif(length: int, gc_fraction: float, motif: str, rng: random.Random) -> str:
    while True:
        seq = sample_background(length, gc_fraction, rng)
        if motif not in seq:
            return seq


def generate_gc_conflict() -> dict[str, pd.DataFrame]:
    rng = random.Random(SEED)
    rows: list[dict[str, object]] = []
    specs = {
        "train": [(1, 2500, 0.70), (0, 2500, 0.30)],
        "validation": [(1, 500, 0.70), (0, 500, 0.30)],
        "test": [(1, 1000, 0.30), (0, 1000, 0.70)],
    }
    for split, configs in specs.items():
        for label, count, gc_fraction in configs:
            for idx in range(count):
                if label == 1:
                    seq = plant(sample_background(SEQ_LEN, gc_fraction, rng), MOTIF_A, rng.randint(40, 140))
                else:
                    seq = sample_without_motif(SEQ_LEN, gc_fraction, MOTIF_A, rng)
                rows.append(
                    {
                        "id": f"gc_conflict_{split}_{label}_{idx}",
                        "orig_split": split,
                        "label": label,
                        "sequence": seq,
                        "task_id": "gc_conflict",
                        "shortcut_label": 1 if gc_fraction > 0.5 else 0,
                    }
                )
    frame = annotate_sequence_confounders(pd.DataFrame(rows))
    return {split: frame[frame["orig_split"] == split].reset_index(drop=True) for split in ["train", "validation", "test"]}


def generate_gc_aligned(task_id: str, positive_gc: float, negative_gc: float) -> dict[str, pd.DataFrame]:
    rng = random.Random(SEED + (10 if task_id == "gc_matched" else 9))
    rows: list[dict[str, object]] = []
    specs = {"train": 2500, "validation": 500, "test": 1000}
    for split, count in specs.items():
        for label in [0, 1]:
            gc_fraction = positive_gc if label == 1 else negative_gc
            for idx in range(count):
                if label == 1:
                    seq = plant(sample_background(SEQ_LEN, gc_fraction, rng), MOTIF_A, rng.randint(40, 140))
                else:
                    seq = sample_without_motif(SEQ_LEN, gc_fraction, MOTIF_A, rng)
                rows.append(
                    {
                        "id": f"{task_id}_{split}_{label}_{idx}",
                        "orig_split": split,
                        "label": label,
                        "sequence": seq,
                        "task_id": task_id,
                        "shortcut_label": 1 if gc_fraction > 0.5 else 0,
                    }
                )
    frame = annotate_sequence_confounders(pd.DataFrame(rows))
    return {split: frame[frame["orig_split"] == split].reset_index(drop=True) for split in ["train", "validation", "test"]}


def generate_two_motif_grammar() -> dict[str, pd.DataFrame]:
    rng = random.Random(SEED + 1)
    rows: list[dict[str, object]] = []
    specs = {"train": 2500, "validation": 500, "test": 1000}
    for split, count in specs.items():
        for label in [0, 1]:
            for idx in range(count):
                gc_fraction = 0.5
                background = sample_background(SEQ_LEN, gc_fraction, rng)
                if label == 1:
                    start_a = rng.randint(30, 60)
                    start_b = start_a + 18
                    seq = plant(plant(background, MOTIF_A, start_a), MOTIF_B, start_b)
                else:
                    start_a = rng.randint(30, 60)
                    if idx % 2 == 0:
                        start_b = start_a + 8
                        seq = plant(plant(background, MOTIF_A, start_a), MOTIF_B, start_b)
                    else:
                        seq = plant(background, MOTIF_A, start_a)
                rows.append(
                    {
                        "id": f"grammar_{split}_{label}_{idx}",
                        "orig_split": split,
                        "label": label,
                        "sequence": seq,
                        "task_id": "two_motif_grammar",
                        "shortcut_label": 1 if (MOTIF_A in seq and MOTIF_B in seq) else 0,
                    }
                )
    frame = annotate_sequence_confounders(pd.DataFrame(rows))
    return {split: frame[frame["orig_split"] == split].reset_index(drop=True) for split in ["train", "validation", "test"]}


def generate_position_conflict() -> dict[str, pd.DataFrame]:
    rng = random.Random(SEED + 2)
    rows: list[dict[str, object]] = []
    specs = {"train": 2500, "validation": 500, "test": 1000}
    for split, count in specs.items():
        for label in [0, 1]:
            for idx in range(count):
                background = sample_background(SEQ_LEN, 0.5, rng)
                if split == "test":
                    start_positive = 25
                    start_negative = 95
                else:
                    start_positive = 95
                    start_negative = 25
                if label == 1:
                    seq = plant(background, MOTIF_A, start_positive)
                    shortcut_label = 1 if start_positive == 95 else 0
                else:
                    seq = plant(background, MOTIF_B, start_negative)
                    shortcut_label = 1 if start_negative == 95 else 0
                rows.append(
                    {
                        "id": f"position_{split}_{label}_{idx}",
                        "orig_split": split,
                        "label": label,
                        "sequence": seq,
                        "task_id": "motif_position_conflict",
                        "shortcut_label": shortcut_label,
                    }
                )
    frame = annotate_sequence_confounders(pd.DataFrame(rows))
    return {split: frame[frame["orig_split"] == split].reset_index(drop=True) for split in ["train", "validation", "test"]}


def generate_task(task_id: str) -> dict[str, pd.DataFrame]:
    if task_id == "gc_correlated":
        return generate_gc_aligned("gc_correlated", positive_gc=0.70, negative_gc=0.30)
    if task_id == "gc_matched":
        return generate_gc_aligned("gc_matched", positive_gc=0.50, negative_gc=0.50)
    if task_id == "gc_conflict":
        return generate_gc_conflict()
    if task_id == "two_motif_grammar":
        return generate_two_motif_grammar()
    if task_id == "motif_position_conflict":
        return generate_position_conflict()
    raise ValueError(task_id)


def create_counterfactuals(test_frame: pd.DataFrame, motifs: list[str]) -> dict[str, list[str]]:
    rng = random.Random(SEED)
    return {
        "reverse_complement": [reverse_complement(seq) for seq in test_frame["sequence"]],
        "k1_shuffle": [klet_preserving_shuffle(seq, 1, rng) for seq in test_frame["sequence"]],
        "k2_shuffle": [klet_preserving_shuffle(seq, 2, rng) for seq in test_frame["sequence"]],
        "motif_disruption": [disrupt_first_motif(seq, motifs, random.Random(SEED + idx)) for idx, seq in enumerate(test_frame["sequence"])],
    }


def run_suite() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task_id in TASKS:
        splits = generate_task(task_id)
        motifs = [MOTIF_A, MOTIF_B]
        for model_id, mode in MODELS:
            runner = build_runner(get_model_spec(model_id), mode=mode, seed=SEED)
            runner.fit(splits["train"], splits["validation"])
            test = splits["test"].copy()
            labels = test["label"].to_numpy()
            original = runner.predict_proba(test)
            metrics = standard_metrics(labels, original)
            counterfactuals = create_counterfactuals(test, motifs)
            cf_metrics = {}
            for name, sequences in counterfactuals.items():
                probs = runner.predict_proba(test.assign(sequence=sequences))
                cf_metrics[name] = counterfactual_metrics(labels, original, probs)
            preds = (original >= 0.5).astype(int)
            shortcut_labels = test["shortcut_label"].to_numpy()
            rows.append(
                {
                    "task_id": task_id,
                    "task_label": TASK_LABELS[task_id],
                    "model_id": model_id,
                    "model_label": MODEL_LABELS[model_id],
                    "seed": SEED,
                    "auroc": float(metrics["auroc"]),
                    "ece": float(metrics["ece"]),
                    "brier": float(metrics["brier"]),
                    "rc_mean_abs_delta": float(cf_metrics["reverse_complement"]["mean_abs_delta"]),
                    "mono_positive_prob_drop": float(cf_metrics["k1_shuffle"]["positive_prob_drop"]),
                    "dinuc_positive_prob_drop": float(cf_metrics["k2_shuffle"]["positive_prob_drop"]),
                    "motif_positive_prob_drop": float(cf_metrics["motif_disruption"]["positive_prob_drop"]),
                    "shortcut_conflict_accuracy": float((preds == labels).mean()),
                    "rule_following_rate": float((preds == labels).mean()),
                    "shortcut_following_rate": float((preds == shortcut_labels).mean()),
                }
            )
            print(
                "synthetic_ext",
                task_id,
                model_id,
                f"auroc={metrics['auroc']:.3f}",
                f"rule={rows[-1]['rule_following_rate']:.3f}",
                f"shortcut={rows[-1]['shortcut_following_rate']:.3f}",
            )
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    results = run_suite()
    results.to_csv(RESULTS_ROOT / "synthetic_extended_summary.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
