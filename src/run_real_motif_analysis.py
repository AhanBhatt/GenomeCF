from __future__ import annotations

from pathlib import Path
import random
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from genomecf.config import get_model_spec, get_split_spec, get_task_spec
from genomecf.data import build_split_frames, load_task_frame
from genomecf.metrics import bootstrap_ci, standard_metrics
from genomecf.models import build_runner
from genomecf.perturbations import (
    disrupt_first_motif,
    disrupt_first_motif_gc_preserving,
    find_exact_motif_spans,
    random_non_motif_edit,
)


RESULTS_ROOT = PROJECT_ROOT / "results" / "release"
SEED = 2026
MAX_PROBES = 1000
TASKS = [
    "human_nontata_promoters",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
]
MODELS = [
    ("kmer_logistic_regression", "classical"),
    ("small_cnn", "supervised"),
    ("small_cnn_rc_aug", "supervised"),
    ("dnabert2", "frozen"),
    ("caduceus_ph", "frozen"),
]
TASK_LABELS = {
    "human_nontata_promoters": "Promoters",
    "human_enhancers_cohn": "Enhancers (Cohn)",
    "human_enhancers_ensembl": "Enhancers (Ensembl)",
}
MODEL_LABELS = {
    "kmer_logistic_regression": "6-mer logistic regression",
    "small_cnn": "CNN",
    "small_cnn_rc_aug": "RC-aug CNN",
    "dnabert2": "DNABERT-2",
    "caduceus_ph": "Caduceus-Ph",
}
MOTIF_MAP = {
    "human_nontata_promoters": ["TATAAA", "CCAAT", "GGGCGG"],
    "human_enhancers_cohn": ["CACGTG", "TGACTCA", "GGAA"],
    "human_enhancers_ensembl": ["CACGTG", "TGACTCA", "GGAA"],
}


def _sample_probe_rows(test_frame: pd.DataFrame, motifs: list[str]) -> pd.DataFrame:
    positives = test_frame[test_frame["label"] == 1].copy()
    positives["motif_spans"] = positives["sequence"].map(lambda seq: find_exact_motif_spans(seq, motifs))
    positives["has_motif"] = positives["motif_spans"].map(bool)
    hits = positives[positives["has_motif"]].copy()
    if len(hits) > MAX_PROBES:
        hits = hits.sample(n=MAX_PROBES, random_state=SEED).reset_index(drop=True)
    else:
        hits = hits.reset_index(drop=True)
    return positives, hits


def _probability_drop_summary(values: pd.Series) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    mean_value = float(array.mean())
    low, high = bootstrap_ci(lambda idx: float(array[idx].mean()), len(array), seed=SEED)
    return mean_value, float(low), float(high)


def run_real_motif_suite() -> tuple[pd.DataFrame, pd.DataFrame]:
    split_spec = get_split_spec("official")
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for task_name in TASKS:
        task_spec = get_task_spec(task_name)
        frame = load_task_frame(task_spec, PROJECT_ROOT)
        splits = build_split_frames(frame, split_spec, seed=SEED)
        test_frame = splits["test"].copy().reset_index(drop=True)
        motifs = MOTIF_MAP[task_name]
        positives, probes = _sample_probe_rows(test_frame, motifs)
        if probes.empty:
            continue
        for model_name, mode in MODELS:
            runner = build_runner(get_model_spec(model_name), mode=mode, seed=SEED)
            runner.fit(splits["train"], splits["validation"])
            original_probs = runner.predict_proba(probes)
            motif_disrupted = []
            gc_preserving = []
            random_control = []
            for idx, sequence in enumerate(probes["sequence"].tolist()):
                rng = random.Random(SEED + idx)
                motif_disrupted.append(disrupt_first_motif(sequence, motifs, rng))
                gc_preserving.append(disrupt_first_motif_gc_preserving(sequence, motifs, rng))
                random_control.append(random_non_motif_edit(sequence, motifs, rng, preserve_gc_class=True))
            motif_probs = runner.predict_proba(probes.assign(sequence=motif_disrupted))
            gc_preserving_probs = runner.predict_proba(probes.assign(sequence=gc_preserving))
            random_probs = runner.predict_proba(probes.assign(sequence=random_control))
            official_metrics = standard_metrics(test_frame["label"].to_numpy(), runner.predict_proba(test_frame))
            probe_frame = pd.DataFrame(
                {
                    "task_id": task_name,
                    "task_label": TASK_LABELS[task_name],
                    "model_id": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "sequence_id": probes.get("id", probes.index).tolist(),
                    "original_prob": original_probs,
                    "motif_disrupted_prob": motif_probs,
                    "gc_preserving_prob": gc_preserving_probs,
                    "random_edit_prob": random_probs,
                }
            )
            probe_frame["motif_drop"] = probe_frame["original_prob"] - probe_frame["motif_disrupted_prob"]
            probe_frame["gc_preserving_motif_drop"] = probe_frame["original_prob"] - probe_frame["gc_preserving_prob"]
            probe_frame["random_edit_drop"] = probe_frame["original_prob"] - probe_frame["random_edit_prob"]
            probe_frame["motif_minus_random"] = probe_frame["motif_drop"] - probe_frame["random_edit_drop"]
            detail_rows.extend(probe_frame.to_dict(orient="records"))

            motif_mean, motif_low, motif_high = _probability_drop_summary(probe_frame["motif_drop"])
            gc_mean, gc_low, gc_high = _probability_drop_summary(probe_frame["gc_preserving_motif_drop"])
            rand_mean, rand_low, rand_high = _probability_drop_summary(probe_frame["random_edit_drop"])
            diff_mean, diff_low, diff_high = _probability_drop_summary(probe_frame["motif_minus_random"])
            summary_rows.append(
                {
                    "task_id": task_name,
                    "task_label": TASK_LABELS[task_name],
                    "model_id": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "scanned_positive_count": int(len(positives)),
                    "motif_hit_count": int(len(positives[positives["has_motif"]])),
                    "evaluated_count": int(len(probe_frame)),
                    "motifs_tested": ";".join(motifs),
                    "official_auroc": float(official_metrics["auroc"]),
                    "mean_original_prob": float(probe_frame["original_prob"].mean()),
                    "mean_motif_disrupted_prob": float(probe_frame["motif_disrupted_prob"].mean()),
                    "mean_gc_preserving_prob": float(probe_frame["gc_preserving_prob"].mean()),
                    "mean_random_edit_prob": float(probe_frame["random_edit_prob"].mean()),
                    "motif_drop": motif_mean,
                    "motif_drop_ci_low": motif_low,
                    "motif_drop_ci_high": motif_high,
                    "gc_preserving_motif_drop": gc_mean,
                    "gc_preserving_ci_low": gc_low,
                    "gc_preserving_ci_high": gc_high,
                    "random_edit_drop": rand_mean,
                    "random_edit_ci_low": rand_low,
                    "random_edit_ci_high": rand_high,
                    "motif_minus_random": diff_mean,
                    "motif_minus_random_ci_low": diff_low,
                    "motif_minus_random_ci_high": diff_high,
                }
            )
            print(
                "motif",
                task_name,
                model_name,
                f"hits={len(probe_frame)}",
                f"motif_drop={motif_mean:.4f}",
                f"random_drop={rand_mean:.4f}",
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    summary, details = run_real_motif_suite()
    summary.to_csv(RESULTS_ROOT / "real_motif_probe_summary.csv", index=False)
    details.to_csv(RESULTS_ROOT / "real_motif_probe_details.csv", index=False)
    print(summary[["task_id", "model_id", "motif_drop", "random_edit_drop", "motif_minus_random"]].to_string(index=False))


if __name__ == "__main__":
    main()
