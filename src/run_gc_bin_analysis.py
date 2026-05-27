from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from genomecf.calibration import fit_calibrator
from genomecf.config import get_model_spec, get_split_spec, get_task_spec
from genomecf.data import build_split_frames, load_task_frame
from genomecf.metrics import brier_score, expected_calibration_error, standard_metrics
from genomecf.models import DEVICE, build_runner

SEED = 2026
DEFAULT_TASKS = [
    "human_nontata_promoters",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ocr_ensembl",
]
DEFAULT_MODELS = [
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
    "human_ocr_ensembl": "Open chromatin",
    "gue_human_tf_0": "External TF binding (human_tf_0)",
    "gue_human_tf_1": "External TF binding (human_tf_1)",
    "gue_emp_h3k4me3": "External histone mark (H3K4me3)",
    "gue_emp_h3k14ac": "External histone mark (H3K14ac)",
}
MODEL_LABELS = {
    "kmer_logistic_regression": "6-mer logistic regression",
    "small_cnn": "CNN",
    "small_cnn_rc_aug": "RC-aug CNN",
    "dnabert2": "DNABERT-2",
    "caduceus_ph": "Caduceus-Ph",
}


def _gc_bin_ids(frame: pd.DataFrame, n_bins: int = 5) -> pd.Series:
    return pd.qcut(
        frame["gc_fraction"],
        q=min(n_bins, frame["gc_fraction"].nunique()),
        labels=False,
        duplicates="drop",
    ).fillna(0).astype(int)


def _safe_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(standard_metrics(labels, probs)["auroc"])


def _dedupe(frame: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.drop_duplicates(subset=key_columns, keep="last").sort_values(key_columns).reset_index(drop=True)


def _save_rows(frame: pd.DataFrame, path: Path, key_columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True, sort=False)
    frame = _dedupe(frame, key_columns)
    frame.to_csv(path, index=False)


def run_gc_bin_suite(
    *,
    task_names: list[str],
    models: list[tuple[str, str]],
    split_name: str,
    train_per_label: int | None,
    val_per_label: int | None,
    test_per_label: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_spec = get_split_spec(split_name)
    if train_per_label is not None:
        split_spec.train_per_label = train_per_label
    if val_per_label is not None:
        split_spec.val_per_label = val_per_label
    if test_per_label is not None:
        split_spec.test_per_label = test_per_label

    per_bin_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for task_name in task_names:
        task_spec = get_task_spec(task_name)
        frame = load_task_frame(task_spec, PROJECT_ROOT)
        splits = build_split_frames(frame, split_spec, seed=seed)
        test_frame = splits["test"].copy().reset_index(drop=True)
        test_frame["gc_bin"] = _gc_bin_ids(test_frame)
        for model_name, mode in models:
            runner = build_runner(get_model_spec(model_name), mode=mode, seed=seed)
            artifacts = runner.fit(splits["train"], splits["validation"])
            calibrator = fit_calibrator("none", artifacts.validation_probs, artifacts.validation_labels)
            probs = calibrator.transform(runner.predict_proba(test_frame))
            overall = standard_metrics(test_frame["label"].to_numpy(), probs)
            bin_rows: list[dict[str, object]] = []
            for gc_bin, bin_frame in test_frame.groupby("gc_bin", sort=True):
                indices = bin_frame.index.to_numpy()
                bin_probs = probs[indices]
                labels = bin_frame["label"].to_numpy()
                bin_row = {
                    "task_id": task_name,
                    "task_label": TASK_LABELS.get(task_name, task_name),
                    "model_id": model_name,
                    "model_label": MODEL_LABELS.get(model_name, model_name),
                    "seed": seed,
                    "split_id": split_name,
                    "gc_bin": int(gc_bin),
                    "count": int(len(bin_frame)),
                    "positive_count": int((bin_frame["label"] == 1).sum()),
                    "negative_count": int((bin_frame["label"] == 0).sum()),
                    "gc_mean": float(bin_frame["gc_fraction"].mean()),
                    "gc_min": float(bin_frame["gc_fraction"].min()),
                    "gc_max": float(bin_frame["gc_fraction"].max()),
                    "auroc": _safe_auroc(labels, bin_probs),
                    "ece": expected_calibration_error(labels, bin_probs),
                    "brier": brier_score(labels, bin_probs),
                }
                per_bin_rows.append(bin_row)
                bin_rows.append(bin_row)

            bin_frame = pd.DataFrame(bin_rows)
            valid_aurocs = bin_frame["auroc"].dropna()
            summary_rows.append(
                {
                    "task_id": task_name,
                    "task_label": TASK_LABELS.get(task_name, task_name),
                    "model_id": model_name,
                    "model_label": MODEL_LABELS.get(model_name, model_name),
                    "seed": seed,
                    "split_id": split_name,
                    "overall_auroc": float(overall["auroc"]),
                    "overall_ece": float(overall["ece"]),
                    "overall_brier": float(overall["brier"]),
                    "n_bins": int(bin_frame["gc_bin"].nunique()),
                    "worst_bin_auroc": float(valid_aurocs.min()) if not valid_aurocs.empty else float("nan"),
                    "best_bin_auroc": float(valid_aurocs.max()) if not valid_aurocs.empty else float("nan"),
                    "gc_bin_auroc_gap": float(valid_aurocs.max() - valid_aurocs.min()) if len(valid_aurocs) >= 2 else float("nan"),
                    "worst_bin_ece": float(bin_frame["ece"].max()),
                    "best_bin_ece": float(bin_frame["ece"].min()),
                    "gc_bin_ece_gap": float(bin_frame["ece"].max() - bin_frame["ece"].min()),
                    "runtime_seconds": float(artifacts.runtime_s),
                    "device": DEVICE.type,
                }
            )
            print(
                "gc-bin",
                split_name,
                task_name,
                model_name,
                f"overall={overall['auroc']:.3f}",
                f"worst_bin={summary_rows[-1]['worst_bin_auroc']:.3f}",
                f"gap={summary_rows[-1]['gc_bin_auroc_gap']:.3f}",
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(per_bin_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GC-bin robustness analysis for selected GenomeCF tasks and models.")
    parser.add_argument("--split", default="official")
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--models", nargs="*", default=[name for name, _ in DEFAULT_MODELS])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-per-label", type=int, default=None)
    parser.add_argument("--val-per-label", type=int, default=None)
    parser.add_argument("--test-per-label", type=int, default=None)
    parser.add_argument("--summary-output", type=Path, default=PROJECT_ROOT / "results" / "release" / "gc_bin_summary.csv")
    parser.add_argument("--by-bin-output", type=Path, default=PROJECT_ROOT / "results" / "release" / "gc_bin_by_bin.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode_lookup = dict(DEFAULT_MODELS)
    models = [(model_name, mode_lookup[model_name]) for model_name in args.models]
    summary, per_bin = run_gc_bin_suite(
        task_names=args.tasks,
        models=models,
        split_name=args.split,
        train_per_label=args.train_per_label,
        val_per_label=args.val_per_label,
        test_per_label=args.test_per_label,
        seed=args.seed,
    )
    _save_rows(summary, args.summary_output, ["task_id", "model_id", "split_id", "seed"])
    _save_rows(per_bin, args.by_bin_output, ["task_id", "model_id", "split_id", "seed", "gc_bin"])
    print(f"Wrote {len(summary)} GC-bin summary rows to {args.summary_output}")
    print(f"Wrote {len(per_bin)} GC-bin bin rows to {args.by_bin_output}")


if __name__ == "__main__":
    main()
