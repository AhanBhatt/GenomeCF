from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from genomecf.variant_tasks import ensure_variant_tasks_prepared, run_variant_evaluation, variant_task_ids


DEFAULT_MODELS = [
    ("kmer_logistic_regression", "classical", "none"),
    ("small_cnn", "supervised", "none"),
    ("small_cnn_rc_aug", "supervised", "none"),
    ("dnabert2", "frozen", "none"),
]

CALIBRATION_MODELS = [
    ("small_cnn", "supervised", "temperature"),
    ("small_cnn_rc_aug", "supervised", "temperature"),
    ("dnabert2", "frozen", "temperature"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GenomeCF variant-effect suite on public MaveDB regulatory MPRA tasks.")
    parser.add_argument("--tasks", nargs="*", default=variant_task_ids() or None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--include-calibration", action="store_true")
    args = parser.parse_args()

    tasks = args.tasks or variant_task_ids()
    ensure_variant_tasks_prepared(project_root=PROJECT_ROOT, task_ids=tasks)
    runs = list(DEFAULT_MODELS)
    if args.include_calibration:
        runs.extend(CALIBRATION_MODELS)

    for task_id in tasks:
        for model_id, mode, calibration in runs:
            print(f"variant {task_id} {model_id} calibration={calibration}")
            _, _, summary_path = run_variant_evaluation(
                task_name=task_id,
                model_name=model_id,
                split_name="official",
                mode=mode,
                seed=args.seed,
                calibration=calibration,
                project_root=PROJECT_ROOT,
            )
            print(f"  wrote {summary_path}")


if __name__ == "__main__":
    main()
