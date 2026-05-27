from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from genomecf import (
    create_counterfactuals,
    evaluate_prediction_set,
    load_task_frame,
    predict_cnn_probabilities,
    save_json,
    train_cnn_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
REPO_ROOT = PROJECT_ROOT / "external" / "genomic_benchmarks"
RESULTS_ROOT = PROJECT_ROOT / "results" / "genomecf_holdout_cnn"
SEEDS = [11, 23, 37, 49, 61]
TASKS = ["human_nontata_promoters", "human_enhancers_cohn"]


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for dataset_name in TASKS:
        task = load_task_frame(
            dataset_name,
            DATA_ROOT,
            REPO_ROOT,
            split_scheme="chromosome_holdout",
            seed=2026,
            train_per_label=3000,
            test_per_label=1000,
        )

        for rc_augment, model_family in [(False, "small_cnn"), (True, "small_cnn_rc_aug")]:
            per_seed_predictions = []
            for seed in SEEDS:
                model, history = train_cnn_model(
                    task.train["sequence"].tolist(),
                    task.train["label"].tolist(),
                    seed=seed,
                    rc_augment=rc_augment,
                )
                original_probs = predict_cnn_probabilities(model, task.test["sequence"].tolist())
                counterfactuals = create_counterfactuals(task.test["sequence"].tolist(), seed=seed)
                cf_probs = {name: predict_cnn_probabilities(model, seqs) for name, seqs in counterfactuals.items()}
                metrics = evaluate_prediction_set(task.test["label"].to_numpy(), original_probs, cf_probs, seed=seed)
                row = {
                    "dataset": dataset_name,
                    "split_scheme": "chromosome_holdout",
                    "model_family": model_family,
                    "seed": seed,
                    **metrics,
                }
                rows.append(row)
                per_seed_predictions.append({"original": original_probs, **cf_probs})
                save_json(row, RESULTS_ROOT / "per_run" / f"{dataset_name}__{model_family}__seed{seed}.json")
                save_json({"history": history}, RESULTS_ROOT / "per_run" / f"{dataset_name}__{model_family}__seed{seed}__history.json")
                print("holdout", dataset_name, model_family, seed, metrics["auroc"])

            ensemble_original = np.mean([item["original"] for item in per_seed_predictions], axis=0)
            ensemble_counterfactuals = {
                name: np.mean([item[name] for item in per_seed_predictions], axis=0)
                for name in ["reverse_complement", "mono_shuffle", "dinuc_shuffle"]
            }
            ensemble_metrics = evaluate_prediction_set(task.test["label"].to_numpy(), ensemble_original, ensemble_counterfactuals, seed=2026)
            save_json(
                {
                    "dataset": dataset_name,
                    "split_scheme": "chromosome_holdout",
                    "model_family": model_family,
                    "n_seeds": len(SEEDS),
                    **ensemble_metrics,
                },
                RESULTS_ROOT / f"{dataset_name}__{model_family}__ensemble.json",
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_ROOT / "per_run_metrics.csv", index=False)
    print(frame.groupby(["dataset", "model_family"])["auroc"].mean().reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
