# Result Schema

The canonical release registry lives at:

- `results/release/benchmark_registry.csv`
- `results/release/benchmark_registry.jsonl`

One row represents one:

- task
- split
- split fold
- model
- mode
- seed
- perturbation

The canonical long-form file is:

- `results/release/benchmark_registry.csv`

and the publication-facing aggregated file is:

- `results/release/benchmark_summary.csv`

Important fields include:

- task metadata:
  - `task_id`
  - `task_readable_name`
  - `tier`
  - `species`
- split metadata:
  - `split_id`
  - `split_fold`
- model metadata:
  - `model_id`
  - `model_readable_name`
  - `mode`
  - `model_checkpoint`
  - `tokenizer_name`
  - `pooling`
  - `max_length`
  - `embedding_dim`
  - `batch_size`
  - `truncation`
- predictive metrics:
  - `auroc`
  - `auprc`
  - `accuracy`
  - `balanced_accuracy`
  - `mcc`
  - `f1`
- calibration metrics:
  - `ece`
  - `brier`
  - `calibration_shift`
  - `calibration_method`
- counterfactual metrics:
  - `mean_abs_delta`
  - `flip_rate`
  - `positive_prob_drop`
  - perturbation-specific summaries in the release table:
    - `rc_*`
    - `mono_*`
    - `dinuc_*`
    - `motif_*`
- intervention metadata:
  - `intervention_id`
- counts and sequence-length summary:
  - `train_count`
  - `val_count`
  - `test_count`
  - `positive_count`
  - `negative_count`
  - `sequence_length_summary`
- reproducibility metadata:
  - `runtime_seconds`
  - `device`
  - `package_versions`
  - `config_hash`
  - `data_hash`
  - `created_at`

The release summary lives at:

- `results/release/benchmark_summary.csv`

and the model-by-task coverage matrix lives at:

- `results/release/model_task_matrix.csv`

Paper-facing derived artifacts are written to:

- `results/publication/*.csv`
- `results/publication/*.tex`

Additional validation and environment artifacts used by the release include:

- `results/release/foundation_loader_status.csv`
- `results/release/nt_validation_report.json`
- `results/release/validation_report.json`

Nature Methods release artifacts additionally include:

- `results/release/external_validation_summary.csv`
- `results/release/external_validation_family_summary.csv`
- `results/release/external_transfer_prediction.csv`
- `results/release/external_transfer_stats.json`
- `results/release/biological_case_study.csv`
- `results/release/nature_methods_artifact_manifest.json`
- `results/release/nature_methods/nature_methods_summary.csv`

Website and reporting-standard artifacts:

- `docs/site/leaderboard.csv`
- `docs/site/index.html`
- `docs/reporting_checklist.yaml`
- `results/release/reporting_check_report.json`
