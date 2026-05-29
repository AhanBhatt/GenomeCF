# Fig. 1 source data

This folder documents the schematic source mapping for Fig. 1 in the GenomeCF manuscript.
Fig. 1 is a resource overview schematic rather than a quantitative plot, so the key source-data file is `panel_definitions.csv`.

- Manuscript figure: Fig. 1
- Figure title: GenomeCF resource overview. GenomeCF combines counterfactual stress tests on core real-data tasks, external biological assays, GenomeCF-Synth mechanism tasks, and a software/reporting stack designed for reusable community evaluation.
- Source files: `results/publication/table1_task_overview.csv`, `results/release/benchmark_registry.csv`, `results/release/external_validation_summary.csv`, `results/release/biological_case_study.csv`, `results/release/synthetic_extended_summary.csv`, `docs/reporting_checklist.yaml`, `docs/site/leaderboard.csv`
- Generation script: `package_src/genomecf/submission_data.py` via `genomecf build-submission-data`

Columns:
- `panel_or_box`: logical block in the schematic.
- `label`: short visible label used in the schematic.
- `description`: summary of the block content.
- `input_artifact_or_doc`: registry-backed artifact or documentation source used for the block.
- `notes`: additional interpretation guidance.