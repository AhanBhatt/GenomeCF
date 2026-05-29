# Reproducibility

## Editable Install

```bash
pip install -e .[benchmark,dev]
```

## Smoke Reproduction

```bash
genomecf smoke-test
```

## Nature Methods Quickstart

```bash
genomecf reproduce-quickstart
```

## One Evaluation

```bash
genomecf evaluate --task human_nontata_promoters --model kmer_logistic_regression --split official --mode frozen --seed 42
```

## Export Perturbations

```bash
genomecf perturb --task human_nontata_promoters --perturbation k2_shuffle
```

## Build Release Registry and Summaries

```bash
python src/generate_release_upgrade_artifacts.py
genomecf summarize --suite core
genomecf summarize --suite nature_methods
genomecf build-appendix
genomecf validate-results
genomecf check-report --results results/release/benchmark_registry.csv
genomecf build-website
```


## Caduceus Environment

Completed Caduceus-Ph rows were run through WSL2/Linux CUDA rather than the default Windows CPU environment. See:

- `docs/CADUCEUS_SETUP.md`
- `envs/caduceus.yml`
- `scripts/run_caduceus_cv.sh`
- `scripts/run_foundation_mitigation.sh`

## Canonical Outputs


- canonical registry: `results/release/benchmark_registry.csv`
- release summary: `results/release/benchmark_summary.csv`

- release validation report: `results/release/validation_report.json`
- website root: `docs/site/index.html`
- reporting checklist: `docs/reporting_checklist.yaml`

## Environment Note

The benchmark and smoke paths can be reproduced locally on CPU. Larger foundation-model experiments are intended for WSL2 or Linux with CUDA.

## Honest Scope Note

- DNABERT-2 official frozen rows are completed on the four core human tasks.
- Caduceus-Ph official frozen rows are completed on the four core human tasks through the WSL2/Linux CUDA environment.
- Caduceus-Ph chromosome grouped CV is completed on the four core human tasks through the WSL2/Linux CUDA environment.
- Nucleotide Transformer v2 official frozen rows are completed on the two focal human tasks, but the current frozen protocol is not validated strongly enough for main-paper use and is treated as appendix-only diagnostic coverage.
- Five-fold chromosome CV is completed for the lightweight baselines, for focal DNABERT-2 rows, and for Caduceus-Ph on the four core human tasks.
- Matched-negative evaluation is complete for GC-only, 6-mer, CNN, RC-aug CNN, DNABERT-2, and Caduceus-Ph on the three focal real tasks.
- Temperature scaling is complete for the focal CNN, DNABERT-2, and Caduceus-Ph rows.
- GC-balanced training and matched-negative retraining are complete for CNN and RC-aug CNN on the completed focal tasks.
- matched-negative-trained frozen heads are complete for DNABERT-2 and Caduceus-Ph on the focal tasks.
- GC-bin robustness summaries are complete on the four core human tasks.
- Real-task motif probes now include GC-preserving edits and random-edit controls, but they remain limited and do not yet constitute a full attribution or motif-recovery study.
