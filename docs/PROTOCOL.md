# GenomeCF protocol

This document summarizes the intended user protocol for running GenomeCF as a reusable counterfactual validation standard for DNA sequence models.

## 1. Install

```bash
pip install -e .[benchmark,dev]
```

For Caduceus-Ph, use the documented WSL2/Linux CUDA route in [CADUCEUS_SETUP.md](CADUCEUS_SETUP.md).

## 2. Reproduce the quickstart

```bash
genomecf reproduce-quickstart
```

This validates the local install, runs the lightweight smoke path, rebuilds the summary outputs, and emits:

- `results/release/quickstart/quickstart_report.json`

## 3. Run a focal benchmark row

```bash
genomecf evaluate --task human_enhancers_cohn --model dnabert2 --split official --mode frozen
```

Useful related commands:

```bash
genomecf external --task gue_human_tf_0 --model caduceus_ph --split official --mode frozen
genomecf variant --task mpra_bcl11a_enhancer --model small_cnn --split official --mode supervised
genomecf synth --task gc_conflict --model dnabert2 --split official --mode frozen
```

## 4. Validate the release registry

```bash
genomecf validate-results
genomecf check-report --results results/release/benchmark_registry.csv
```

These commands validate:

- registry completeness
- reporting-checklist compliance

## 5. Rebuild release-facing artifacts

```bash
genomecf reproduce-focal
genomecf reproduce-external
genomecf build-website
```

## 6. Inspect the canonical outputs

- registry: `results/release/benchmark_registry.csv`
- website: `docs/site/index.html`

## 7. Recommended reporting workflow

When evaluating a new DNA sequence model:

1. Run `official`, `matched_test`, and `chromosome_5fold_cv` where supported.
2. Export original, reverse-complement, mono-shuffle, and dinuc-shuffle rows.
3. Summarize GC-bin robustness.
4. Run at least one GenomeCF-Synth conflict task.
5. Validate the final result table with `genomecf check-report`.
6. Archive the registry row IDs used in the manuscript or report.
