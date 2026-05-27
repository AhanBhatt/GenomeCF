# GC-Bin Robustness

GenomeCF reports group robustness across composition strata, not only average AUROC.

## Protocol

- Use the official-split test set for each core human task.
- Compute GC fraction per sequence.
- Divide the test set into 5 GC-quantile bins.
- Report per-bin:
  - `AUROC`
  - `ECE`
  - `Brier`
- Also report:
  - `overall_auroc`
  - `worst_bin_auroc`
  - `best_bin_auroc`
  - `gc_bin_auroc_gap`
  - `worst_bin_ece`
  - `gc_bin_ece_gap`

## Completed coverage

Tasks:

- `human_nontata_promoters`
- `human_enhancers_cohn`
- `human_enhancers_ensembl`
- `human_ocr_ensembl`

Models:

- `kmer_logistic_regression`
- `small_cnn`
- `small_cnn_rc_aug`
- `dnabert2`
- `caduceus_ph`

Outputs:

- `results/release/gc_bin_summary.csv`
- `results/release/gc_bin_by_bin.csv`
- `results/publication/table6_gc_bin_summary.csv`
- `figures/genomecf_gc_bin_robustness.png`

This analysis is used to show that average AUROC can hide substantial worst-group robustness gaps across composition regimes.
