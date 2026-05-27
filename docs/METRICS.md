# Metrics

GenomeCF combines predictive, calibration, and perturbation metrics.

## Predictive Metrics

- `AUROC`
- `AUPRC`
- `accuracy`
- `balanced_accuracy`
- `MCC`
- `F1`

## Calibration Metrics

- `ECE`
- `Brier`
- `calibration_shift`

GenomeCF uses:

- 10 equal-width probability bins for `ECE`
- binary-label Brier score on predicted positive-class probabilities
- perturbation calibration shift defined as perturbed `ECE` minus original-sequence `ECE`

## Counterfactual Consistency Metrics

- `mean_abs_delta`
- `flip_rate`
- `positive_prob_drop`

Important publication-facing variants are:

- `RC mean absolute delta`
- `RC flip rate`
- `mononucleotide-shuffle positive-probability drop`
- `dinucleotide-shuffle positive-probability drop`
- `motif-disruption positive-probability drop`

Interpretation:

- lower `RC mean absolute delta` and lower `RC flip rate` are better
- positive shuffle-drop values are desirable because confidence should fall when sequence order is weakened
- negative shuffle-drop values mean the model became more confident after perturbation

## Shortcut Score

`GenomeCF Shortcut Score` is a summary ranking computed from within-task ranks of:

- reverse-complement instability
- reverse-complement flip rate
- mononucleotide retention
- dinucleotide retention
- calibration shift
- worst-GC-bin AUROC gap
- matched-negative AUROC drop
- GC-only explainability ratio

Higher is worse. This score is a summary diagnostic, not the primary scientific proof.

## Current Paper Scope

The publication PDF highlights:

- predictive metrics for the core official matrix
- Caduceus-Ph rows in the official matrix
- perturbation metrics for reverse complement, mononucleotide shuffle, and dinucleotide shuffle
- calibration metrics on focal tasks
- five-fold chromosome-CV metrics for lightweight baselines on all core tasks plus focal DNABERT-2 rows
- five-fold chromosome-CV metrics for Caduceus-Ph on all core tasks
- matched-negative deltas for GC-only, 6-mer, CNN, RC-aug CNN, DNABERT-2, and Caduceus-Ph on the three focal real tasks
- mitigation deltas for focal CNNs, focal DNABERT-2 rows, focal Caduceus-Ph rows, GC-balanced CNN training, and matched-negative retraining where completed
- GC-bin robustness summaries on all four core human tasks
- controlled real-task motif-probe summaries with random-edit controls

Appendix-only diagnostic coverage:

- Nucleotide Transformer v2 focal official rows
- Nucleotide Transformer v2 motif-disruption rows

Not every metric is completed for every model family. The public release-facing scope is summarized in `docs/RELEASE_SCOPE.md`, and the canonical source of truth remains the release registry in `results/release/`.
