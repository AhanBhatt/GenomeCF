# Release Scope

GenomeCF is released as a benchmark package with honest tiering.

## Included in Main Paper Claims

- core short-context real tasks
- screening tasks clearly separated from core claims
- synthetic ground-truth tasks
- GC-only, 6-mer, CNN, RC-aug CNN, DNABERT-2, and Caduceus-Ph results where available
- reverse-complement, mononucleotide, and dinucleotide perturbations
- calibration metrics and paper-ready figures

## Completed in the Current Release Registry

- official diagnostic baselines on the four core human tasks
- DNABERT-2 frozen official rows on all four core human tasks
- Caduceus-Ph frozen official rows on all four core human tasks
- five-fold chromosome grouped CV for 6-mer, CNN, and RC-aug CNN on the four core human tasks plus focal DNABERT-2 rows
- matched-negative evaluation for GC-only, 6-mer, CNN, RC-aug CNN, DNABERT-2, and Caduceus-Ph on the three focal real tasks
- temperature-scaled focal CNN and DNABERT-2 variants
- GC-balanced CNN variants on the three focal human tasks
- matched-negative retraining for CNN and RC-aug CNN on the three focal human tasks
- real-task motif-disruption metrics for package-run official and matched-negative evaluations

## Appendix-Only or Expanded-Scope Coverage

- Nucleotide Transformer benchmark-scale results beyond the appendix-only focal diagnostic rows
- GROVER benchmark-scale results
- HyenaDNA benchmark-scale results
- full long-context track
- broader foundation-model matched-negative retraining
- real-task attribution and motif recovery

Internal project-tracking files are kept outside the public GitHub repo. For public use, rely on this scope document together with the canonical release registry in `results/release/`.
