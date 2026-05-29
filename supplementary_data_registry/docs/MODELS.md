# Models

The release model catalog is exported to:

- `configs/model_manifests.jsonl`

## Completed in the current release

- `gc_only`
- `cpg_only`
- `length_only`
- `repeat_only`
- `kmer_logistic_regression`
- `small_cnn`
- `small_cnn_rc_aug`
- `dnabert2`
- `caduceus_ph`

Additional completed foundation-model coverage:

- `dnabert2`
  - official rows on all four core human tasks
  - five-fold chromosome grouped CV on:
    - `human_nontata_promoters`
    - `human_enhancers_cohn`
  - temperature scaling and matched-negative-trained head on the two focal tasks
- `caduceus_ph`
  - official rows on all four core human tasks
  - five-fold chromosome grouped CV on all four core human tasks
  - matched-negative evaluation on the three focal real tasks
  - temperature scaling and matched-negative-trained head on the two focal tasks

## Completed only as appendix-only diagnostic rows

- `nucleotide_transformer_v2`
  - loader works
  - frozen focal-task rows exist
  - current validation verdict is not strong enough for main-paper use under the current mean-pooled frozen protocol
  - see `results/release/nt_validation_report.json`

## Failed in the default Windows CPU environment

- `caduceus_ph`
  - blocked by the missing `mamba_ssm` dependency required by the checkpoint on the default Windows CPU setup
  - completed instead through the documented WSL2/Linux CUDA path

Environment-specific foundation-model status is documented in:

- `results/release/foundation_loader_status.csv`
- `results/release/nt_validation_report.json`
- `docs/CADUCEUS_SETUP.md`

## Implemented or scaffolded but not completed

- `xgboost_sequence`
- `grover`
- `hyenadna_long`

## Included in the paper

- `GC-only logistic regression`
- `6-mer logistic regression`
- `CNN`
- `RC-aug CNN`
- `DNABERT-2`
- `Caduceus-Ph`

Main-paper model comparisons now include:

- official real-data matrix
- chromosome-CV summary
- matched-negative evaluation
- mitigation summary
- GC-bin robustness
- controlled motif probes

## Appendix-only or diagnostic in the paper

- `Nucleotide Transformer v2` on the focal official tasks

## Interpretation

- Diagnostic baselines test whether simple confounders already explain task labels.
- Classical baselines test whether local composition and short k-mer content are enough.
- CNNs test supervised sequence learning with and without reverse-complement augmentation.
- Foundation-model entries record modern pretrained baselines and their current release status. In the current release, DNABERT-2 and Caduceus-Ph are completed on the four core human tasks, while Nucleotide Transformer v2 is retained as an appendix-only diagnostic focal-task baseline because the current frozen protocol is not trusted as main-paper evidence.
