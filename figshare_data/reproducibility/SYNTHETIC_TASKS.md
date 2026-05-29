# Synthetic Tasks

GenomeCF includes a synthetic mechanism benchmark with deterministic seeds and explicit shortcut controls.

## Completed tasks

- `gc_correlated`
  - motif and GC shortcut agree
- `gc_matched`
  - motif is the only reliable signal
- `gc_conflict`
  - training aligns motif and GC, test makes them disagree
- `two_motif_grammar`
  - positive requires two motifs with a spacing rule
- `motif_position_conflict`
  - training makes motif position predictive, test removes or reverses that shortcut

## Completed model coverage

- `kmer_logistic_regression`
- `small_cnn`
- `small_cnn_rc_aug`
- `dnabert2`
- `caduceus_ph`

## Reported metrics

- `AUROC`
- `ECE`
- `Brier`
- `RC mean absolute delta`
- `mononucleotide-shuffle drop`
- `dinucleotide-shuffle drop`
- `motif-disruption drop`
- `shortcut_conflict_accuracy`
- `rule_following_rate`
- `shortcut_following_rate`

Outputs:

- `results/release/synthetic_extended_summary.csv`
- `results/publication/appendix_synthetic_extended.csv`
- `figures/genomecf_synthetic_publication.png`

These tasks are the cleanest mechanism evidence in the benchmark because the intended rule and the shortcut are known by construction.
