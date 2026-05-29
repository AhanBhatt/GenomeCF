# Motif Analysis

GenomeCF now includes controlled real-task motif probes in addition to the synthetic planted-motif benchmark.

## Real-task motif probe pipeline

Tasks:

- `human_nontata_promoters`
- `human_enhancers_cohn`
- `human_enhancers_ensembl`

Models:

- `kmer_logistic_regression`
- `small_cnn`
- `small_cnn_rc_aug`
- `dnabert2`
- `caduceus_ph`
- appendix-only diagnostic `nucleotide_transformer_v2`

Procedure:

1. Scan official-split positive sequences for curated exact-match motif hits from the task manifests.
2. Sample up to 1,000 motif-containing positives per task.
3. Evaluate:
   - original sequence
   - motif-core disruption
   - GC-preserving motif-core disruption
   - random non-motif edit control with the same number of changed bases
4. Report:
   - mean original probability
   - motif-disruption drop
   - GC-preserving motif-disruption drop
   - random-edit drop
   - motif-minus-random effect
   - bootstrap confidence intervals

Outputs:

- `results/release/real_motif_probe_summary.csv`
- `results/release/real_motif_probe_details.csv`
- `results/publication/table7_motif_summary.csv`

## Interpretation

The current real-task motif evidence is intentionally cautious.

- The synthetic tasks show that GenomeCF can detect motif-vs-shortcut behavior when ground truth is known.
- The real-task motif-minus-random effects are generally small.
- That is treated as a controlled negative result, not as evidence that the benchmark lacks signal.

GenomeCF therefore uses the real-task motif section to show that current benchmark predictions are not largely explained by a small set of simple exact-match motifs alone.
