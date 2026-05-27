# GenomeCF Expanded Summary

## Scope

- Real benchmark tasks: 6 binary genomic sequence tasks.
- Human chromosome-held-out evaluation: 4 tasks for k-mer logistic regression, plus learned-model holdout runs on the 2 core human tasks.
- Learned models with 5 random seeds: small CNN and reverse-complement-augmented CNN.
- Pretrained model baseline: DNABERT-2 frozen embeddings with logistic regression on the 2 core tasks.
- Perturbations: reverse complement, mononucleotide-preserving shuffle, dinucleotide-preserving shuffle.
- Synthetic benchmark: planted motif tasks with GC-correlated and GC-matched conditions.

## Key real-data findings

- The strongest official-split AUROC in the expanded suite is 0.898 from kmer_logistic_regression on human_nontata_promoters.
- The largest reverse-complement instability among the real tasks is 0.400 from kmer_logistic_regression on drosophila_enhancers_stark (official).
- Reverse-complement augmentation consistently helps on several human tasks. On `human_enhancers_ensembl`, ensemble AUROC rises from 0.839 to 0.884 and mono-shuffle drop rises from 0.178 to 0.320.
- DNABERT-2 embeddings give the best AUROC on `human_enhancers_cohn` at 0.830, but still show negative mononucleotide-shuffle drop (-0.049), which means high accuracy does not prevent shortcut-like behavior.
- On promoters, the k-mer model remains strongest by AUROC (0.898 official) but has much larger reverse-complement instability (0.263) than either CNN ensemble (0.044) or DNABERT-2 (0.078).

## Split findings

- On `human_nontata_promoters`, k-mer AUROC drops from 0.898 on the official split to 0.840 on the chromosome-held-out split.
- On the same promoter task, the chromosome-held-out CNN ensemble reaches 0.877 AUROC, which nearly closes the gap to the official k-mer result while keeping reverse-complement instability low at 0.038.
- On `human_enhancers_cohn`, reverse-complement-augmented CNN improves chromosome-held-out AUROC from 0.671 to 0.687 relative to the non-augmented CNN.

## Synthetic findings

- In the GC-correlated synthetic condition, every model reaches AUROC about 1.0 while motif-disruption and shuffle drops stay near zero. This means the models can solve the task without relying on the planted motif.
- In the GC-matched synthetic condition, AUROC stays about 1.0 but motif-disruption and shuffle drops jump to about 0.87 to 0.95, showing that once the shortcut is removed, the models track the true motif rule.
- This synthetic result supports the interpretation of the real-data benchmark: high AUROC alone cannot tell whether the model learned the intended biological mechanism.

## Files

- Real benchmark summary: `C:\Users\bhatt\Documents\Masters\Subjects\AM-234 Machine Learning and Artificial Intelligence in Genomics\Project\results\genomecf_real\summary_metrics.csv`
- Synthetic benchmark summary: `C:\Users\bhatt\Documents\Masters\Subjects\AM-234 Machine Learning and Artificial Intelligence in Genomics\Project\results\genomecf_synthetic\summary_metrics.csv`
- Split comparison table: `C:\Users\bhatt\Documents\Masters\Subjects\AM-234 Machine Learning and Artificial Intelligence in Genomics\Project\results\genomecf_split_comparison.csv`
- Holdout comparison figure: `C:\Users\bhatt\Documents\Masters\Subjects\AM-234 Machine Learning and Artificial Intelligence in Genomics\Project\figures\genomecf_holdout_comparison.png`
- Official benchmark overview figure: `C:\Users\bhatt\Documents\Masters\Subjects\AM-234 Machine Learning and Artificial Intelligence in Genomics\Project\figures\genomecf_overview.png`
- Synthetic benchmark figure: `C:\Users\bhatt\Documents\Masters\Subjects\AM-234 Machine Learning and Artificial Intelligence in Genomics\Project\figures\genomecf_synthetic_summary.png`