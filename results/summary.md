# Experimental Summary

## Main findings

### human_nontata_promoters
- AUROC: k-mer logistic regression 0.936, small CNN 0.917.
- Reverse-complement mean absolute probability change: k-mer 0.258, CNN 0.121.
- Positive-sequence GC-shuffle probability drop: k-mer -0.014, CNN -0.024.

### human_enhancers_cohn
- AUROC: k-mer logistic regression 0.769, small CNN 0.757.
- Reverse-complement mean absolute probability change: k-mer 0.198, CNN 0.049.
- Positive-sequence GC-shuffle probability drop: k-mer -0.111, CNN 0.010.

## Shortcut check

- human_nontata_promoters: a GC-fraction-only logistic regression reaches AUROC 0.796 with negative vs positive mean GC fractions of 0.484 and 0.627.
- human_enhancers_cohn: a GC-fraction-only logistic regression reaches AUROC 0.734 with negative vs positive mean GC fractions of 0.394 and 0.469.

## Interpretation

- The promoter dataset is easier than the enhancer dataset for both models.
- The k-mer model reaches the highest raw AUROC on both datasets, especially on promoters.
- The CNN is substantially more stable under reverse-complement perturbations, especially on promoters.
- Both models show shortcut sensitivity on GC-matched shuffled promoters, which suggests they are still using composition-level cues rather than only biologically meaningful motifs.
- On enhancers, the CNN shows better counterfactual behavior than the k-mer model even though its AUROC is slightly lower.
- The GC-only baseline shows that simple composition already carries substantial predictive signal, which helps explain why shuffled sequences can still receive strong positive scores.

## Files

- Combined metrics: `C:\Users\bhatt\Documents\Masters\Subjects\AM-234 Machine Learning and Artificial Intelligence in Genomics\Project\results\combined_model_comparison.csv`
- Comparison figure: `C:\Users\bhatt\Documents\Masters\Subjects\AM-234 Machine Learning and Artificial Intelligence in Genomics\Project\figures\model_comparison_grid.png`
- GC shortcut metrics: `C:\Users\bhatt\Documents\Masters\Subjects\AM-234 Machine Learning and Artificial Intelligence in Genomics\Project\results\gc_shortcut_check\gc_baseline_metrics.csv`
- GC distribution figure: `C:\Users\bhatt\Documents\Masters\Subjects\AM-234 Machine Learning and Artificial Intelligence in Genomics\Project\figures\gc_fraction_by_class.png`