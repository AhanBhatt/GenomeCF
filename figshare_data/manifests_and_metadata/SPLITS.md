# Splits

GenomeCF currently uses three main split styles in the release codebase.

## Official split

- Split ID: `official`
- Source: native train and test partitions from the underlying task
- Current use:
  - core official benchmark rows
  - diagnostic baselines
  - CNN and RC-aug CNN official rows
  - DNABERT-2 frozen official rows
  - Caduceus-Ph frozen official rows
  - appendix-only Nucleotide Transformer v2 frozen official rows on the focal tasks

## Five-fold chromosome grouped CV

- Split ID: `chromosome_5fold_cv`
- Human-only folds:
  - `A`: `chr1`, `chr2`
  - `B`: `chr3`, `chr4`, `chr5`
  - `C`: `chr6`, `chr7`, `chr8`
  - `D`: `chr9`, `chr10`, `chr11`, `chr12`
  - `E`: `chr13` through `chr22`, `chrX`, `chrY`

Protocol:

- test = current fold
- validation = next fold cyclically
- train = all remaining folds

Completed in this release:

- 6-mer logistic regression on all four core human tasks
- CNN on all four core human tasks
- RC-aug CNN on all four core human tasks
- DNABERT-2 on:
  - `human_nontata_promoters`
  - `human_enhancers_cohn`
- Caduceus-Ph on:
  - `human_nontata_promoters`
  - `human_enhancers_cohn`
  - `human_enhancers_ensembl`
  - `human_ocr_ensembl`

Included in the main paper:

- Figure 4 uses the completed five-fold chromosome-CV summary for the lightweight baselines, focal DNABERT-2 rows, and Caduceus-Ph on all four core human tasks
- Appendix tables include both the fold summary and the per-fold metrics

Pending:

- broader foundation-model chromosome-CV beyond DNABERT-2 focal tasks and Caduceus-Ph core-task coverage

## Matched-negative evaluation

- Split ID: `matched_test`
- Base training split: the official split
- Test-set change: the positive and negative evaluation set is rebuilt with confounder-aware matching

Current minimum matching:

- same chromosome or chromosome fold when available
- same length if fixed, or within a small window if variable
- GC fraction tolerance
- repeat fraction tolerance when available

Completed in this release:

- GC-only, 6-mer, CNN, and RC-aug CNN matched-negative evaluation on:
  - `human_nontata_promoters`
  - `human_enhancers_cohn`
  - `human_enhancers_ensembl`
- DNABERT-2 matched-negative evaluation on:
  - `human_nontata_promoters`
  - `human_enhancers_cohn`
  - `human_enhancers_ensembl`
- Caduceus-Ph matched-negative evaluation on:
  - `human_nontata_promoters`
  - `human_enhancers_cohn`
  - `human_enhancers_ensembl`

Included in the paper:

- matched-negative result table
- matched-negative confounder-balance table

Pending:

- broader foundation-model matched-negative retraining beyond focal DNABERT-2 and focal Caduceus-Ph heads
- broader matched-negative matrix beyond the completed focal tasks

## Split manifests

- Internal registry: `genomecf/config/splits.json`
- Exported registry: `configs/split_manifests.jsonl`
- Release summaries:
  - `results/release/chromosome_cv_summary.csv`
  - `results/release/chromosome_cv_fold_metrics.csv`
  - `results/release/matched_negative_confounders.csv`
  - `results/release/matched_negative_model_summary.csv`
