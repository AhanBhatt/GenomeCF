# Foundation Models

GenomeCF currently includes three foundation-model tracks with different release status.

## DNABERT-2

Status:

- completed main-paper frozen baseline

Coverage:

- official rows on all four core human tasks
- five-fold chromosome grouped CV on:
  - `human_nontata_promoters`
  - `human_enhancers_cohn`
- matched-negative evaluation on:
  - `human_nontata_promoters`
  - `human_enhancers_cohn`
  - `human_enhancers_ensembl`
- temperature scaling on focal tasks
- matched-negative-trained frozen head on focal tasks

## Caduceus-Ph

Status:

- completed main-paper frozen baseline
- requires the documented WSL2/Linux CUDA path

Coverage:

- official rows on all four core human tasks
- five-fold chromosome grouped CV on all four core human tasks
- matched-negative evaluation on:
  - `human_nontata_promoters`
  - `human_enhancers_cohn`
  - `human_enhancers_ensembl`
- temperature scaling on focal tasks
- matched-negative-trained frozen head on focal tasks

See:

- `docs/CADUCEUS_SETUP.md`
- `envs/caduceus.yml`

## Nucleotide Transformer v2

Status:

- appendix-only diagnostic baseline

Coverage:

- focal official rows on:
  - `human_nontata_promoters`
  - `human_enhancers_cohn`

Reason for appendix-only status:

- the loader works
- tokenization and embedding mechanics pass
- reproducibility checks pass
- the current frozen mean-pooled protocol underperforms on internal validation probes

See:

- `results/release/nt_validation_report.json`
- `results/release/foundation_loader_status.csv`

## Practical commands

Official frozen evaluation:

```bash
genomecf evaluate --task human_enhancers_cohn --model dnabert2 --split official --mode frozen
genomecf evaluate --task human_enhancers_cohn --model caduceus_ph --split official --mode frozen
```

External assay evaluation:

```bash
genomecf external --task gue_human_tf_0 --model dnabert2 --split official --mode frozen
genomecf external --task gue_human_tf_0 --model caduceus_ph --split official --mode frozen
```
