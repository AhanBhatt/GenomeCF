# Biological case study

The current Nature Methods release uses two MPRA variant-effect case studies to show that GenomeCF changes practical deployment decisions, not only benchmark reporting.

## Case A: BCL11A enhancer MPRA

- task id: `mpra_bcl11a_enhancer`
- assay family: `Variant effect`
- use case: configuration choice within the same CNN backbone

Decision tested:

- an AUROC-only workflow would keep the standard `small_cnn`
- a GenomeCF-aware workflow selects the temperature-scaled `small_cnn` because the core benchmark shows better calibration without changing the backbone

Observed external effect:

- AUROC improves from `0.434` to `0.583`
- AUPRC improves from `0.097` to `0.167`
- top-k enrichment improves from `1.23` to `2.06`
- worst-GC-bin AUROC improves from `0.369` to `0.539`

## Case B: MYC enhancer MPRA

- task id: `mpra_myc_enhancer`
- assay family: `Variant effect`
- use case: model choice for high-confidence variant nomination

Decision tested:

- an AUROC-only workflow would prefer `kmer_logistic_regression`
- a GenomeCF-aware workflow can prefer `dnabert2` when the biological objective is top-ranked variant nomination rather than global rank quality

Observed external trade-off:

- 6-mer logistic regression has stronger global AUROC and AUPRC:
  - AUROC `0.617`
  - AUPRC `0.273`
- DNABERT-2 has stronger top-k prioritization:
  - top-k enrichment `1.93` versus `1.50`

Key artifacts:

- `results/release/biological_case_study.csv`
- `figures/genomecf_biological_case_study.png`
- `results/publication/table10_case_study_summary.csv`

Reproduction:

```bash
genomecf reproduce-external
```
