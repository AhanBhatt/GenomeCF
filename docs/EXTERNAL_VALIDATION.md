# External biological validation

GenomeCF extends beyond the core regulatory benchmark into three external assay families:

- TF binding
  - `gue_human_tf_0`
  - `gue_human_tf_1`
- Histone marks
  - `gue_emp_h3k4me3`
  - `gue_emp_h3k14ac`
- Regulatory MPRA variant effects
  - `mpra_bcl11a_enhancer`
  - `mpra_f9_promoter`
  - `mpra_hbb_promoter`
  - `mpra_ldlr_promoter`
  - `mpra_myc_enhancer`

The current Nature Methods release evaluates 82 model-configuration-task pairs across these assay families.

Key release outputs:

- `results/release/external_validation_summary.csv`
- `results/release/external_validation_family_summary.csv`
- `results/release/external_transfer_prediction.csv`
- `results/release/external_transfer_stats.json`
- `results/release/biological_case_study.csv`
- `results/release/variant_effect/`

Headline result:

- core AUROC alone explains little external biological reliability variance under the expanded analysis (`R^2 = 0.062`)
- the compact GenomeCF Shortcut Score is informative but weaker than the full profile in this release
- a fuller GenomeCF profile combining shortcut, calibration, matched-negative, and GC-bin metrics reaches the strongest external prediction fit in the current release (`R^2 = 0.383`; bootstrap delta over AUROC-only `0.321`, 95% CI `0.182-0.515`)

Reproduction:

```bash
genomecf reproduce-external
genomecf build-paper
```
