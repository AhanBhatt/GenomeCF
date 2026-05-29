# Fig. 4 source data

This folder contains the source data for the external validation and external-reliability prediction figure.
- Manuscript figure: Fig. 4
- Figure title: External biological validation. Left, official and worst-bin assay-family results across TF-binding, histone-mark and MPRA variant-effect tasks. Right, external biological reliability is better explained by GenomeCF counterfactual metrics than by held-out core AUROC alone.
- Source files: `results/release/external_validation_family_summary.csv`, `results/release/external_validation_summary.csv`, `results/release/external_transfer_prediction.csv`, `results/release/external_transfer_stats.json`
- Generation script: `package_src/genomecf/submission_data.py` via `genomecf build-submission-data`

Files:
- `fig4_external_assay_family_summary.csv`: assay-family summary metrics plotted on the left side of Fig. 4.
- `fig4_external_prediction_points.csv`: model-configuration-task points used for the right-side prediction panels.
- `fig4_external_prediction_model_fits.csv`: fitted regressions, LOFO analyses and permutation summaries supporting the prediction comparisons.