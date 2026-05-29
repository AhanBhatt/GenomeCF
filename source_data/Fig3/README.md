# Fig. 3 source data

This folder contains the source data for the foundation-model comparison and adaptation figure.
- Manuscript figure: Fig. 3
- Figure title: Foundation-model comparison on the focal tasks. Top: frozen DNABERT-2 and Caduceus-Ph official AUROC and reverse-complement instability. Bottom: completed interventions on the same models. Temperature scaling improves calibration, whereas matched-negative-trained heads alter matched-test AUROC and sometimes trade off official performance for confounder-controlled robustness.
- Source files: `results/publication/table5_mitigation_summary.csv`
- Generation script: `package_src/genomecf/submission_data.py` via `genomecf build-submission-data`

Rows are in long format. Each row stores one metric value for one task/model/intervention phase while also carrying the phase-wide AUROC, matched-negative AUROC, calibration and RC context values.