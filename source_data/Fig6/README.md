# Fig. 6 source data

This folder contains the source data for the GenomeCF-Synth figure.
- Manuscript figure: Fig. 6
- Figure title: GenomeCF-Synth. The top row revisits the planted-motif benchmark; the bottom row adds shortcut-conflict tasks that separate rule-following from shortcut-following behavior.
- Source files: `results/release/synthetic_extended_summary.csv`
- Generation script: `package_src/genomecf/submission_data.py` via `genomecf build-submission-data`

Note: the current extended synthetic summary does not store AUPRC, so the `AUPRC` column is left blank rather than back-filled from another source.