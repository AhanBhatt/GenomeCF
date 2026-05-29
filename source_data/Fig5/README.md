# Fig. 5 source data

This folder contains the case-study source data for the MPRA biological-use figure.
- Manuscript figure: Fig. 5
- Figure title: MPRA biological case studies. Case A shows that GenomeCF changes configuration choice on the BCL11A enhancer MPRA task by favoring a temperature-scaled CNN over the default CNN. Case B shows that GenomeCF changes model choice on the MYC enhancer MPRA task by favoring DNABERT-2 for top-ranked variant nomination even though the 6-mer baseline has stronger global AUROC and AUPRC.
- Source files: `results/release/biological_case_study.csv`, `results/release/external_validation_summary.csv`
- Generation script: `package_src/genomecf/submission_data.py` via `genomecf build-submission-data`

The `objective` column distinguishes BCL11A variant prioritization from MYC top-k nomination, matching the manuscript wording.