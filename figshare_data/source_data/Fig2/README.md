# Fig. 2 source data

This folder contains the numerical source data for Fig. 2 (held-out AUROC versus GenomeCF reliability axes).
- Manuscript figure: Fig. 2
- Figure title: Held-out AUROC versus three GenomeCF axes on the four core human tasks. High AUROC does not guarantee low reverse-complement instability, biologically sensible shuffle behavior or good calibration. Task labels are Pr = Promoters, EC = Enhancers (Cohn), EE = Enhancers (Ensembl), and OCR = Open chromatin.
- Source files: `results/publication/table2_main_results.csv`, `results/release/benchmark_summary.csv`
- Generation script: `package_src/genomecf/submission_data.py` via `genomecf build-submission-data`

Columns:
- `AUROC`: held-out AUROC on the official split.
- `RC_delta`: reverse-complement mean absolute probability delta.
- `mono_drop`: positive-class probability drop after mononucleotide shuffle.
- `ECE`: expected calibration error on the official split.
- `Brier`: Brier score on the official split.
- `source_registry_key`: composite identifier linking the point back to the benchmark summary rows.