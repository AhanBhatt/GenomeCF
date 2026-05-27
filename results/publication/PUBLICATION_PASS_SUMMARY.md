# GenomeCF Publication Pass Summary

## What changed
- Rewrote `genomecf_report.tex` as a standalone benchmark paper.
- Removed course-specific framing from the publication version.
- Replaced first-person singular phrasing with formal paper style.
- Expanded the introduction around the benchmark-gap framing.
- Added a substantial related-work section with BibTeX citations in `refs.bib`.
- Reframed the benchmark explicitly as:
  - core real-data benchmark
  - synthetic mechanism benchmark
  - extended screening tasks
- Switched the report to generated tables and figures under `results/publication/` and `figures/`.
- Configured `hyperref` to use colored links without PDF border boxes.
- Added a one-command publication build path via `python -m genomecf.build_publication`.
- Reformatted Table 1 to show readable task names with raw dataset IDs underneath.

## Regenerated figures
- `figures/genomecf_tradeoff_publication.png`
- `figures/genomecf_calibration_publication.png`
- `figures/genomecf_generalization_gap.png`
- `figures/genomecf_synthetic_publication.png`

## Regenerated tables and artifacts
- `results/publication/table1_task_overview.csv`
- `results/publication/table1_task_overview.tex`
- `results/publication/table2_main_results.csv`
- `results/publication/table2_main_results.tex`
- `results/publication/appendix_real_results.csv`
- `results/publication/appendix_real_results.tex`
- `results/publication/appendix_synthetic_results.csv`
- `results/publication/appendix_synthetic_results.tex`
- `results/publication/appendix_gc_only.csv`
- `results/publication/appendix_gc_only.tex`
- `results/publication/appendix_confounders.csv`
- `results/publication/appendix_confounders.tex`
- `results/publication/appendix_chrom_folds.csv`
- `results/publication/appendix_chrom_folds.tex`
- `results/publication/key_numbers.json`
- `results/publication/artifact_manifest.json`

## Commands run
- `python src\generate_publication_artifacts.py`
- `bibtex genomecf_report`
- `pdflatex -interaction=nonstopmode genomecf_report.tex`
- `python -m unittest tests\test_genomecf.py`
- `python -m genomecf.build_publication`

## Tests passed
- Registry and config sanity checks
- Reverse-complement involution
- 1-mer, 2-mer, and 3-mer preservation for shuffle tests
- Motif-preserving flank shuffle preserves the target motif span
- Motif disruption edits only the intended motif window
- Metric and bootstrap sanity checks
- Bootstrap determinism with a fixed seed
- Registry roundtrip

## Experiments not rerun in this pass
- No new full benchmark rerun for the real-data model matrix
- No new DNABERT-2 training or embedding extraction runs
- No Caduceus, Nucleotide Transformer, GROVER, or HyenaDNA result generation
- No new matched-negative benchmark matrix
- No new temperature-scaling or GC-balanced intervention results
- No new motif-attribution experiments on real tasks
- No five-fold chromosome-cross-validation result set

## Important honesty note
- The publication paper is grounded in the existing completed result files under:
  - `results/genomecf_real/`
  - `results/genomecf_synthetic/`
  - `results/genomecf_holdout_cnn/`
  - `results/gc_shortcut_check/`
- The newer `genomecf/` package scaffolding remains in the repo and supports future benchmark expansion, but the publication PDF does not claim results that were not completed in those legacy benchmark outputs.

## Remaining limitations
- The reported chromosome-held-out analysis is still the completed legacy `chr1/chr2` holdout, not the newer five-fold chromosome CV scaffold.
- Foundation-model coverage remains partial in the publication results.
- The appendix tables are aggregate tables plus confounder summaries, not a full per-seed per-perturbation dump in the PDF itself.
- Some LaTeX overfull-box warnings remain in raw-ID appendix tables, but the PDF compiles successfully and the rendered tables remain readable.
