# Nature Methods submission checklist (GenomeCF)

This checklist covers the public Resource-manuscript package. The cover letter is intentionally excluded because the author will complete it separately.

## Manuscript format

- [x] Content type selected: `Resource`
- [x] Abstract is at or below 150 words (`149`)
- [x] Main-text word count estimated and recorded (`2,969`)
- [x] Main display items are no more than 6 total
- [x] Introduction has no heading
- [x] Results section is present
- [x] Discussion section is present and has no subheadings
- [x] Online Methods section is present with short topical subheadings
- [x] Availability sections are present:
  - Data Availability
  - Code Availability
  - Benchmark Availability
  - Model Availability
  - Environment Availability
  - Reproducibility Statement
- [x] Supplementary Information statement is placed after the references

## Figures, tables and supplementary material

- [x] Main figures are cited sequentially as `Fig. 1` to `Fig. 6`
- [x] Supplementary items are cited in sequence
- [x] Every supplementary item is cited at least once in the main text or Online Methods
- [x] Dense numeric tables moved to the supplement
- [x] Main figures have readable labels and consistent styling for initial submission
- [x] Source-data files generated for all six main figures in `source_data/`

## Citations, references and traceability

- [x] Main PDF builds
- [x] Supplement PDF builds
- [x] All citations resolved
- [x] All references resolved
- [x] No `Figure ??`, `Table ??`, `Section ??`, `Reference ??` or unresolved question-mark placeholders in the built PDFs
- [x] Strict traceability pass available:
  - `results/release/paper_claim_traceability.csv`
  - `results/release/paper_claim_traceability.html`
  - `results/release/statistical_claims.csv`
- [x] Numeric-reference path documented without breaking the default build:
  - copy `paper/reference_style.numeric.example.tex` to `paper/reference_style.tex` for the optional numeric `unsrtnat` build

## Software and reproducibility package

- [x] `pyproject.toml` present
- [x] `README.md` present
- [x] `LICENSE` present
- [x] `CITATION.cff`, `.zenodo.json`, and `codemeta.json` present
- [x] `environment.yml` present
- [x] `Dockerfile` present
- [x] `apptainer.def` present
- [x] GitHub Actions workflows present:
  - `.github/workflows/ci.yml`
  - `.github/workflows/docker.yml`
  - `.github/workflows/docs.yml`
- [x] Quickstart, protocol and reporting-standard docs present
- [x] Benchmark registry and release bundle paths recorded

## Validation commands

Run and archive the outputs used for the submission package:

```bash
python -m pip install -e .[benchmark,dev]
genomecf --help
python -m pytest
python -m genomecf.cli validate-results
python -m genomecf.cli check-report --results results/release/benchmark_registry.csv
python -m genomecf.cli trace-paper --strict
python -m genomecf.cli reproduce-quickstart
python -m genomecf.cli build-paper
python -m genomecf.cli build-supplement
python -m genomecf.cli build-website
```

If Docker is not available locally, keep the CI-configured Docker workflow but do not claim local Docker verification.

## Author-only items before submission

- [ ] Cover letter to be written by the author
- [ ] Journal submission-account choices to be confirmed by the author
  - double-blind / single-blind selection
  - corresponding-author metadata
  - ORCID / profile metadata if requested
- [ ] Confirm whether any AI-assisted writing or coding disclosure is required under the current Nature portfolio journal policy
