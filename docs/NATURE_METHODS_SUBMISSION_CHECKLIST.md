# Nature Methods submission checklist (GenomeCF)

This checklist is intended to be concrete and auditable. Where possible, each item points to a command and an artifact path.

## 1) Repository and licensing

- [ ] License present: `LICENSE`
- [ ] Citation metadata present: `CITATION.cff`, `codemeta.json`
- [ ] Code availability statement: `docs/CODE_AVAILABILITY.md`
- [ ] Data availability statement: `docs/DATA_AVAILABILITY.md`
- [ ] Model availability statement: `docs/MODEL_AVAILABILITY.md`
- [ ] Benchmark availability statement: `docs/BENCHMARK_AVAILABILITY.md`
- [ ] Environment availability statement: `docs/ENVIRONMENT_AVAILABILITY.md`

## 2) Reproducibility and validation

Run (and record the console logs used for the submission package):

```bash
python -m pip install -e .[benchmark,dev]
python -m pytest

genomecf --help
genomecf reproduce-quickstart

genomecf validate-results
genomecf check-report --results results/release/benchmark_registry.csv
genomecf trace-paper --strict

genomecf build-website --regenerate
genomecf build-paper
```

Artifacts expected:
- Registry: `results/release/benchmark_registry.csv`
- Release tables: `results/release/*.csv` (see manifests)
- Publication tables: `results/publication/*.csv` and `*.tex`
- Traceability report: `results/release/paper_claim_traceability.csv` and `.html`
- Statistical claims index: `results/release/statistical_claims.csv`
- Paper PDFs:
  - `paper/genomecf_report.pdf`
  - `paper/genomecf_supplement.pdf`
- Website:
  - `docs/site/index.html`
  - `docs/site/leaderboard.csv`

## 3) Reporting standard and machine-readable checklist

- [ ] Reporting standard document: `docs/REPORTING_STANDARD.md`
- [ ] Machine-readable checklist: `docs/reporting_checklist.yaml`
- [ ] Checklist schema: `docs/reporting_checklist_schema.json`
- [ ] Example completed checklist: `docs/example_completed_checklist.md`

## 4) Manuscript and supplement

- [ ] Main manuscript PDF exists and is current: `paper/genomecf_report.pdf`
- [ ] Supplement PDF exists and is current: `paper/genomecf_supplement.pdf`
- [ ] All headline quantitative claims are indexed and traceable:
  - Claims index: `paper/claims.yaml`
  - Trace report: `results/release/paper_claim_traceability.csv`
  - Statistical claims: `results/release/statistical_claims.csv`

## 5) Containers (if claimed)

- [ ] Dockerfile present: `Dockerfile`
- [ ] Apptainer/Singularity definition present: `apptainer.def`
- [ ] Documented commands: `docs/DOCKER.md`

If Docker is not verified locally, document CI-only verification and do not claim local verification.

## 6) Submission readiness narrative

- [ ] Release overview: `docs/NATURE_METHODS_RELEASE.md`
- [ ] Reviewer-risk audit: `docs/NATURE_METHODS_REVIEWER_RISK_AUDIT.md`
- [ ] Scope and limitations are explicit: `docs/RELEASE_SCOPE.md`
