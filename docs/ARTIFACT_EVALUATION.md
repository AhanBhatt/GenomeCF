# Artifact evaluation

GenomeCF is intended to function as a reusable benchmark resource, not only as a paper.

## Core artifact classes

- registry artifacts:
  - `results/release/benchmark_registry.csv`
  - `results/release/benchmark_summary.csv`
- validation artifacts:
  - `results/release/validation_report.json`
  - `results/release/reporting_check_report.json`
- resource artifacts:
  - `docs/site/index.html`
  - `docs/reporting_checklist.yaml`
  - `docs/reporting_checklist.md`

## What an artifact evaluator should verify

1. `pip install -e .[benchmark,dev]` succeeds.
2. `python -m pytest` passes in the default environment.
3. `python -m genomecf.cli validate-results` passes.
4. `python -m genomecf.cli check-report --results results/release/benchmark_registry.csv` passes.
5. `python -m genomecf.cli build-website` creates a local site with a leaderboard.

## Known environment caveat

- `caduceus_ph` completion depends on the documented WSL2/Linux CUDA setup.
- The default Windows CPU environment remains the main verification path for fast tests and website builds.
