# GenomeCF release status

This file records what has been verified *in this repo* (commands run locally, artifacts present), and what remains.

## Verdict

- **Reproducibility and traceability:** Ready (validated locally).
- **Manuscript/visual polish for a Nature Methods submission:** Not yet fully complete (see Remaining gaps).

## Verified locally (Windows)

The following commands were run successfully in `Project\GenomeCF`:

```bash
python -m pip install -e .[benchmark,dev]

genomecf --help
python -m pytest

python -m genomecf.cli validate-results
python -m genomecf.cli check-report --results results/release/benchmark_registry.csv

python -m genomecf.cli reproduce-quickstart
python -m genomecf.cli build-website
```

Notes:
- The manuscript build is intentionally not part of the public GitHub repo and is not included in these verification commands.

## Key artifacts present
- Registry: `results/release/benchmark_registry.csv`
- Release bundle:
  - `release/GenomeCF_v1_manifest.json`
  - `release/GenomeCF_v1_checksums.txt`
  - `release/GenomeCF_v1_reproduction_commands.sh`
  - `release/GenomeCF_v1_expected_outputs.md`
- External robustness table: `results/release/external_prediction_robustness.csv`
- Website: `docs/site/index.html` (and `docs/site/leaderboard{,_rows}.csv`)

## CI / containers

- GitHub Actions workflows exist under `.github/workflows/`:
  - `ci.yml` (tests + CLI validation)
  - `docker.yml` (Docker build + quickstart smoke)
  - `docs.yml` (site regeneration)

Local Docker and Apptainer builds were **not** re-validated in this status run.

## Remaining gaps (Nature Methods polish)

- Final figure redesign to Nature Methods visual quality (layout, typography, consistent palettes; reduce table-heaviness in main text).
- Website/leaderboard can be extended further (additional metrics/columns, richer narrative pages), although it is now non-placeholder and filterable.
- Optional deeper external-prediction robustness analyses beyond the current task/family stratifications (if targeting a dedicated robustness appendix).
