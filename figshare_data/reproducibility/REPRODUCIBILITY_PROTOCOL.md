# Reproducibility protocol

This file provides the exact local command path used for the current GenomeCF Nature Methods-style release.

## Editable install

```bash
pip install -e .[benchmark,dev]
```

## Fast verification

```bash
python -m pytest
python -m genomecf.cli validate-results
python -m genomecf.cli check-report --results results/release/benchmark_registry.csv
```

## Reproduction commands

```bash
python -m genomecf.cli reproduce-quickstart
python -m genomecf.cli reproduce-focal
python -m genomecf.cli reproduce-external
python -m genomecf.cli build-website
```

## Script-level artifact regeneration

```bash
python src/generate_release_upgrade_artifacts.py
python src/generate_nature_methods_artifacts.py
```

## Canonical outputs

- release registry: `results/release/benchmark_registry.csv`

- website root: `docs/site/index.html`

## Environment notes

- default Windows CPU environment is sufficient for:
  - tests
  - registry validation
  - website build
  - quickstart reproduction
- the completed Caduceus benchmark path uses WSL2/Linux CUDA as documented in [CADUCEUS_SETUP.md](CADUCEUS_SETUP.md)
- Docker is not installed on the current local Windows verification host, so local container smoke is documented but not executed here
