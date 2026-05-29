# GenomeCF Supplementary Data and Registry

This package contains the canonical GenomeCF benchmark registry, derived publication tables, external validation summaries, traceability records, reporting-checklist materials and supporting metadata for the manuscript.

- Canonical registry: `results/release/benchmark_registry.csv`
- Main release summary: `results/release/benchmark_summary.csv`
- Publication-derived summary tables: `results/publication/*.csv`
- Traceability: `results/release/paper_claim_traceability.csv` and `results/release/statistical_claims.csv`

How to load the registry:
```python
import pandas as pd
df = pd.read_csv('results/release/benchmark_registry.csv')
```

How to regenerate figures/tables from the repository:
- `python -m genomecf.cli build-paper`
- `python -m genomecf.cli build-supplement`
- `python -m genomecf.cli build-submission-data`

Source data versus derived summaries:
- `source_data/` in the companion source-data archive contains the numerical inputs for the main display items.
- This registry archive contains the canonical registry plus derived summary tables and supporting metadata.

What is not included:
- raw public benchmark datasets
- pretrained model checkpoints
- heavyweight embedding caches

GitHub repository:
https://github.com/AhanBhatt/GenomeCF