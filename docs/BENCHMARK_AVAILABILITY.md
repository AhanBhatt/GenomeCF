# Benchmark availability

GenomeCF is distributed as a local benchmark package and release artifact bundle.

Primary release paths:

- registry: `results/release/benchmark_registry.csv`
- summary: `results/release/benchmark_summary.csv`
- website: `docs/site/`

Large local-only runtime assets such as raw benchmark text folders, checkpoint caches, and embedding caches can live in:

- `../local_runtime_assets/`

Primary commands:

```bash
genomecf validate-results
genomecf build-website
```
