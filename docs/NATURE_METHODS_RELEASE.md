# GenomeCF Nature Methods release

GenomeCF is released as a counterfactual validation resource for DNA sequence models.

Core release artifacts:

- main paper: `../paper/genomecf_report.pdf`
- supplement: `../paper/genomecf_supplement.pdf`
- canonical registry: `results/release/benchmark_registry.csv`
- publication tables: `results/publication/*.csv`
- local site: `docs/site/index.html`

Key commands:

```bash
genomecf build-paper
genomecf build-appendix
genomecf validate-results
genomecf build-website
genomecf reproduce-quickstart
genomecf check-report --results results/release/benchmark_registry.csv
```
