# Task Manifests

Release-facing task manifests are exported to:

- `configs/task_manifests.jsonl`

Each manifest records:

- task ID and readable name
- benchmark tier
- species
- source and dataset provenance
- sequence length range
- class-construction notes
- metadata availability
- recommended split protocols
- allowed perturbations
- current observed model coverage

Synthetic tasks are included alongside real tasks so the release has a single manifest catalog.
