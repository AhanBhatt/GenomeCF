# Tasks

GenomeCF currently separates tasks into benchmark tiers instead of pretending every dataset has identical model coverage.

## Core real-data tasks

- `human_nontata_promoters`
- `human_enhancers_cohn`
- `human_enhancers_ensembl`
- `human_ocr_ensembl`

These four human tasks are the main real-data benchmark. They now have completed official-split rows for:

- GC-only
- CpG-only
- repeat-only
- 6-mer logistic regression
- CNN
- RC-aug CNN
- DNABERT-2 frozen embeddings
- Caduceus-Ph frozen embeddings

The variable-length tasks also have:

- `length_only` on `human_enhancers_ensembl`
- `length_only` on `human_ocr_ensembl`

## Screening tasks

- `dummy_mouse_enhancers_ensembl`
- `drosophila_enhancers_stark`

These are kept separate from the main claims because model coverage is lighter.

## Synthetic tasks

- `gc_correlated`
- `gc_matched`

These synthetic planted-motif tasks are used to isolate mechanism under known ground truth.

## Scaffolded or pending tasks

- `ssp_reconstructed`
- `wgEncodeEH000552`
- `wgEncodeEH000606`
- `wgEncodeEH001546`
- `wgEncodeEH001776`
- `wgEncodeEH002829`
- `tad_boundary_recognition`

The manifests exist, but they are not part of the completed benchmark release unless local data and completed result files are present.

## Task metadata sources

- Internal manifests: `genomecf/config/tasks/*.json`
- Exported manifests: `configs/task_manifests.jsonl`
- Public release scope: `docs/RELEASE_SCOPE.md`
- Canonical release registry: `results/release/benchmark_registry.csv`
