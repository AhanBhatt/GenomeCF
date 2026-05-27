# GenomeCF Benchmark

GenomeCF is organized as a tiered benchmark release.

## Tiers

- `core`
  - main real-data tasks with the clearest model coverage and publication relevance
- `screening`
  - extra tasks with lighter or partial coverage
- `synthetic`
  - planted-rule tasks that isolate shortcut behavior
- `long_context_pending`
  - scaffolded future tasks for long-context evaluation

## Core Scientific Question

Does strong held-out performance imply counterfactual biological consistency?

GenomeCF answers this by pairing standard predictive metrics with:

- reverse-complement stress tests
- mononucleotide and dinucleotide shuffle tests
- calibration analysis
- stronger split protocols
- synthetic ground-truth tasks

## Current Core Tasks

- `human_nontata_promoters`
- `human_enhancers_cohn`
- `human_enhancers_ensembl`
- `human_ocr_ensembl`

## Current Screening Tasks

- `dummy_mouse_enhancers_ensembl`
- `drosophila_enhancers_stark`

## Current Synthetic Tasks

- `gc_correlated`
- `gc_matched`

## Main Model Families

- diagnostic baselines
- classical sequence baselines
- CNN baselines
- foundation-model baselines where results are available

The release keeps partial coverage explicit. Scaffolded models are documented, but they are not treated as completed paper results unless real outputs exist.
