# GenomeCF

GenomeCF is a counterfactual validation standard for DNA sequence models.

It is built around a practical question:

**If a DNA sequence model scores well on a held-out split, is it also stable under counterfactual perturbations, confounder control, split changes, and external biological transfer?**

GenomeCF answers that question with a reusable benchmark, a release registry, a reporting checklist, and a reproducible evaluation workflow.

## What This Repository Contains

This folder is the GitHub-ready repository root.

The manuscript is intentionally **not** stored in this GitHub upload folder. In the local project layout used for manuscript work, the paper lives in the sibling folder:

- `../paper/`

Large local-only runtime assets are also kept outside the public repo in the sibling folder:

- `../local_runtime_assets/`

GenomeCF includes:

- core real-data benchmark tasks
- external biological validation tasks
- MPRA variant-effect tasks
- GenomeCF-Synth shortcut-mechanism tasks
- model manifests, task manifests, split manifests, and perturbation manifests
- a canonical results registry
- a local documentation site and reporting checklist
- reproducible CLI commands for evaluation, summarization, validation, and artifact builds

## Repository Layout

- `package_src/genomecf/`
  - installable Python package
  - CLI
  - benchmark and release helpers
- `configs/`
  - task, model, split, perturbation, and synthetic configs
- `data/`
  - lightweight public task bundles shipped with the repo
  - see `data/README.md` for large local-only benchmark assets
- `docs/`
  - user docs, reproducibility docs, release notes, and static website output
- `envs/`
  - optional environment definitions, including the Caduceus CUDA path
- `external/`
  - lightweight helper assets that are safe to ship publicly
  - local model checkpoints are excluded from Git and live under `../local_runtime_assets/`
- `figures/`
  - generated figures used by the benchmark website and local manuscript workflow
- `release/`
  - release manifest, checksums, reproduction commands, and expected outputs
- `results/`
  - canonical registry, release summaries, validation reports, publication tables, and traceability outputs
  - heavyweight embedding caches are excluded from Git and live under `../local_runtime_assets/`
- `scripts/`
  - reproducibility helpers
- `src/`
  - artifact-generation scripts
- `tests/`
  - fast tests, release tests, and paper/build smoke tests

Key top-level metadata files:

- `pyproject.toml`
- `LICENSE`
- `CITATION.cff`
- `codemeta.json`
- `.zenodo.json`
- `CHANGELOG.md`

## What GenomeCF Measures

GenomeCF is not trying to prove that a model "understands biology." Instead, it measures whether a model behaves consistently under checks that matter for interpretation and transfer.

The current release covers:

- held-out AUROC and related standard metrics
- reverse-complement instability and RC flip behavior
- mononucleotide and dinucleotide shuffle sensitivity
- calibration on original and perturbed inputs
- chromosome-grouped split robustness
- matched-negative evaluation
- GC-bin worst-group robustness
- external biological validation
- MPRA variant-effect reliability
- controlled synthetic shortcut-conflict tasks

## Installation

### Editable install

```bash
pip install -e .
```

### Full local development / benchmark install

```bash
pip install -e .[benchmark,dev]
```

Verify the installation:

```bash
genomecf --help
```

## Environment Options

### CPU-first path

The default local path is enough for:

- quickstart reproduction
- release validation
- registry summaries
- most documentation and website builds
- lightweight smoke tests

### WSL2 / Linux CUDA path

Some larger foundation-model runs, especially `caduceus_ph`, use the documented CUDA route.

See:

- `docs/CADUCEUS_SETUP.md`
- `envs/caduceus.yml`

## Quickstart

The fastest end-to-end check is:

```bash
python -m pip install -e .[benchmark,dev]
genomecf reproduce-quickstart
```

Expected artifact:

- `results/release/quickstart/quickstart_report.json`

## Public Repo vs Local Runtime Assets

The GitHub repo intentionally excludes large local-only files that make version control fragile or impossible to publish cleanly:

- raw per-sequence benchmark text directories for the large core tasks
- local foundation-model checkpoints such as `external/dnabert2_local/`
- embedding caches under `results/cache/`
- temporary runtime outputs under `results/tmp/`

When those assets exist in the sibling folder `../local_runtime_assets/`, GenomeCF will automatically pick them up locally. This keeps the public repository small while preserving the full local workflow on your machine.

## CLI Overview

GenomeCF is designed to be driven primarily through the CLI.

### Inspect help

```bash
genomecf --help
```

### Core benchmark query

```bash
genomecf evaluate \
  --task human_nontata_promoters \
  --model kmer_logistic_regression \
  --split official \
  --mode frozen
```

### External biological validation query

```bash
genomecf external \
  --task gue_human_tf_0 \
  --model dnabert2
```

### Variant-effect summary query

```bash
genomecf variant \
  --task mpra_bcl11a_enhancer \
  --model dnabert2
```

### GenomeCF-Synth summary query

```bash
genomecf synth \
  --task gc_conflict \
  --model caduceus_ph
```

### Release summaries

```bash
genomecf summarize --suite core
genomecf summarize --suite nature_methods
```

### Validation and traceability

```bash
genomecf validate-results
genomecf check-report --results results/release/benchmark_registry.csv
genomecf trace-paper --strict
```

### Reproduction helpers

```bash
genomecf reproduce-quickstart
genomecf reproduce-focal
genomecf reproduce-external
```

### Website build

```bash
genomecf build-website
```

### Optional local manuscript build

If you keep the manuscript outside the repo in the sibling folder `../paper/`, the local code can still build manuscript artifacts against that sibling directory:

```bash
genomecf build-supplement
genomecf build-paper
```

## Typical Workflows

### 1. Confirm the repo works after cloning

```bash
genomecf reproduce-quickstart
genomecf validate-results
```

### 2. Inspect the shipped release

```bash
genomecf summarize --suite nature_methods
```

Then open:

- `results/release/benchmark_summary.csv`
- `results/release/external_validation_summary.csv`
- `results/release/biological_case_study.csv`

### 3. Build the local website

```bash
genomecf build-website
```

Open:

- `docs/site/index.html`

### 4. Optional local manuscript rebuild

```bash
genomecf build-supplement
genomecf build-paper
```

This works when the manuscript companion folder exists at:

- `../paper/`

For the full local benchmark workflow, the optional runtime assets can also live at:

- `../local_runtime_assets/`

## Included Task Families

### Core real-data tasks

- `human_nontata_promoters`
- `human_enhancers_cohn`
- `human_enhancers_ensembl`
- `human_ocr_ensembl`

### Screening tasks

- `dummy_mouse_enhancers_ensembl`
- `drosophila_enhancers_stark`

### External biological validation

- TF-binding tasks
- histone-mark tasks
- MPRA variant-effect tasks

### GenomeCF-Synth

- `gc_correlated`
- `gc_matched`
- `gc_conflict`
- `two_motif_grammar`
- `motif_position_conflict`

## Included Models

Main models:

- `kmer_logistic_regression`
- `small_cnn`
- `small_cnn_rc_aug`
- `dnabert2`
- `caduceus_ph`

Diagnostic baselines:

- `gc_only`
- `cpg_only`
- `repeat_only`
- `length_only`

Appendix-only diagnostic foundation baseline:

- `nucleotide_transformer_v2`

## Canonical Outputs

Main release artifacts:

- registry CSV: `results/release/benchmark_registry.csv`
- registry JSONL: `results/release/benchmark_registry.jsonl`
- release summary: `results/release/benchmark_summary.csv`
- model-task matrix: `results/release/model_task_matrix.csv`
- validation report: `results/release/validation_report.json`
- paper-claim traceability:
  - `results/release/paper_claim_traceability.csv`
  - `results/release/paper_claim_traceability.html`

Publication tables and figure inputs:

- `results/publication/`

Release-bundle files:

- `release/GenomeCF_v1_manifest.json`
- `release/GenomeCF_v1_checksums.txt`
- `release/GenomeCF_v1_reproduction_commands.sh`
- `release/GenomeCF_v1_expected_outputs.md`

## Manuscript Companion Folder

The manuscript is kept outside this GitHub repo folder in the sibling project directory:

- `../paper/`

That companion folder contains:

- `genomecf_report.tex`
- `genomecf_report.pdf`
- `genomecf_supplement.tex`
- `genomecf_supplement.pdf`
- `refs.bib`
- `claims.yaml`

This separation keeps the public GitHub upload focused on the software/resource release while leaving the paper bundle in the private parent project folder.

## Documentation Map

Start here:

- `docs/QUICKSTART.md`
- `docs/PROTOCOL.md`
- `docs/REPRODUCIBILITY_PROTOCOL.md`

Benchmark and methods docs:

- `docs/BENCHMARK.md`
- `docs/TASKS.md`
- `docs/MODELS.md`
- `docs/METRICS.md`
- `docs/SPLITS.md`
- `docs/RESULT_SCHEMA.md`
- `docs/REPORTING_STANDARD.md`
- `docs/EXTERNAL_VALIDATION.md`
- `docs/BIOLOGICAL_CASE_STUDY.md`
- `docs/SYNTHETIC_TASKS.md`
- `docs/MOTIF_ANALYSIS.md`
- `docs/GC_BIN_ROBUSTNESS.md`

Availability and release docs:

- `docs/NATURE_METHODS_RELEASE.md`
- `docs/CODE_AVAILABILITY.md`
- `docs/DATA_AVAILABILITY.md`
- `docs/BENCHMARK_AVAILABILITY.md`
- `docs/MODEL_AVAILABILITY.md`
- `docs/ENVIRONMENT_AVAILABILITY.md`
- `docs/ARTIFACT_EVALUATION.md`

## GitHub Upload Notes

This folder is the version intended for GitHub.

Before pushing:

1. run `genomecf validate-results`
2. run `python -m pytest`
3. inspect `docs/RELEASE_SCOPE.md`
4. inspect `release/GenomeCF_v1_manifest.json`
5. confirm that the private manuscript companion remains outside this repo in `../paper/`
6. initialize Git in this folder if you want a fresh repository:

```bash
git init
git add .
git commit -m "Initial GenomeCF release"
```

## Known Local Limitations

- Docker definitions are included, but Docker cannot be smoke-tested locally on a machine where Docker is not installed.
- The heaviest Caduceus runs use the documented WSL2/Linux CUDA path rather than the default CPU path.
- Some supplement tables can still trigger non-fatal LaTeX overfull or underfull warnings during builds.

## Release Scope

For the public release-facing description of what is included in this repo, read:

- `docs/RELEASE_SCOPE.md`

Internal upgrade logs, private manuscript planning notes, and submission-management files are intentionally kept outside the public repo.

## Citation and License

- citation metadata: `CITATION.cff`
- machine-readable metadata: `codemeta.json`, `.zenodo.json`
- license: `LICENSE`
