# Nature Methods reviewer-risk audit (GenomeCF)

This document anticipates likely reviewer objections for a benchmark and software resource submission and maps each objection to (i) why it matters, (ii) evidence in the paper and artifacts, (iii) the fix implemented in the release, and (iv) any remaining caveat.

## How to verify evidence

- Canonical registry: `results/release/benchmark_registry.csv`
- Release validation: `genomecf validate-results`
- Reporting standard check: `genomecf check-report --results results/release/benchmark_registry.csv`
- Paper claim traceability: `genomecf trace-paper --strict`
  - Outputs: `results/release/paper_claim_traceability.csv` and `.html`
- External transfer statistics: `results/release/external_transfer_stats.json`
- Statistical claims index: `results/release/statistical_claims.csv`
- Paper PDFs:
  - `paper/genomecf_report.pdf`
  - `paper/genomecf_supplement.pdf`

## Risk register

### R1. “The paper depends on a single Shortcut Score, which could be brittle or misleading.”

**Why it matters:** Reviewers will reject a benchmark if its central summary metric collapses important failure modes.

**Evidence:**
- External transfer analysis explicitly compares AUROC-only, Shortcut Score-only, and a multimetric GenomeCF profile.
  - `results/release/external_transfer_stats.json`
  - Claims indexed in `results/release/statistical_claims.csv`

**Fix implemented:**
- The manuscript framing treats the Shortcut Score as a useful diagnostic summary, but not a standalone predictor of external reliability.
- The primary predictive object for external biological reliability is the full GenomeCF profile (calibration, RC stability, matched-negative behavior, GC-bin robustness, and shortcut metrics).

**Remaining caveat:**
- Any single scalar summary can be misused if reported without component metrics. GenomeCF’s reporting standard requires component reporting.

### R2. “Cross-family generalization (leave-one-family-out) is weak, so the external prediction claim may not hold.”

**Why it matters:** A general-purpose “reliability predictor” claim is vulnerable if it does not extrapolate across assay families.

**Evidence:**
- Leave-one-family-out results in `results/release/external_transfer_stats.json`.
- Family-stratified regressions reported in the same file.

**Fix implemented:**
- The paper frames LOFO as a hard problem because TF binding, histone marks, and MPRA variant effects define different reliability axes.
- The primary claim is that the GenomeCF profile improves prediction within and across mixed assay settings relative to AUROC-only, and exposes assay-specific reliability axes that AUROC misses.

**Remaining caveat:**
- Assay-agnostic extrapolation remains difficult and is explicitly stated as a limitation.

### R3. “The MYC MPRA case study overclaims; DNABERT-2 is not globally better than simpler baselines.”

**Why it matters:** Case studies are a reviewer focal point and must match the decision objective.

**Evidence:**
- MPRA case-study artifacts: `results/release/biological_case_study.csv`.
- Case-study figure: `figures/genomecf_biological_case_study.png`.

**Fix implemented:**
- The MYC claim is objective-specific: DNABERT-2 is preferred for top-k nomination (enrichment/precision), while 6-mer remains stronger on global discrimination summaries.

**Remaining caveat:**
- Case-study conclusions are not generalized beyond the stated objective.

### R4. “Foundation-model conclusions are confounded by adaptation choices (frozen vs head-only vs fine-tuned).”

**Why it matters:** Reviewers will question whether performance differences are due to training protocol rather than model class.

**Evidence:**
- Release documentation for foundation-model setup and constraints: `docs/FOUNDATION_MODELS.md` and `docs/CADUCEUS_SETUP.md`.

**Fix implemented:**
- The release documents adaptation scope for each model configuration and avoids claiming full fine-tuning unless it was performed and recorded in the registry.

**Remaining caveat:**
- The current release focuses on frozen or head-adapted variants for reproducibility and cost; full fine-tuning is a future expansion.

### R5. “Synthetic tasks (GenomeCF-Synth) are artificial and may not translate to real genomics.”

**Why it matters:** Synthetic benchmarks can be dismissed if not connected to real failure modes.

**Evidence:**
- Synthetic results summary: `results/release/synthetic_extended_summary.csv`.
- Synthetic documentation: `docs/SYNTHETIC_TASKS.md`.

**Fix implemented:**
- The paper positions synthetic tasks as stress tests that isolate shortcut following (e.g., GC-conflict) rather than as biological ground truth.

**Remaining caveat:**
- Synthetic tasks complement, not replace, external biological validation.

### R6. “Negative sampling and matched-negative evaluation could drive the conclusions.”

**Why it matters:** Benchmark claims can hinge on evaluation protocol choices.

**Evidence:**
- Matched-negative tables and docs: `docs/PERTURBATIONS.md`, `docs/METRICS.md`, and publication tables under `results/publication/`.

**Fix implemented:**
- The release documents matched-negative construction and reports both official and matched-negative metrics.

**Remaining caveat:**
- Matched-negative design remains a choice; GenomeCF makes the choice explicit and reportable.

### R7. “Reproducibility and container claims are unverified.”

**Why it matters:** Nature Methods software reviewers will check whether the release can be executed.

**Evidence:**
- CLI quickstart: `genomecf reproduce-quickstart`.
- Container docs: `docs/DOCKER.md` and `apptainer.def`.

**Fix implemented:**
- The repo distinguishes locally verified commands from CI-only validations.

**Remaining caveat:**
- If Docker is not available locally, Docker verification must be performed in CI and documented as such.

### R8. “The main paper is too table-heavy or visually report-like.”

**Why it matters:** Resource papers are judged on clarity, narrative, and figure quality.

**Evidence:**
- Current paper PDFs: `paper/genomecf_report.pdf`, `paper/genomecf_supplement.pdf`.

**Fix implemented:**
- Main tables are kept compact; complete matrices are placed in the supplement.

**Remaining caveat:**
- Figure redesign is an iterative polish task and should be evaluated directly in the PDFs.
