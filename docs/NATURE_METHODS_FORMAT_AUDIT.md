# Nature Methods format audit (GenomeCF)

## Scope

- Target content type: `Resource`
- This audit covers the private manuscript build in `paper/`
- Cover letter is intentionally excluded from this task and remains author-managed

## Word counts and display items

- Abstract word count: `149`
- Main-text word count estimate (Introduction + Results + Discussion; excluding abstract, Online Methods, references and figure legends): `2,969`
- Main display-item count: `6`

Main display items:
1. Fig. 1. GenomeCF resource overview
2. Fig. 2. Held-out AUROC versus GenomeCF reliability axes
3. Fig. 3. Foundation-model comparison and adaptation
4. Fig. 4. External validation and full GenomeCF profile prediction
5. Fig. 5. MPRA biological case studies
6. Fig. 6. GenomeCF-Synth shortcut and rule-learning results

## Supplementary item list

Supplementary tables are numbered and cited in the same order they appear:

1. Supplementary Table S1. Real-data task overview
2. Supplementary Table S2. GenomeCF minimum reporting standard
3. Supplementary Table S3. Official-split real-data matrix
4. Supplementary Table S4. Five-fold chromosome-grouped CV summary
5. Supplementary Table S5. Per-fold chromosome-grouped CV metrics
6. Supplementary Table S6. Matched-negative evaluation
7. Supplementary Table S7. Matched-negative confounder balance
8. Supplementary Table S8. Mitigation experiments
9. Supplementary Table S9. GC-bin robustness summary
10. Supplementary Table S10. Full GC-bin metrics
11. Supplementary Table S11. External biological validation matrix
12. Supplementary Table S12. External transfer-prediction summary
13. Supplementary Table S13. External prediction robustness
14. Supplementary Table S14. MPRA case-study results
15. Supplementary Table S15. External GC-bin summary
16. Supplementary Table S16. External GC-bin by-bin metrics
17. Supplementary Table S17. Real-task motif probes
18. Supplementary Table S18. Original synthetic benchmark summary
19. Supplementary Table S19. Extended GenomeCF-Synth results
20. Supplementary Table S20. Shortcut Score summary
21. Supplementary Table S21. Per-task Shortcut Score components
22. Supplementary Table S22. Confounder summary
23. Supplementary Table S23. Chromosome-fold counts
24. Supplementary Table S24. Statistical support for headline claims
25. Supplementary Table S25. Availability and registry traceability

Status:
- Supplementary items cited in order: `Yes`
- Every supplementary item cited at least once: `Yes`

## Structure audit

- Introduction heading removed: `Yes`
- Results section present: `Yes`
- Discussion present with no subheadings: `Yes`
- Online Methods present with short subheadings: `Yes`
- Availability sections present: `Yes`
  - Data Availability
  - Code Availability
  - Benchmark Availability
  - Model Availability
  - Environment Availability
  - Reproducibility Statement
- References precede the Supplementary Information statement: `Yes`

## Citation and reference status

- All references resolved in the main PDF: `Yes`
- All references resolved in the supplement PDF: `Yes`
- Main figures cited sequentially: `Yes`
- No unresolved `Figure ??`, `Table ??`, `Section ??`, `Reference ??`, `Supplementary Table ??`, or question-mark citation placeholders in the extracted PDF text: `Yes`
- Current submission-ready reference style: `author-year (plainnat)`
- Optional numeric reference path documented: `Yes`
  - Copy `paper/reference_style.numeric.example.tex` to `paper/reference_style.tex` to switch the build to `unsrtnat` numeric citations without editing the manuscript body

## Figure source-data status

Main-figure source data are generated in `source_data/`:

- `source_data/Fig1_source_data.json`
- `source_data/Fig2_source_data.csv`
- `source_data/Fig3_source_data.csv`
- `source_data/Fig4_source_data.csv`
- `source_data/Fig5_source_data.csv`
- `source_data/Fig6_source_data.csv`
- `source_data/manifest.json`

Status: `Generated and referenced in the availability text`

## Availability and reproducibility status

- Benchmark registry present: `results/release/benchmark_registry.csv`
- Claim traceability present:
  - `results/release/paper_claim_traceability.csv`
  - `results/release/paper_claim_traceability.html`
- Statistical support present: `results/release/statistical_claims.csv`
- Website build output present: `docs/site/index.html`
- Quickstart and release docs present: `Yes`
- Docker locally verified: `No`
- Docker CI path configured: `Yes`

## What remains before submission

Only author-controlled tasks remain:

- write the cover letter
- confirm submission-account options such as corresponding-author metadata and any ORCID requests
- confirm whether any AI-assisted writing or coding disclosure is required under the current Nature journal policy
