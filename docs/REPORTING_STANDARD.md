# GenomeCF reporting standard

GenomeCF is meant to function as a practical reporting standard for DNA sequence model papers.

Minimum standard:

1. Report held-out performance and calibration.
2. Report reverse-complement sensitivity.
3. Report mononucleotide and dinucleotide shuffle behavior.
4. Report matched-negative evaluation.
5. Report chromosome- or group-held-out evaluation.
6. Report at least one GenomeCF-Synth shortcut-conflict task.
7. Provide registry-traceable seeds and configs.

Machine-readable checklist:

- `docs/reporting_checklist.yaml`

Validator command:

```bash
genomecf check-report --results results/release/benchmark_registry.csv
```

