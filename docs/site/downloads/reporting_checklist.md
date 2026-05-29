# GenomeCF reporting checklist

GenomeCF recommends that DNA sequence model papers report:

- held-out `AUROC`, `AUPRC`, `ECE`, and `Brier`
- reverse-complement instability and reverse-complement flip rate
- mononucleotide- and dinucleotide-shuffle behavior
- matched-negative evaluation
- chromosome- or group-held-out evaluation
- at least one GenomeCF-Synth shortcut-conflict task
- registry-traceable metadata including seeds and config hashes

Machine-readable version:

- `docs/reporting_checklist.yaml`

Validator:

```bash
genomecf check-report --results results/release/benchmark_registry.csv
```

