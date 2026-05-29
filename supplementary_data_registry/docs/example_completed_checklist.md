# Example completed GenomeCF reporting checklist

This example illustrates what a completed checklist looks like for a GenomeCF-style result bundle.

## Checklist summary

- held-out metrics present: yes
- reverse-complement metrics present: yes
- mononucleotide shuffle metrics present: yes
- dinucleotide shuffle metrics present: yes
- matched-negative evaluation present: yes
- chromosome/group-held-out split present: yes
- at least one GenomeCF-Synth conflict task present: yes
- registry traceability fields present: yes

## Example validation command

```bash
genomecf check-report --results results/release/benchmark_registry.csv --checklist docs/reporting_checklist.yaml
```

## Expected output

- machine-readable JSON report:
  - `results/release/reporting_check_report.json`
- overall status:
  - `passed: true`

## Example fields expected in the result table

- `task_id`
- `model_id`
- `split_id`
- `auroc`
- `auprc`
- `ece`
- `brier`
- `mean_abs_delta`
- `flip_rate`
- `positive_prob_drop`
- `created_at`
- `config_hash`
- `data_hash`
