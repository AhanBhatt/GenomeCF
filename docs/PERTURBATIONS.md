# Perturbations

Release-facing perturbation manifests are exported to:

- `configs/perturbation_manifests.jsonl`

Current perturbations:

- `reverse_complement`
- `k1_shuffle`
- `k2_shuffle`
- `k3_shuffle`
- `motif_disruption`
- `motif_preserving_flank_shuffle`
- `matched_negative_replacement`

These perturbations are diagnostic stress tests. They do not claim to be perfect biological interventions.

## Interpretation Notes

- Reverse complement probes strand sensitivity and equivariance behavior.
- Mononucleotide and dinucleotide shuffles probe reliance on composition and short-range statistics.
- Motif disruption probes dependence on the intended planted or candidate motif signal.
- Matched-negative replacement probes whether predictions survive after replacing positives with confounder-matched negatives.
