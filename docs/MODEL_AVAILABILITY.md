# Model availability

Completed main-paper models:

- `kmer_logistic_regression`
- `small_cnn`
- `small_cnn_rc_aug`
- `dnabert2`
- `caduceus_ph`

Diagnostic or appendix-only coverage:

- `nucleotide_transformer_v2`

Foundation-model notes:

- `dnabert2` runs in the default environment when its local checkpoint cache is available.
- `caduceus_ph` requires the documented WSL2/Linux CUDA route in `docs/CADUCEUS_SETUP.md`.

Local checkpoints are intentionally excluded from GitHub history. Keep them in:

- `../local_runtime_assets/external/`
