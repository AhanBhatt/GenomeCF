#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(pwd)}"
SEED="${GENOMECF_SEED:-2026}"
export GENOMECF_HF_BATCH_SIZE="${GENOMECF_HF_BATCH_SIZE:-4}"
export GENOMECF_CACHE_ROOT="${GENOMECF_CACHE_ROOT:-$PROJECT_ROOT/results/cache/embeddings}"

TASKS=(
  human_nontata_promoters
  human_enhancers_cohn
)
MODELS=(
  dnabert2
  caduceus_ph
)

cd "$PROJECT_ROOT"
for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "[GenomeCF] Foundation mitigation :: model=$model task=$task calibration=temperature split=official"
    python -m genomecf.cli evaluate \
      --task "$task" \
      --model "$model" \
      --split official \
      --mode frozen \
      --seed "$SEED" \
      --calibration temperature

    echo "[GenomeCF] Foundation mitigation :: model=$model task=$task calibration=temperature split=matched_test"
    python -m genomecf.cli evaluate \
      --task "$task" \
      --model "$model" \
      --split matched_test \
      --mode frozen \
      --seed "$SEED" \
      --calibration temperature

    echo "[GenomeCF] Foundation mitigation :: model=$model task=$task matched_retraining split=official"
    python -m genomecf.cli evaluate \
      --task "$task" \
      --model "$model" \
      --split official \
      --mode frozen \
      --seed "$SEED" \
      --matched-retraining

    echo "[GenomeCF] Foundation mitigation :: model=$model task=$task matched_retraining split=matched_test"
    python -m genomecf.cli evaluate \
      --task "$task" \
      --model "$model" \
      --split matched_test \
      --mode frozen \
      --seed "$SEED" \
      --matched-retraining
  done
done
