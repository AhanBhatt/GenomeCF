#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(pwd)}"
SEED="${GENOMECF_SEED:-2026}"
export GENOMECF_HF_BATCH_SIZE="${GENOMECF_HF_BATCH_SIZE:-4}"
export GENOMECF_CACHE_ROOT="${GENOMECF_CACHE_ROOT:-$PROJECT_ROOT/results/cache/embeddings}"

TASKS=(
  human_nontata_promoters
  human_enhancers_cohn
  human_enhancers_ensembl
  human_ocr_ensembl
)
FOLDS=(A B C D E)

cd "$PROJECT_ROOT"
for task in "${TASKS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    if [[ "$task" == "human_nontata_promoters" && "$fold" == "A" ]]; then
      continue
    fi
    echo "[GenomeCF] Caduceus chromosome CV :: task=$task fold=$fold seed=$SEED batch_size=$GENOMECF_HF_BATCH_SIZE"
    python -m genomecf.cli evaluate \
      --task "$task" \
      --model caduceus_ph \
      --split chromosome_5fold_cv \
      --fold "$fold" \
      --mode frozen \
      --seed "$SEED"
  done
done
