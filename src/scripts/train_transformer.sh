#!/bin/bash
# 阶段 3：训练 Transformer prior
# 所有超参在 configs/transformer.yaml；本脚本只负责选卡与进程数。
#   NGPU=4 bash scripts/train_transformer.sh
#   bash scripts/train_transformer.sh --set train.max_steps=1000 data.limit=5000
set -e
cd "$(dirname "$0")/.."

NGPU=${NGPU:-1}
CONFIG=${CONFIG:-configs/transformer.yaml}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [ "$NGPU" -gt 1 ]; then
    torchrun --nproc_per_node="$NGPU" --master_port="${MASTER_PORT:-29502}" \
        train_transformer.py --config "$CONFIG" "$@"
else
    python train_transformer.py --config "$CONFIG" "$@"
fi
