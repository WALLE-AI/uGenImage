#!/bin/bash
# 阶段 1：训练 VQGAN tokenizer
# 所有超参在 configs/vqgan.yaml；本脚本只负责选卡与进程数。
#   NGPU=4 bash scripts/train_vqgan.sh
#   bash scripts/train_vqgan.sh --set train.lr=5e-5 data.batch_size=64
set -e
cd "$(dirname "$0")/.."

NGPU=${NGPU:-1}
CONFIG=${CONFIG:-configs/vqgan.yaml}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [ "$NGPU" -gt 1 ]; then
    torchrun --nproc_per_node="$NGPU" --master_port="${MASTER_PORT:-29501}" \
        train_vqgan.py --config "$CONFIG" "$@"
else
    python train_vqgan.py --config "$CONFIG" "$@"
fi
