#!/bin/bash
# 阶段 2：用训练好的 VQGAN 将图片预编码为 token 序列 (.npy)
# 模型结构自动从 checkpoint 的 config 读取，无需与训练超参手工对齐。
set -e
cd "$(dirname "$0")/.."

CKPT=${CKPT:-runs/vqgan/ckpt/final.pt}
IMAGE_DIR=${IMAGE_DIR:-/home/dataset0/images/ALLaVA-4V/allava_laion/image_chunks/images}
OUT_DIR=${OUT_DIR:-data/tokens_train}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python preencode.py \
    --vqgan_ckpt "$CKPT" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUT_DIR" \
    --batch_size "${BATCH_SIZE:-32}" "$@"
