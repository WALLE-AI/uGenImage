#!/bin/bash
# 端到端连通性检查：合成小图跑完 训VQGAN -> 预编码 -> 训Transformer -> 采样 四步。
# 目的是验证管线可执行（含配置系统、断点续训、验证、采样落盘），不是验证画质。
set -e
cd "$(dirname "$0")/.."

PY=${PY:-python}
RUN=${RUN:-/tmp/ugenimage_e2e}
DEVICE=${DEVICE:-cpu}
export UGEN_CACHE="$RUN/cache"   # 文件清单缓存隔离，避免污染 ~/.cache

rm -rf "$RUN"
mkdir -p "$RUN/images"

$PY - "$RUN/images" <<'EOF'
import sys
import numpy as np
from PIL import Image
rng = np.random.default_rng(0)
for i in range(64):
    a = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    Image.fromarray(a).save(f"{sys.argv[1]}/img_{i:03d}.png")
EOF

COMMON_V="--config configs/vqgan.yaml --set
  data.image_dir=$RUN/images data.image_size=64 data.batch_size=4 data.num_workers=0
  data.val_size=8 model.codebook_size=64 model.embedding_dim=32
  train.run_dir=$RUN/vqgan train.max_steps=6 train.warmup_steps=2
  train.log_every=2 train.eval_every=6 train.eval_batches=2 train.save_every=3"

echo "=== 1/4 训练 VQGAN ==="
CUDA_VISIBLE_DEVICES="" $PY train_vqgan.py $COMMON_V

echo "=== 1b. 断点续训检查（应从 step=6 接上并跑到 10）==="
CUDA_VISIBLE_DEVICES="" $PY train_vqgan.py $COMMON_V --set train.max_steps=10 > "$RUN/resume.log" 2>&1
grep -q "恢复，step=6" "$RUN/resume.log" && echo "  resume OK" || { cat "$RUN/resume.log"; exit 1; }
grep -q "step 10" "$RUN/resume.log" && echo "  续训到 step=10 OK" || { cat "$RUN/resume.log"; exit 1; }

echo "=== 2/4 预编码 ==="
CUDA_VISIBLE_DEVICES="" $PY preencode.py --vqgan_ckpt "$RUN/vqgan/ckpt/final.pt" \
    --image_dir "$RUN/images" --output_dir "$RUN/tokens" --batch_size 4 --device "$DEVICE"

echo "=== 3/4 训练 Transformer ==="
CUDA_VISIBLE_DEVICES="" $PY train_transformer.py --config configs/transformer.yaml --set \
    data.token_dir="$RUN/tokens" data.batch_size=4 data.num_workers=0 data.val_size=8 \
    model.dim=64 model.n_layers=2 model.n_heads=4 model.n_kv_heads=2 model.head_dim=16 \
    model.vocab_size=67 model.seq_len=17 \
    train.run_dir="$RUN/transformer" train.max_steps=6 train.warmup_steps=2 \
    train.log_every=2 train.eval_every=6 train.eval_batches=2 train.save_every=3

echo "=== 4/4 采样 ==="
CUDA_VISIBLE_DEVICES="" $PY generate.py \
    --transformer_ckpt "$RUN/transformer/ckpt/final.pt" \
    --vqgan_ckpt "$RUN/vqgan/ckpt/final.pt" \
    --output "$RUN/sample.png" --n_samples 2 --top_k 16 --device "$DEVICE"

echo "=== 产物 ==="
ls -1 "$RUN/vqgan/samples" | head -3
test -f "$RUN/vqgan/log.txt" && echo "  日志 OK"
test -f "$RUN/vqgan/metrics.jsonl" && echo "  metrics OK"
test -f "$RUN/sample.png" && echo "  采样图 OK"
echo "=== 全部通过 ==="
