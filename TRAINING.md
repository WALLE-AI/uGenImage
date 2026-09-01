# 训练技术文档

> 适用版本：方案 A（VQGAN tokenizer + 20 层 LLaMA 式自回归 prior）
> 最后更新：2026-09-01
> 相关文档：`SCHEME_A_BASELINE.md`（设计规格）· `SCHEME_B_OPTIMIZED.md`（改进方向）· `PLAN.md`（工程路线）
>
> 本文档中的所有性能数字均为**本机实测**，非估算。硬件：A100-SXM4-40GB × 8（共享，通常只有 1–2 张可用）。

---

## 1. 环境

### 1.1 依赖

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv torch torchvision pillow numpy tqdm pyyaml pytest
```

> ⚠️ **已知问题**：本机代理带宽约 80 KB/s，torch 的 CUDA 依赖约 2GB，两次尝试均因超时失败。
> 更麻烦的是 **uv 失败时退出码仍是 0**（日志里是 `broken pipe` / `network timeout`），
> 会被误判成安装成功。安装后务必显式验证：
>
> ```bash
> .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
> ```
>
> 当前所有训练均使用系统 `anaconda3` 的 torch 2.7.0+cu126 运行，功能无差异。

### 1.2 运行方式

所有命令在 `src/` 目录下执行。若未安装为包，需要 `PYTHONPATH=.`：

```bash
cd src
PYTHONPATH=. python train_vqgan.py --config configs/vqgan.yaml
```

### 1.3 自检

```bash
pytest                        # 18 个 CPU 冒烟用例，约 3 秒
bash scripts/e2e_check.sh     # 合成数据跑完四步 + 断点续训验证，约 1 分钟
```

**开训前务必先跑这两个**。它们覆盖了 RoPE 是否生效、掩码分支是否真双向、
采样 token 数是否正确、配置覆盖是否被静默忽略等一批不会报错但会让训练白跑的问题。

---

## 2. 数据

### 2.1 本机可用语料

| 路径 | 数量 | 规格 | 适用 |
|---|---|---|---|
| `/home/dataset0/images/ALLaVA-4V/allava_laion/image_chunks/images` | **481,613 张 .jpeg** | 全 RGB，中位 1200×800 | ✅ 256px 训练（**默认**） |
| `/home/dataset0/images/minimind-v_dataset/pretrain_images` | 595,375 张 .jpg | 全部 128×128 | ⚠️ 仅适合 128px |

扁平目录 + 标准扩展名，与 `ImageDataset` / `preencode.py` 直接兼容，**无需复制、无需下载**。

### 2.2 关于下载

本机网络实测：

| 项 | 实测 |
|---|---|
| 直连（不走代理） | 完全不通 |
| 代理单连接 | 26–52 KB/s |
| 代理 4 并发合计 | ~85 KB/s |
| huggingface.co | 不可达；用 `export HF_ENDPOINT=https://hf-mirror.com` |

按 80 KB/s 估算：Flowers-102(330MB) 约 1–2 小时；CelebA-HQ(3GB) 约 10 小时；
COCO train2017(19GB) 约 3 天；ImageNet(150GB) 不可行（且磁盘不足）。
**优先用本机已有语料。**

### 2.3 磁盘

| 挂载点 | 可用 | 建议 |
|---|---|---|
| `/home/dataset1` | ~123G（已用 96%） | 谨慎，大 checkpoint 不要放这里 |
| `/` | ~314G（已用 25%） | **推荐放 run_dir 和 token** |

---

## 3. 四阶段流程

```
图片 ──[1 train_vqgan]──> VQGAN ──[2 preencode]──> token .npy
                                                      │
                              ┌───────[3 train_transformer]───────┘
                              ▼
                        prior ──[4 generate]──> 图片
```

### 阶段 1 · 训练 VQGAN

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python train_vqgan.py \
  --config configs/vqgan.yaml --set \
  data.batch_size=64 data.num_workers=24 \
  train.run_dir=runs/vqgan_full train.max_steps=40000 train.warmup_steps=1000

# 多卡
NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_vqgan.sh
```

**实测性能**（A100 单卡，batch 64，256px）：

| 项 | 实测值 |
|---|---|
| 吞吐 | **855 img/s** |
| 显存 | 16.2 GB |
| 4 万步耗时 | **53 分钟**（0.88h，= 5.3 个 epoch over 48 万张） |
| 模型参数量 | 19.7M |
| 单 checkpoint | 0.24 GB（全量，含 optimizer） |

### 阶段 2 · 预编码

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python preencode.py \
  --vqgan_ckpt runs/vqgan_full/ckpt/final.pt \
  --image_dir /home/dataset0/images/ALLaVA-4V/allava_laion/image_chunks/images \
  --output_dir data/tokens_full --batch_size 128 --num_workers 24 --limit 120000
```

**实测性能**：**1,650 img/s**（12 万张 72 秒）。

> 模型结构自动从 checkpoint 的 config 读取，不需要手工对齐超参。
> 默认跳过已存在的输出，可断点续跑；加 `--overwrite` 强制重编。
>
> 📌 原实现在主进程串行解码 JPEG，只有 70 img/s，全量 48 万张需近 2 小时。
> 已改为 DataLoader 多进程，提速 23 倍。

产物：每张图一个 `.npy`，257 个 int32 = `BOS + 16×16`。12 万张约 473MB。

### 阶段 3 · 训练 Transformer prior

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_transformer.py --config configs/transformer.yaml --set \
  data.token_dir=data/tokens_full data.batch_size=32 train.grad_accum=4 \
  train.run_dir=runs/tf_full train.max_steps=4000 train.save_weights_only=true
```

**实测性能**（A100 单卡，1.32B，seq 257）：

| 项 | 实测值 |
|---|---|
| 吞吐 | **31–43 seq/s**（micro-batch 8 × 累积 4） |
| 显存 | 40.1 GB（几乎打满 40GB 卡） |
| 4000 步耗时 | **1.02 小时** |
| 单 checkpoint | **5.3 GB**（weights_only）/ **16 GB**（全量） |

### 阶段 4 · 采样

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python generate.py \
  --transformer_ckpt runs/tf_full/ckpt/final.pt \
  --vqgan_ckpt runs/vqgan_full/ckpt/final.pt \
  --output outputs/sample.png --n_samples 8 --temperature 1.0 --top_k 100 --seed 0
```

> 推理无 KV-cache，每步重算整个前缀，256 步的复杂度是 O(n³)。8 张图约 1 分钟。

---

## 4. 配置参考

超参的唯一真相是 `configs/*.yaml`，命令行只做覆盖：

```bash
--set train.lr=5e-5 data.batch_size=64 data.limit=null
```

- 点号路径可覆盖任意层级
- **覆盖不存在的键会直接报错**（防止拼错键名后被静默忽略）；确需新增用 `--set-new`
- 每次运行把最终配置快照存到 `run_dir/config.yaml`

> ⚠️ YAML 1.1 不把 `3e-4` 当浮点数（要求写 `3.0e-4`），会静默解析成**字符串**。
> `config.py` 已对科学计数法专门做了纠正，YAML 文件里和 `--set` 里都可以放心写 `3e-4`。

### 4.1 `configs/vqgan.yaml`

| 键 | 默认 | 说明 |
|---|---|---|
| `data.image_dir` | ALLaVA-LAION 路径 | 图片目录 |
| `data.image_size` | 256 | 下采样率固定 16 → 16×16=256 token |
| `data.batch_size` | 32 | 全局 batch，DDP 下按进程数均分 |
| `data.num_workers` | 16 | JPEG 解码是瓶颈，建议 ≥16 |
| `data.val_size` | 2000 | 从打乱后的头部切出，与训练集不重叠 |
| `data.limit` | null | 设整数只用前 N 张，快速试跑 |
| `data.recursive` | false | 是否递归扫子目录 |
| `model.codebook_size` | 1024 | 码本条目数 |
| `model.embedding_dim` | 256 | **码本维度，坍塌的主因之一** |
| `train.max_steps` | 200000 | 按 step 计，非 epoch |
| `train.lr` | 1.0e-4 | 原 yaml 的 4.5e-6 是带判别器时的配置 |
| `train.lr_schedule` | cosine | `cosine` \| `constant` |
| `train.warmup_steps` | 2000 | |
| `train.min_lr_ratio` | 0.1 | cosine 衰减到峰值的 10% |
| `train.grad_clip` | 1.0 | |
| `train.amp` | true | CUDA 上启用 fp16 AMP |
| `train.ema_decay` | null | 设 0.999 启用 EMA |
| `train.resume` | auto | `auto` = 接上 `run_dir/ckpt/latest.pt` |
| `train.log_every` | 100 | |
| `train.eval_every` | 2000 | 同时落盘一张重建对比图 |
| `train.eval_batches` | 30 | 验证只跑前 N 个 batch |
| `train.save_every` | 5000 | |
| `train.keep_last` | 3 | 只保留最近 N 个，自动清理 |
| `train.save_weights_only` | false | true = 体积降到 1/3，但**无法续训** |
| `train.tensorboard` | false | |

### 4.2 `configs/transformer.yaml`

| 键 | 默认 | 说明 |
|---|---|---|
| `data.token_dir` | data/tokens_train | `.npy` 目录 |
| `data.batch_size` | 64 | 全局 = world_size × grad_accum × micro-batch |
| `model.dim` | 2048 | |
| `model.n_layers` | 20 | |
| `model.n_heads` | 24 | 注意 `24×128=3072 ≠ dim=2048`（方案 A 原样保留） |
| `model.n_kv_heads` | 6 | GQA 4:1 |
| `model.head_dim` | 128 | |
| `model.vocab_size` | 1027 | 必须与 `constants.py` 一致 |
| `model.seq_len` | 257 | BOS + 16×16 |
| `model.init_std` | 0.02 | **null = 复现原实现**（初始 CE 约 1742，理论 6.93） |
| `train.grad_accum` | 1 | 显存不够时调大，等效 batch 不变 |
| `train.mask_ratio` | 0.2 | 走**双向掩码目标的概率**（不是被掩比例） |
| `train.mask_prob` | 0.15 | 掩码分支中被掩 token 的比例 |
| `train.weight_decay` | 0.1 | norm/bias/embedding 自动排除 |
| `train.keep_last` | 2 | 1.32B 全量 checkpoint 约 16GB/个 |

其余 `train.*` 键同 VQGAN。

---

## 5. 监控指标与判读

### 5.1 VQGAN

日志示例：

```
step 40000 | rec 0.1172 | vq 0.1164 | lr 1.02e-05 | grad_norm 0.152 |
            codebook_usage 0.0635 | img_per_s 724.7 | eta 0:01:50
step 40000 | val_rec 0.1202 | val_psnr 18.8 | val_codebook_usage 0.06152
```

| 指标 | 含义 | 健康范围 | 本项目实测 |
|---|---|---|---|
| `rec` | L1 重建损失 | 越低越好 | 0.60 → 0.117 |
| `vq` | commitment + codebook | 应快速降到 <1 并稳定 | 早期数千（震荡），后期 0.12 |
| `codebook_usage` | **最关键指标**，窗口内用到的码字比例 | >50% | **6.15%（严重偏低）** |
| `val_psnr` | 重建 PSNR (dB) | >22 可用 | **18.8** |
| `grad_norm` | 裁剪前梯度范数 | 稳定 | 早期数百，后期 <1 |

> `codebook_usage` 是**按日志窗口统计**的（每次 log 后清零），不是累计值——
> 累计值会单调上升而失去意义。

**判读规则**：

- 使用率持续 <10% → 码本已坍塌，重建质量被硬性限制，继续训收益很小
- 使用率 >50% 且 PSNR 仍低 → 是容量/损失函数问题，需要加 LPIPS/判别器
- `vq` 一直是几百上千 → 码本在剧烈震荡，需要 EMA 或降低码本维度

### 5.2 Transformer

```
step 4000 | loss 2.264 | ppl 9.62 | lr 4.71e-05 | grad_norm 0.60 |
            masked_frac 0.19 | seq_per_s 30.8 | eta 0:10:24
step 4000 | val_loss 2.352 | val_ppl 10.51
```

| 指标 | 含义 | 判读 |
|---|---|---|
| `loss` / `ppl` | 训练交叉熵 / 困惑度 | 初值应接近 `ln(vocab)=6.93`；若上千说明初始化有问题 |
| `val_ppl` | 验证困惑度（只跑因果目标） | **上限约 2^(码本熵)**，见下 |
| `masked_frac` | 走掩码分支的比例 | 应接近 `mask_ratio`(0.2) |
| `grad_norm` | | 前几步 `inf` 属正常（fp16 溢出，GradScaler 跳过） |

> ⚠️ **`val_ppl` 低不等于模型好。** 它的上限由 tokenizer 的码本熵决定：
> 码本熵 2.74 bit 时 ppl 上限约 6.7，模型轻松到 2.1；
> 码本熵 5.37 bit 时上限约 41，模型到 10.5。
> **只看 prior 的 loss 曲线会被完全误导，必须同时看 tokenizer 的码本使用率。**

### 5.3 必须人工看的

- `run_dir/samples/recon_*.png` —— 上排原图、下排重建。**这是判断 tokenizer 好坏最快的方式**
- `generate.py` 的采样图
- `run_dir/metrics.jsonl` —— 逐条指标，便于画曲线

---

## 6. 实验记录

### 实验 1 · VQGAN 小样本（基线）

| 配置 | 值 |
|---|---|
| 数据 | 5 万张子集 |
| 步数 / batch | 3000 / 32 |
| 耗时 | 2.5 分钟 |

**结果**：val PSNR 15.0，重建为灰褐色模糊团块，**颜色完全丢失**。

对 1024 张验证图统计 26 万个 token：

```
码字使用: 7 / 1024 (0.68%)
码本熵:   2.74 bit/token   (上限 10.00)
每图信息: 88 字节           (原图 196,608 字节)
top7 占比: 20.6 20.2 14.1 13.5 13.0 10.5 8.0 % = 100%
```

### 实验 2 · VQGAN 全量（当前基线）

| 配置 | 值 |
|---|---|
| 数据 | 全量 482,529 张（val 2000） |
| 步数 / batch | 40000 / 64（≈5.3 epoch） |
| lr | 1e-4，warmup 1000，cosine |
| 耗时 / 吞吐 / 显存 | 53 分钟 / 855 img/s / 16.2GB |

**结果**：

| 指标 | 实验 1 | 实验 2 |
|---|---|---|
| val PSNR | 15.0 | **18.8** (+3.8 dB) |
| val 重建 L1 | 0.219 | **0.120** |
| 码字使用 | 7 (0.68%) | **63 (6.15%)** |
| 码本熵 | 2.74 bit | **5.37 bit** |
| 每图信息量 | 88 字节 | **172 字节** |
| top8 集中度 | 100% | **40%** |

重建图恢复了颜色与主体布局（草地、天空、地毯、商品包装的配色都对），仍无细节与文字。

**码本使用率随训练的轨迹**：

```
step  2000 → 1.56%   PSNR 14.96
step 10000 → 3.03%   PSNR 17.49   (+1.47)
step 18000 → 4.69%   PSNR 18.34   (+1.66)
step 26000 → 5.57%   PSNR 18.66   (+0.88)
step 34000 → 5.96%   PSNR 18.77   (+0.39)
step 40000 → 6.15%   PSNR 18.80   (+0.19)   ← 增速衰减到 1/8
```

**结论**：数据量从 5 万加到 48 万把使用率从 0.68% 提到 6.15%，效果显著且真实；
但增速在明确衰减，PSNR 在 18.8 走平。外推再训 40 万步约到 8–10%，**到不了健康的 50%+**。
瓶颈是结构（码本 256 维、无 EMA、无死码复活），不是数据量。

### 实验 3 · Transformer prior（完整 1.32B）

| 配置 | 值 |
|---|---|
| 数据 | 12 万条 token（来自实验 2 的 tokenizer） |
| 模型 | dim 2048 / 20 层 / **1323.4M** |
| batch | 32（micro-batch 8 × 累积 4） |
| 步数 | 4000（≈1.08 epoch） |
| 耗时 / 吞吐 / 显存 | 1.02 小时 / 31–43 seq/s / 40.1GB |

```
step 1000 → val_ppl 11.97
step 2000 → val_ppl 11.22
step 3000 → val_ppl 10.74
step 4000 → val_ppl 10.51
```

采样图从灰团块变为有色块、有构图（可辨认「白底商品图」「街上人群」等构图模式），
但**尚无可辨识的物体**。

### 实验汇总

| 实验 | 硬件 | 耗时 | 关键产出 |
|---|---|---|---|
| 1 VQGAN 小样本 | A100 ×1 | 2.5 min | PSNR 15.0，使用率 0.68% |
| 2 VQGAN 全量 | A100 ×1 | 53 min | PSNR 18.8，使用率 6.15% |
| — 预编码 12 万 | A100 ×1 | 1.2 min | 473MB token |
| 3 Prior 1.32B | A100 ×1 | 61 min | val_ppl 10.51 |

---

## 7. 资源规划

### 7.1 显存

| 模型 | batch | 显存 | 备注 |
|---|---|---|---|
| VQGAN 19.7M | 64 @256px | 16.2 GB | |
| Prior 1.32B | 8 × accum 4 | 40.1 GB | 几乎打满 40GB 卡 |

1.32B 的显存构成：参数 5.3 + 梯度 5.3 + AdamW 双动量 10.6 = **21.2GB** 固定开销，
其余是激活值。**同卡上有其他人的进程时必然 OOM**，务必先看 `nvidia-smi`。

### 7.2 Checkpoint 体积

| 模型 | 全量 | weights_only |
|---|---|---|
| VQGAN 19.7M | 0.24 GB | 0.08 GB |
| Prior 1.32B | **16 GB** | **5.3 GB** |

启动日志会打印预估体积。`keep_last=3` 对 1.32B 就是 48GB，务必确认磁盘。

### 7.3 时间外推

| 任务 | 单卡 A100 |
|---|---|
| VQGAN 4 万步（5.3 epoch） | 53 分钟 |
| VQGAN 20 万步（默认配置） | 约 4.4 小时 |
| 预编码全量 48 万张 | 约 5 分钟 |
| Prior 1.32B 4000 步 | 1 小时 |
| Prior 1.32B 10 万步 | 约 25 小时 |

---

## 8. 分布式训练

```bash
NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_vqgan.sh
NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_transformer.sh
```

底层是 `torchrun` + 真 DDP。两个实现要点（改代码时不要破坏）：

1. **必须经 `LossWrapper.forward` 调用**。DDP 只在 forward 上挂梯度同步钩子，
   直接调 `model.module.compute_loss` 会绕过同步，多卡静默退化成各练各的。
2. **训练目标的分支选择必须各 rank 一致**。`mask_ratio` 的随机分支用独立的
   同种子生成器 `branch_rng` 决定，不能用各 rank 不同的全局 RNG。

梯度累积时，非最后一个 micro-step 走 `no_sync()`，省掉 `accum-1` 次 all-reduce。

> `train_transformer.py` 原本没有任何 DDP 逻辑，而 `train_transformer.sh` 却调用了
> `torch.distributed.launch --nproc_per_node=4` —— 会起 4 个互不通信的进程抢写同一个
> checkpoint。此问题已修复。

---

## 9. 断点续训

默认 `train.resume: auto`，自动接上 `run_dir/ckpt/latest.pt`：

```bash
# 直接重跑同一条命令即可续训
python train_vqgan.py --config configs/vqgan.yaml --set train.run_dir=runs/vqgan_full
# 指定具体 checkpoint
--set train.resume=runs/vqgan_full/ckpt/step_00020000.pt
# 强制从头开始
--set train.resume=null
```

checkpoint 内含 `model / optimizer / scheduler / scaler / ema / step / config`，
写入采用**原子替换**（先写 `.tmp` 再 rename），训练被杀不会留下半个文件。

> `save_weights_only=true` 的 checkpoint **不能用于续训**（不含 optimizer 状态）。

---

## 10. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `CUDA out of memory` | 同卡有他人进程 | `nvidia-smi` 换卡；调大 `train.grad_accum`；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| `val_loss 0` / `val_psnr` 异常 | 验证集不足一个 batch | 已改为验证集不 `drop_last`，且为空时直接报错退出 |
| 初始 loss 上千 | `model.init_std=null`，embedding 用了 `N(0,1)` 默认初始化 | 设 `init_std=0.02`（默认值） |
| `lr` 变成字符串 | YAML 1.1 不认 `3e-4` | 已在 `config.py` 修正，无需手工写 `3.0e-4` |
| 覆盖参数不生效 | 键名拼错 | 已改为直接报错；确需新增键用 `--set-new` |
| `for x in dataset` 卡死 | 越界索引被取模回绕，永不抛 `IndexError` | 已修复，两个 Dataset 都做显式越界检查 |
| 多卡 loss 不下降 | DDP 梯度同步被绕过 | 见第 8 节，必须经 `LossWrapper.forward` |
| uv 装包"成功"但 import 失败 | uv 网络失败时退出码仍为 0 | 安装后显式 `import torch` 验证 |
| 训练中途崩在某张图 | 语料里有损坏文件 | 已自动跳过并打印路径，不中断训练 |
| 启动慢 | 48 万文件 glob 约 7s | 已缓存到 `~/.cache/ugenimage`，可用 `UGEN_CACHE` 改位置 |
| 前几步 `grad_norm inf` | fp16 AMP 溢出 | 正常，GradScaler 会跳过这些步 |

---

## 11. 当前结论与下一步

### 11.1 已确立的基线

| 组件 | 指标 | 值 |
|---|---|---|
| Tokenizer | val PSNR / 码字使用 / 码本熵 | 18.8 dB / 6.15% / 5.37 bit |
| Prior | val_ppl | 10.51 |
| 全链路 | 采样质量 | 有颜色有构图，无可辨识物体 |

### 11.2 瓶颈定位

**Tokenizer 是全链路的天花板**，且已被量化：每张 256×256 图只携带 172 字节信息
（原图 196,608 字节，压缩比 1143:1）。prior 再强也无法超越这个上限。

三个结构性成因（见 `SCHEME_A_BASELINE.md` A.8）：
1. 码本 **256 维**过高 → 大面积死码
2. 无 EMA 更新、无死码复活
3. 无判别器、无感知损失 → 重建模糊

### 11.3 建议改进顺序

现在有了干净的对照基线，改一处跑一轮（53 分钟）就能直接比数字。

| 优先级 | 改动 | 工作量 | 预期 |
|---|---|---|---|
| 1 | 码本降到 **8 维 + L2 归一化** | ~5 行 | 使用率 6% → 50%+（LlamaGen 的关键发现） |
| 2 | 码本 **EMA + 死码复活** | ~30 行 | 进一步提升使用率与稳定性 |
| 3 | 加 **LPIPS 感知损失** | ~20 行 + 依赖 | 解决模糊（需下载 `lpips`，注意带宽） |
| 4 | 加 **PatchGAN 判别器** | ~80 行 | 细节与锐度 |
| 5 | 码本扩到 **16384** | 1 行 | 需先解决 1–2，否则更多死码 |

**对照实验协议**：固定 `data.limit=null`、`batch_size=64`、`max_steps=40000`、
`lr=1e-4`、`seed=0`，只改一个变量，比 `val_psnr` 与 `val_codebook_usage`。

### 11.4 未验证项

- **多卡 DDP 未在真实规模上跑过**（GPU 0/2–7 长期被占）。代码路径已由
  `e2e_check.sh` 覆盖，但吞吐与收敛未实测。
- **Prior 未训练到收敛**（仅 1.08 epoch）。在 tokenizer 改进前，投入更多 prior 算力性价比低。
- **无 FID/IS 评估**，目前只有重建 PSNR 与人工看图。

---

## 12. 产物位置

```
src/runs/vqgan_full/          实验 2
├── config.yaml               本次运行的完整配置快照
├── log.txt                   文本日志
├── metrics.jsonl             逐条指标
├── samples/recon_*.png       重建对比图（上排原图 / 下排重建）
└── ckpt/{final,latest}.pt

runs/tf_full/                 实验 3
├── ckpt/{final,latest,step_00004000}.pt
└── sample.png                采样图

data/tokens_full/             12 万条 token .npy（473MB）
src/runs/vqgan_trial/         实验 1
```
