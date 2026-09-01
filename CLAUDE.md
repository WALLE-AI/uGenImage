# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A two-stage discrete image-generation codebase: **VQ tokenizer + LLaMA-style autoregressive prior**.
The real source lives in `src/` — edit it directly.

`deepseek_python_20260831_f520c2.py` is a **retired generator** that originally materialized this
project from string literals. It is now stale (it emits the pre-fix code) and carries a warning
banner. Do not run it; do not edit it as a way of editing the model code. See `PLAN.md` P0-1.

## Documents

| 文件 | 内容 |
|---|---|
| `SCHEME_A_BASELINE.md` | 方案 A（当前实现）的完整规格、参数量核算、10 项结构性缺陷 |
| `SCHEME_B_OPTIMIZED.md` | 方案 B（优化设计）：FSQ tokenizer、掩码并行、条件化+CFG、规模阶梯 |
| `PLAN.md` | 工程路线 P0–P3 与待决策项 D1–D6 |
| `src/README.md` | 使用说明 |

## Layout

```
src/
├── constants.py            # Token 约定的唯一真相 (PAD=0, BOS=1, MASK=2, offset=3)
├── config.py               # YAML 配置系统：--config + --set 点号覆盖
├── configs/{vqgan,transformer}.yaml   # 超参的唯一真相
├── datasets/               # ImageDataset / TokenDataset（清单缓存 + 坏文件跳过 + train/val）
├── models/vqgan/           # Encoder/Decoder + VectorQuantize（无判别器，实为 VQ-VAE）
├── models/transformer/     # GQA + RoPE / RMSNorm + SwiGLU / VisualTransformer
├── utils/                  # distributed(DDP) / logging / checkpoint / ema / schedule
├── train_vqgan.py          # 阶段 1
├── preencode.py            # 阶段 2：图片 -> token .npy
├── train_transformer.py    # 阶段 3
├── inference.py generate.py# 阶段 4
└── scripts/                # shell 封装 + count_params.py + e2e_check.sh
tests/test_smoke.py         # 18 个 CPU 冒烟用例
```

## Pipeline

```bash
cd src
bash scripts/train_vqgan.sh                  # 1（NGPU=4 起 DDP）
python preencode.py --vqgan_ckpt runs/vqgan/ckpt/final.pt ...   # 2
bash scripts/train_transformer.sh            # 3
python generate.py --transformer_ckpt ... --vqgan_ckpt ...      # 4
```

自检：`pytest`（仓库根目录）与 `bash scripts/e2e_check.sh`（合成数据跑完四步 + 验证断点续训）。

## 关键事实

- 超参改动只改 `configs/*.yaml`；命令行用 `--set a.b=c` 覆盖，**拼错键名会报错而非静默忽略**
- YAML 1.1 不认 `3e-4` 这种写法（会得到字符串），`config.py` 已专门纠正这一情况
- Transformer prior 实测 **1.32B**，非 0.39B。复核：`python scripts/count_params.py`
- `n_heads × head_dim = 3072 ≠ dim = 2048`；SwiGLU expansion=4 使 FFN 占 76% 参数
- Token 约定只在 `constants.py` 定义，不要在 preencode/inference 里硬编码 offset
- 序列长度 257 = BOS + 16×16
- `mask_ratio` 控制“走哪个训练目标”，`mask_prob` 才是被掩 token 的比例
- DDP 下训练目标的分支选择必须各 rank 一致（`train_transformer.py` 用独立同种子生成器）
- DDP 必须经 `LossWrapper.forward` 调用，直接调 `model.module.compute_loss` 会绕过梯度同步
- `model.init_std` 默认 0.02；设为 `null` 可复现原实现（初始 CE 约 1742，理论 6.93）
- 数据：`/home/dataset0/images/ALLaVA-4V/allava_laion/image_chunks/images` 有 48 万张 ≥256px 图片

## Conventions

注释与用户可见字符串使用中文，保持这一风格。
