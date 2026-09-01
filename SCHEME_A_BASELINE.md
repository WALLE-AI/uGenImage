# 方案 A：原方案（Baseline）

> 来源：`src/`（等价于 `deepseek_python_20260831_f520c2.py` 的生成产物，两者逐字节相同）
> 性质：**如实记录当前设计**，含其自身声明与实际实现的差异。
> 状态：**从未跑通过**，三条链路均不可执行。详见 `PLAN.md`。

---

## A.1 建模范式

**两阶段离散生成**（DALL·E-1 / VQGAN+Transformer 谱系）：

```
图像 256×256
  └─ Stage 1: VQ 编码器  f=16  ──► 16×16 = 256 个离散 token
       └─ Stage 2: 20 层 LLaMA 式 Transformer 建模 token 序列先验
            └─ 采样 ──► token 序列 ──► VQ 解码器 ──► 图像
```

- 无条件生成（无 class / 无 text / 无 CFG）
- 训练目标：**因果 AR 与掩码预测的随机混合**
- 推理：从 BOS 起逐 token 自回归，256 步

---

## A.2 Stage 1 — Tokenizer（`models/vqgan/`，108 行）

### 结构

| 模块 | 配置 |
|---|---|
| Encoder | 4 级 stride-2 卷积，通道 `3→128→256→512→512→256` |
| 每级构成 | `Conv(4,s2) + ReLU + Conv(3,s1) + ReLU` |
| 归一化 | **无** |
| 残差连接 | **无** |
| 瓶颈自注意力 | **无** |
| 下采样率 | f = 16（256px → 16×16） |
| Decoder | 对称，`Conv(3,s1)+ReLU+ConvTranspose(4,s2)+ReLU` ×4，末层 `Tanh` |

### 量化器（`codebook.py`）

| 项 | 值 |
|---|---|
| 码本 | 1024 条目 × **256 维** |
| 初始化 | `uniform(-1/1024, 1/1024)` |
| 距离 | `torch.cdist(z_flat, codebook)` |
| 梯度 | straight-through：`z_q = z_e + (z_q - z_e).detach()` |
| commitment loss | `0.25 · MSE(z_q.detach(), z_e)` |
| codebook loss | `MSE(z_q, z_e.detach())` |
| EMA 更新 | **无** |
| 死码复活 | **无** |
| L2 归一化 / 低维投影 | **无** |

### 损失

| 项 | 状态 |
|---|---|
| VQ loss（commit + codebook） | ✅ 已实现 |
| 重建损失（L1/L2） | ❌ **未实现** |
| 感知损失（LPIPS） | ❌ 无 |
| 对抗损失（判别器） | ❌ **无判别器** |

> 因此该模块名为 VQGAN，实为一个**连重建损失都没写的 VQ-VAE 骨架**。
> 训练脚本 `train_vqgan.py` 被 `src/README.md:5` 引用，但**文件不存在**。

---

## A.3 Stage 2 — Prior（`models/transformer/`，118 行）

### 声明配置

| 参数 | 值 |
|---|---|
| `dim` | 2048 |
| `n_layers` | 20 |
| `n_heads` | 24 |
| `n_kv_heads` | 6（GQA 4:1） |
| `head_dim` | 128 |
| `vocab_size` | 1028 |
| `seq_len` | 256 |
| 归一化 | RMSNorm，pre-norm |
| FFN | SwiGLU，`expansion = 4` |
| 位置编码 | RoPE（声明） |
| 权重共享 | `token_embedding` ↔ `lm_head` 绑定 |
| 标称参数量 | **0.39B** |

### 实际参数量核算

注意 `n_heads × head_dim = 24 × 128 = 3072 ≠ dim = 2048`，Q/O 投影比 dim 宽 50%。

| 项 | 计算 | 参数 |
|---|---|---|
| `wq` | 2048 × 3072 | 6,291,456 |
| `wk` | 2048 × 768 | 1,572,864 |
| `wv` | 2048 × 768 | 1,572,864 |
| `wo` | 3072 × 2048 | 6,291,456 |
| **Attention / 层** | | **15,728,640** |
| `w1` | 2048 × 8192 | 16,777,216 |
| `w2` | 2048 × 8192 | 16,777,216 |
| `w3` | 8192 × 2048 | 16,777,216 |
| **SwiGLU / 层** | | **50,331,648** |
| **合计 / 层** | | **66,060,288** |
| × 20 层 | | 1,321,205,760 |
| Embedding（绑定） | 1028 × 2048 | 2,105,344 |
| RMSNorm × 41 | | 83,968 |
| **总计** | | **≈ 1,323,395,072 ≈ 1.32B** |

> **实际参数量为标称 0.39B 的 3.4 倍。** FFN 占全模型 76%（标准 LLaMA 约 67%）。

### 位置编码实现状态

- `attention.py:6-11` 定义了 `rotate_half` / `apply_rotary_pos_emb`
- `block.py:32` 透传 `rope_cos` / `rope_sin`
- `model.py:22` 调用 `layer(x, mask)` —— **位置参数恒为 `None`**
- 全仓库**没有任何生成 cos/sin 的代码**（`git grep` 零命中）

→ **RoPE 声明存在，实际完全未接入。模型无任何位置信息。**

---

## A.4 训练目标

`model.py:26-44`，每个 step 二选一：

```
p = 0.2  →  掩码分支：随机 15% 非 PAD 位置置为 MASK_ID(1)，预测被掩 token
p = 0.8  →  因果分支：next-token 预测，shift 交叉熵
两分支均 ignore_index = 0 (PAD)
```

**实现缺陷**：掩码分支调用 `self.forward(masked_tokens, mask=None)`，而
`attention.py:37` 为 `is_causal=(mask is None)` —— `mask=None` 恰好触发因果掩码。
→ **掩码分支并非双向。实际训练 = 100% 因果 + 20% 输入被随机破坏。**

---

## A.5 Token 约定

| ID | 含义 | 定义位置 |
|---|---|---|
| 0 | PAD | `token_dataset.py:20` 硬编码 |
| 1 | MASK **且** BOS | `model.py:35` / `inference.py:9` |
| 2 … 1025 | 码本条目（原始 id + 2） | `preencode.py:38` 硬编码 |
| 1026, 1027 | 未使用 | — |

- offset `+2` 在 `preencode.py:38` 与 `inference.py:22` **两处独立硬编码**
- BOS 与 MASK **共用 id 1**
- `preencode.py` 写出的序列**不含 BOS**，而推理从 BOS 起采 → 训练/推理分布不一致

---

## A.6 训练配置

| 项 | 值 | 位置 |
|---|---|---|
| 优化器 | AdamW | `transformer_trainer.py:9` |
| lr | 3e-4 | argparse |
| weight_decay | 0.1（yaml）| **argparse 无此项 → `KeyError` 启动即崩** |
| 调度 | `CosineAnnealingLR(T_max=epochs)` | **全仓库无一处 `.step()` → LR 恒定** |
| warmup | **无** | |
| 混合精度 | `torch.cuda.amp`（torch≥2.4 已废弃） | |
| 梯度裁剪 | 1.0 | |
| batch_size | 16 | |
| epochs | 200 | |
| 梯度累积 | 无 | |
| EMA | 无 | |
| 分布式 | `torch.distributed.launch --nproc_per_node=4`，但**代码无 DDP/rank 逻辑** | `train_transformer.sh:13` |
| 恢复训练 | 无（仅存 `state_dict`） | |
| 随机种子 | 无 | |
| 日志/评估 | 无 | |

---

## A.7 推理

`inference.py`：

```
tokens = [1]                        # BOS
for _ in range(seq_len - 1):        # 255 次
    logits = model(tokens)          # 全量重算，无 KV-cache
    logits = logits[:, -1] / 0.9    # temperature
    top-k = 50
    next = multinomial(softmax(logits))
raw = tokens[:, 1:] - 2             # 255 个 token
raw.view(1, 16, 16)                 # ← 需要 256 个 → shape error
```

- 无 KV-cache → 整体复杂度 O(n³)
- 无 CFG
- token 数 off-by-one，必然崩溃

---

## A.8 方案 A 的能力边界

### 设计上成立的部分
- 两阶段离散生成路线本身有效（LlamaGen 已证明 AR 可达 diffusion 水平）
- 模块切分合理：`models / datasets / trainers / scripts` 四层
- 算子选型正确：RMSNorm + SwiGLU + GQA + RoPE + weight tying 是现代 LLM 标准配方

### 结构性缺陷（即使修完所有 bug 仍然存在）

| # | 缺陷 | 后果 |
|---|---|---|
| 1 | Tokenizer 无判别器/感知损失/残差块/归一化 | **重建模糊，构成全链路画质硬上限** |
| 2 | 码本 1024 × 256 维，无 EMA、无低维投影 | **码本坍塌几乎必然**，有效码字可能不足 200 |
| 3 | 混合因果/掩码目标 | 一套权重拟合两个不兼容推理模式，两边都不最优；且推理只用 AR，20% 掩码算力纯损耗 |
| 4 | raster-scan 顺序 | 对 2D 数据是弱先验，第 0 个 token 需在零上下文下决定全局结构 |
| 5 | 无条件化、无 CFG | 无条件生成不可评估；CFG 通常值数个 FID 点 |
| 6 | GQA (24:6) @ seq_len=256 | KV-cache 本就微不足道，GQA 在此损质量而不省资源 |
| 7 | 无 QK-norm、无深度缩放初始化 | 1.3B 视觉 token 训练易出现 attention logit 爆炸 |
| 8 | 1.32B 参数 / 256 token 序列 | 若数据为 ImageNet 规模（约 3.3 亿 token），模型约超配 10×，直接记忆训练集 |
| 9 | 推理步数 = 序列长度 | 扩展到视频（4096+ token）时不可接受 |
| 10 | 参数量标称错误 3.4× | 显存与算力预算全盘失准 |

### 视频扩展性评估
- Tokenizer 为纯 2D，无时间维接口
- RoPE 为 1D（且未接入），无 t/h/w 轴分解
- 推理步数随 token 数线性增长，序列长度随帧数线性增长 → **总成本随视频长度平方增长**
- 结论：**方案 A 无法直接扩展到视频**

---

## A.9 若坚持方案 A 的最小修复清单

仅为"能跑起来并训出东西"，不含质量优化：

1. `block.py` 补 `import torch.nn.functional as F`
2. `model.py` 补 `RMSNorm` 导入
3. 补 `weight_decay` 配置项
4. `inference.py` 循环改 `range(seq_len)`，修复 off-by-one
5. 编写 `train_vqgan.py`（至少含 L1 重建损失）
6. 实现并接入 RoPE（cos/sin 预计算 + 逐层传入）
7. 解耦 `causal` 与 `attn_mask`，令掩码分支真正双向
8. 拆分 BOS 与 MASK 的 id，预编码时写入 BOS
9. 调用 `scheduler.step()` 并补 warmup
10. 修正参数量标称，或调整结构使之落到 0.39B

完成后仍受 A.8 的 10 项结构性缺陷限制。改进方向见 `SCHEME_B_OPTIMIZED.md`。

---

## A.10 执行记录（2026-08-31）

A.9 全部 10 项已实施并验证。

| # | 修复 | 落点 | 验证 |
|---|---|---|---|
| 1 | 补 `import torch.nn.functional as F` | `models/transformer/block.py` | `test_transformer_forward_shape` |
| 2 | 补 `RMSNorm` 导入 | `models/transformer/model.py` | 同上 |
| 3 | 补 `--weight_decay` | `train_transformer.py` | e2e 第 3 步 |
| 4 | 修 off-by-one，采样恰好 `n_tokens` 个 | `inference.py` | `test_generate_token_count` |
| 5 | 新增 `train_vqgan.py`（L1 + VQ loss，含码本使用率日志） | `train_vqgan.py` | e2e 第 1 步 |
| 6 | 实现 `precompute_rope` 并逐层接入 | `attention.py` / `model.py` | `test_rope_is_actually_applied` |
| 7 | `causal` 与 `attn_mask` 解耦，掩码分支真双向 | `attention.py` / `block.py` / `model.py` | `test_causal_and_bidirectional_differ`、`test_bidirectional_sees_future` |
| 8 | BOS(1) 与 MASK(2) 拆分，offset→3，预编码写入 BOS | 新增 `constants.py` | `test_special_tokens_are_distinct` |
| 9 | warmup(2000) + 按 step 的 cosine，且真正 `.step()` | `trainers/transformer_trainer.py` | e2e 日志中 lr 变化 |
| 10 | 新增 `scripts/count_params.py`，标称改正为 1.32B | `configs/`、`README.md` | 实测 1323.39M |

**顺带修复**（属 A.8 C 组，成本极低）：`encode` 返回 `[B,H,W]`、`strict=True` 加载、
jpg/jpeg/png 扩展名统一、`torch.amp` 新 API、空掩码 NaN 保护、`--device` 支持 CPU、
checkpoint 含 optimizer/scheduler/scaler/step 可续训、`--seed`、
`train_transformer.sh` 去掉不可用的 `torch.distributed.launch`、新增 `generate.py` 采样入口。

### 验证结果

```
pytest                      → 13 passed
scripts/e2e_check.sh (CPU)  → 训VQGAN → 预编码 → 训Transformer → 采样，四步全通过
scripts/count_params.py     → 1323.39M (1.323B)，attention 23.8% / ffn 76.1%
A100 实机 (dim2048/20层/bs16/seq257) → 正常前向反传，loss 下降，lr 按调度变化
```

### 执行中新发现的问题（未修复，需决策）

**`token_embedding` 没有任何显式初始化**，沿用 `nn.Embedding` 默认的 `N(0,1)`：

| 指标 | 实测 | 应为 |
|---|---|---|
| embedding std | 1.000 | 0.02（GPT/LLaMA 惯例） |
| logits std | 71.0 | ~1 |
| 初始交叉熵 | **1741.9** | ln(1027) = 6.93 |

由于 `lm_head` 与 embedding 绑定，std=1.0 会被放大到 logits 上，模型开局需要花费大量步数
仅仅把 logit 幅度压回正常范围；配合 AMP fp16，前若干步梯度直接溢出（实测 `grad_norm inf`，
GradScaler 跳过更新）。这属于 A.8 #7 的范畴，修复只需在 `VisualTransformer.__init__` 末尾加：

```python
def _init(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.normal_(m.weight, std=0.02)
self.apply(_init)
# 输出投影按深度缩放
for blk in self.layers:
    nn.init.normal_(blk.attn.wo.weight, std=0.02 / math.sqrt(2 * config['n_layers']))
    nn.init.normal_(blk.ffn.w3.weight, std=0.02 / math.sqrt(2 * config['n_layers']))
```

**未擅自实施**，因为它改变模型初始化行为，超出 A.9 清单范围。建议采纳。
