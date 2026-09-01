# uGenImage 执行方案

> 制定日期：2026-08-31
> 目标：把当前"一份从未跑通的 460 行草稿"改造成可长期演进的图像/视频生成模型训练仓库。
> 依据：对 `deepseek_python_20260831_f520c2.py`(561行) 与 `src/`(461行) 的逐行审计。

---

## 0. 当前状态基线

| 项 | 现状 |
|---|---|
| 代码总量 | 生成器 561 行；`src/` 461 行（**两者逐字节等价，是同一份代码的两个副本**） |
| 技术路线 | 两阶段离散生成：VQ tokenizer + LLaMA 式自回归 prior |
| 可运行性 | **三条链路（VQGAN训练 / 预编码 / Transformer训练）无一可端到端执行** |
| 崩溃级缺陷 | 5 处 |
| 设计级缺陷 | 6 处（不报错，但会让训练学不到东西） |
| 工程缺失 | 无依赖清单 / 无测试 / 无 lint / 无 CI / 无 DDP / 无评估 / 无恢复 |
| 标称参数量 | README 写 0.39B，实测 **≈1.32B**（偏差 3.4×） |

**核心风险**：`deepseek_python_20260831_f520c2.py` 在 `__main__` 里对输出目录执行 `shutil.rmtree`。只要它还是"源码真相"，任何直接改 `src/` 的工作都会在下次运行时被清空。

---

## 阶段 P0：结构去重与工程底座（前置，阻塞其余一切）

**目的**：让"改一次代码"只需要改一个地方，并让后续每一处修复都能被自动验证。

### P0-1 确立 `src/` 为唯一真相
- 归档生成器：`git mv deepseek_python_20260831_f520c2.py tools/legacy_generator.py.bak`，并在文件头加注释说明"仅作历史留档，**不要运行**，它会 rmtree 输出目录"。
- 或直接删除（`src/` 已完整保留其全部产物，无信息损失）。
- 更新 `CLAUDE.md`：删除"编辑 FILES[...] 再重新生成"的工作流描述，改为"直接编辑 `src/`"。
- 更新根 `README.md`（当前仅 2 行占位）。

**验收**：仓库中不存在任何会 `rmtree` 源码目录的可执行脚本。

### P0-2 依赖与工具链
新建 `pyproject.toml`：
```toml
[project]
name = "ugenimage"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["torch>=2.4", "torchvision", "pillow", "numpy", "tqdm", "pyyaml", "einops"]

[project.optional-dependencies]
dev = ["pytest", "ruff"]

[tool.ruff]
line-length = 120

[tool.setuptools.packages.find]
where = ["src"]
```
- `uv sync`（`.venv` 已建好，Python 3.12.7）
- `uv pip install -e .` → 使 `models.*` / `datasets.*` 的绝对导入在任意 cwd 下生效（当前必须在 `src/` 下运行）
- 新建 `.gitignore`：`.venv/ __pycache__/ *.pth *.npy data/ checkpoints/ outputs/ wandb/`

**验收**：`uv run python -c "import models.transformer.model"` 从仓库根目录成功。

### P0-3 冒烟测试（本阶段最高价值项）
新建 `tests/test_smoke.py`，**全部在 CPU 上、用极小配置运行**：
1. `test_vqgan_roundtrip`：随机 `[2,3,64,64]` → VQGAN → 断言重建 shape 一致、indices 落在 `[0,1023]`、vq_loss 有限
2. `test_transformer_forward`：tiny config（dim=64, layers=2, heads=4, kv_heads=2, head_dim=16）→ 断言 logits shape `[B,T,vocab]`
3. `test_transformer_backward`：`compute_loss` → `backward()` → 断言**所有参数 grad 非 None 且非 NaN**
4. `test_causal_vs_bidirectional`：断言掩码分支与因果分支的注意力可见性**确实不同**（直接覆盖 P1-7 这条设计缺陷）
5. `test_rope_applied`：断言打乱输入 token 顺序会改变输出（**直接覆盖 P1-6，位置编码未接入**）
6. `test_generate_shape`：`generate_autoregressive` 产出 token 数 == seq_len（**直接覆盖 P1-4 的 off-by-one**）
7. `test_preencode_roundtrip`：造 3 张临时图 → 预编码 → TokenDataset 读取 → 断言 shape/dtype/取值范围

**验收**：`uv run pytest` —— P0 阶段 7 个用例中预计 6 个失败，这正是 P1 的工作清单。

---

## 阶段 P1：缺陷修复（让代码真的能训练）

### A 组 — 崩溃级（5 项，不改则无法启动）

| # | 文件:行 | 问题 | 修复 |
|---|---|---|---|
| 1 | `src/models/transformer/block.py:22` | 用 `F.silu` 未导入 F | 加 `import torch.nn.functional as F` |
| 2 | `src/models/transformer/model.py:15` | 用 `RMSNorm` 未导入 | `from models.transformer.block import TransformerBlock, RMSNorm` |
| 3 | `src/train_transformer.py:23` ↔ `trainers/transformer_trainer.py:9` | config 来自 `vars(args)`，但无 `--weight_decay` → 启动即 KeyError | 由 P2-1 的 YAML 配置层统一解决；临时可加 argparse 项 |
| 4 | `src/inference.py:22-25` | 循环产出 256 token，去 BOS 剩 255，却 `view(1,16,16)` 需 256 → shape error | 循环改为 `range(seq_len)`，或保留 BOS 的正确切片 |
| 5 | `src/train_vqgan.py` | 被 `src/README.md:5` 引用但**文件不存在** | 见 P1-B-5 |

### B 组 — 设计级（6 项，不报错但训不出东西）

**6. RoPE 完全未接入（最严重）**
`attention.py:10` 定义了 `apply_rotary_pos_emb`，`block.py:32` 透传了 `rope_cos/rope_sin`，但 `model.py:22` 调用 `layer(x, mask)` —— 位置参数恒为 `None`；且**全仓库没有任何生成 cos/sin 的代码**。
→ 模型对 16×16 token 网格无任何位置感知，等价于词袋模型。
**修复**：在 `attention.py` 增加 `precompute_rope(head_dim, max_seq_len, theta=10000.0)`；`VisualTransformer.__init__` 中注册为 buffer；`forward` 中按当前 T 切片并逐层传入。为 P3 的视频扩展预留 `axes` 参数（1D→3D t/h/w）。

**7. "双向掩码预测"实为单向**
`model.py:36` 传 `mask=None`，而 `attention.py:37` 是 `is_causal=(mask is None)` —— `None` 恰好触发因果掩码。混合目标塌缩为"两支都是因果，只是一支输入被破坏"。
**修复**：给 `forward` 增加显式 `causal: bool` 参数，与 `attn_mask` 解耦；掩码分支传 `causal=False`。

**8. BOS 与 MASK 共用 id=1，且训练序列无 BOS**
`preencode.py:38` 只写 `codebook_id + 2`，序列不含 BOS；`inference.py:9` 却从 token 1 起采。模型从未见过以 1 开头的序列 → 推理第一步即 OOD。
**修复**：确立并集中定义特殊 token 表 —— 新建 `src/constants.py`：
```python
PAD_ID = 0
BOS_ID = 1
MASK_ID = 2
CODEBOOK_OFFSET = 3
CODEBOOK_SIZE = 1024
VOCAB_SIZE = CODEBOOK_OFFSET + CODEBOOK_SIZE  # 1027
```
`preencode.py` / `inference.py` / `model.compute_loss` 三处一律引用此文件（当前 offset=+2 硬编码在两处，改一处就会静默错位）。预编码时在序列头部写入 BOS。

**9. LR 调度器创建但从不 step，且无 warmup**
`transformer_trainer.py:11` 建了 `CosineAnnealingLR`，全仓库无一处 `.step()` → LR 恒为 3e-4。对十亿级模型 + 无 warmup 是典型发散配置。
**修复**：改为 **按 step 计** 的 warmup + cosine（`LambdaLR`），warmup 约 2000 步；在 `train_step` 末尾调用 `scheduler.step()`。

**10. "VQGAN"里没有 GAN**
仅 Encoder/Decoder/VQ + 隐式 MSE，**无判别器、无感知损失(LPIPS)、无对抗损失**；`codebook.py` 无 EMA 更新、无死码复活 → 码本坍塌几乎必然，且重建模糊构成整条链路的画质天花板。
**修复（二选一，需决策）**：
- (a) 补齐：PatchGAN 判别器 + LPIPS + 对抗损失（延迟若干步开启）+ codebook EMA + dead-code revival；
- (b) 降级命名：改名 `VQVAE`，明确接受画质上限，先跑通全链路。
建议先 (b) 跑通，再 (a) 提质。

**11. 参数量标称错误**
实算：Attention/层 15.73M（注意 `n_heads×head_dim = 3072 ≠ dim = 2048`，Q/O 投影比 dim 宽）+ SwiGLU/层 50.33M（三个 2048×8192 矩阵，约为常规 SwiGLU 的 1.5×）= 66.06M/层 × 20 层 + embedding 2.11M ≈ **1.32B**。
**修复**：加 `scripts/count_params.py` 打印真实参数量；然后二选一 —— 改标称为 1.3B，或改 SwiGLU 为 `hidden = 8/3·dim` 并调小 dim/层数真正落到 0.39B。**这个数字直接决定显存与算力预算，必须先定。**

### C 组 — 工程正确性（10 项）

- **12** `train_transformer.sh:13` 用已废弃的 `torch.distributed.launch`，而 `train_transformer.py` 无任何 DDP/rank 逻辑 → 4 进程独立训练并抢写同一个 `checkpoint_*.pth`。改 `torchrun` + 真正的 DDP 初始化（见 P2-3）。
- **13** `preencode.py:19` `strict=False` 加载权重 → checkpoint 不匹配会**静默**用随机码本编码整个数据集。改 `strict=True` 并显式校验。
- **14** `preencode.py:32` 只 glob `*.jpg`，而 `image_dataset.py:10` 收 jpg+png → 数据静默丢失。统一扩展名集合。
- **15** `preencode.py:37` `squeeze(0).flatten()` 仅在 batch=1 时凑巧正确（`encode` 返回展平的 `[B*H*W]`）。改为返回 `[B,H,W]` 并支持批量编码（同时大幅提速）。
- **16** 设备硬编码：`preencode.py:17` `device='cuda'`、`train_transformer.py:27` `.cuda()`、`transformer_trainer.py:7,14` `.cuda()`。统一为 `--device` 参数，支持 CPU（否则 P0-3 的冒烟测试无法运行）。
- **17** `torch.cuda.amp.autocast/GradScaler`（`transformer_trainer.py:3`）在 torch≥2.4 已废弃 → 改 `torch.amp.autocast('cuda')` / `torch.amp.GradScaler('cuda')`。
- **18** 掩码分支若采样出空掩码，`cross_entropy` 作用于 0 元素 → NaN。加空掩码保护（回退到因果分支）。
- **19** 无 checkpoint 恢复：`trainers/transformer_trainer.py:26` 只存 `state_dict`。改存 `{model, optimizer, scaler, scheduler, epoch, step, config}` + `--resume`。
- **20** 无随机种子、无 `torch.backends` 确定性设置。加 `--seed`。
- **21** 无验证集切分、无梯度累积。

---

## 阶段 P2：训练底座

### P2-1 配置系统（消灭三处重复）
当前超参在 **argparse / `scripts/*.sh` / `configs/*.yaml`** 三处各写一遍，而 `configs/*.yaml` **没有任何代码读取**（`git grep yaml` 零命中）。
- 新建 `src/config.py`：加载 YAML → dataclass，支持 `--config x.yaml --override lr=1e-4` 风格覆盖。
- `train_transformer.py` / `preencode.py` / `train_vqgan.py` 全部改为 config 驱动。
- `scripts/*.sh` 瘦身为仅指定 config 路径与分布式参数。
- 顺带解决 A-3（`weight_decay` KeyError）。

**验收**：改一个超参只需改一个文件。

### P2-2 数据管线
- 当前 `TokenDataset` 一图一 `.npy`（256×int32 ≈ 1KB），`list(glob('*.npy'))` 在百万文件下会卡死，且 IOPS 被打爆。
- 改为 **shard 化**：每个 shard 打包 N 万条序列为单个 `.npy`/`.bin` + memmap 索引；或引入 WebDataset。
- 预编码支持批量 + 多进程 + 断点续跑（跳过已存在输出）。
- 加数据校验脚本：统计码本使用率直方图（**直接观测坍塌**）、序列长度分布、损坏文件。

### P2-3 分布式与训练循环
- DDP（先）→ FSDP（模型继续放大时）
- 梯度累积、activation checkpointing、EMA 权重
- 按 step 而非 epoch 组织训练循环（`train_transformer.py:30` 当前按 epoch，与 warmup/cosine 天然冲突）
- 日志：TensorBoard 或 W&B —— loss、LR、grad-norm、throughput(tok/s)、码本使用率
- 定期用固定种子采样并落盘图片（当前**没有任何训练中的视觉反馈**）

### P2-4 评估
- 图像：FID / IS / 重建 PSNR-SSIM（VQGAN 阶段）
- 加 `scripts/eval.py`，产出可对比的 JSON 报告

**P2 完成标志**：能在小数据集（如 COCO 子集 / FFHQ）上训到 loss 稳定下降，并采样出**可辨识内容**的图片。这是"代码正确性"的最终验收——在此之前所有画质讨论都无意义。

---

## 阶段 P3：面向视频的扩展

> 前置：P3 开始前必须先做 **技术路线决策**（见下方"待决策项 D1"）。

### 若延续离散自回归路线
1. **Tokenizer 3D 化**：causal 3D conv encoder，时间维下采样（如 4×）。当前 `seq_len=256` 的假设在视频下彻底失效——16 帧 256px 未做时间压缩就是 4096 token。
2. **3D RoPE**：t/h/w 三轴分解（P1-6 修复时预留接口）。
3. **长序列基础设施**：FlashAttention、序列并行 / context parallel、**KV-cache**（`inference.py:12` 当前每步全量重算，整体复杂度 O(n³)）。
4. **条件化**：text encoder（T5/CLIP）+ cross-attention 或 in-context 注入；**CFG**。当前是纯无条件生成，无任何条件接口。
5. **数据**：视频解码、分桶（分辨率/时长/FPS）、latent 预缓存、场景切分与去重。
6. **评估**：FVD、CLIP-score、时序一致性指标。

### 若转向 Diffusion / Flow-matching DiT
现有代码可复用的仅有目录约定与 RMSNorm/GQA/SwiGLU 几个算子（约占现有代码 20%），codebook 与 AR 训练/推理逻辑全部作废。需新建：连续 VAE、DiT backbone、flow-matching 训练目标、采样器（Euler/DPM）、CFG。

---

## 待决策项（阻塞后续阶段，需明确回答）

| ID | 决策 | 阻塞 | 备注 |
|---|---|---|---|
| **D1** | 最终目标是图像还是视频？走**离散 AR** 还是 **Diffusion DiT**？ | P3 全部 | 视频生成的业界主流（Sora / HunyuanVideo / Wan / CogVideoX）是 DiT + flow matching；现有代码只在 AR 路线上有约 30% 骨架 |
| **D2** | 模型规模：修正标称为 1.3B，还是缩回真实的 0.39B？ | P1-11、全部算力预算 | |
| **D3** | VQGAN 是补齐判别器+LPIPS，还是先降级为 VQ-VAE 跑通？ | P1-10、画质天花板 | 建议先降级跑通 |
| **D4** | 训练硬件规格（GPU 型号 / 卡数 / 显存）？ | P2-3 并行策略、batch size | `train_transformer.sh:2` 假设 4 卡 |
| **D5** | 目标数据集与规模？ | P2-2 分片策略 | |
| **D6** | 是否需要文本条件（文生图/文生视频）？ | P3-4，影响 tokenizer 与架构设计 | 当前架构无条件接口 |

---

## 建议执行顺序与产出

| 顺序 | 内容 | 产出 |
|---|---|---|
| 1 | **P0 全部** | 单一代码源 + `pyproject.toml` + 7 个冒烟测试（预计 6 个失败） |
| 2 | **P1-A（5 项崩溃）** | 冒烟测试 1/2/3 通过 |
| 3 | **P1-B-6,7,8,9（RoPE/双向/BOS/scheduler）** | 冒烟测试全绿 |
| 4 | **P1-B-10,11 + P1-C** | 参数量对齐、VQ 训练脚本可用 |
| 5 | **P2-1 配置系统** | 超参单点维护 |
| 6 | **P2-2/3/4** | 小数据集端到端训练 + 可辨识采样 |
| 7 | **D1 决策 → P3** | 视频扩展 |

**建议立即从第 1 步开始**：P0-3 的冒烟测试会在 30 秒内把 P1 的 11 个缺陷全部自动暴露出来，此后每一处修复都立即可验证——这比先逐个手改缺陷要可靠得多。
