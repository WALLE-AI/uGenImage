# Visual Transformer（方案 A · VQGAN + 20 层 LLaMA 式 prior）

两阶段离散图像生成。设计说明见仓库根目录的 `SCHEME_A_BASELINE.md`，
改进方向见 `SCHEME_B_OPTIMIZED.md`，工程路线见 `PLAN.md`。

> **参数量**：Transformer prior 实测约 **1.32B**（原标称 0.39B 有误），
> 复核：`python scripts/count_params.py`

## 环境

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv torch torchvision pillow numpy tqdm pyyaml pytest
```

## 配置系统

超参的唯一真相是 `configs/*.yaml`。命令行只做覆盖，不重复定义：

```bash
python train_vqgan.py --config configs/vqgan.yaml
python train_vqgan.py --config configs/vqgan.yaml --set train.lr=5e-5 data.batch_size=64
python train_transformer.py --config configs/transformer.yaml --set data.limit=5000 train.max_steps=1000
```

- 点号路径覆盖任意层级；取值按 YAML 语法解析（`null` / `true` / `1e-5` 都正确）
- 覆盖一个配置中不存在的键会**直接报错**，防止拼错键名后被静默忽略
- 每次运行会把最终配置快照存到 `run_dir/config.yaml`

## 流程

所有命令在 `src/` 下执行。

```bash
# 0. 冒烟测试（CPU，3 秒）
cd .. && pytest -q && cd src

# 1. 训练 VQGAN
bash scripts/train_vqgan.sh                     # 单卡
NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_vqgan.sh   # 多卡 DDP

# 2. 预编码为 token（支持断点续跑；模型结构自动从 checkpoint 读取）
python preencode.py --vqgan_ckpt runs/vqgan/ckpt/final.pt \
    --image_dir <图片目录> --output_dir data/tokens_train --batch_size 32

# 3. 训练 Transformer prior
bash scripts/train_transformer.sh
NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_transformer.sh

# 4. 采样
python generate.py --transformer_ckpt runs/transformer/ckpt/final.pt \
    --vqgan_ckpt runs/vqgan/ckpt/final.pt --output outputs/sample.png --n_samples 4
```

端到端连通性自检（合成数据，CPU 可跑，含断点续训验证）：

```bash
bash scripts/e2e_check.sh
```

## 数据

本机 `/home/dataset0/images/ALLaVA-4V/allava_laion/image_chunks/images` 已有
**481,613 张 ≥256px 的 LAION 图片**（扁平目录、全 RGB），已设为 `configs/vqgan.yaml` 的默认值，
无需下载。

## 运行目录结构

```
runs/vqgan/
├── config.yaml          # 本次运行的完整配置快照
├── log.txt              # 文本日志
├── metrics.jsonl        # 逐条指标，便于后续画图
├── samples/             # 定期的重建对比图（上排原图 / 下排重建）
└── ckpt/
    ├── step_00005000.pt # 全量状态：model/optimizer/scheduler/scaler/ema/step/config
    ├── latest.pt        # 符号链接，--set train.resume=auto 会自动接上
    └── final.pt
```

## 训练特性

| 项 | 说明 |
|---|---|
| 训练循环 | 按 step 计（`train.max_steps`），与 warmup/cosine 相容 |
| 学习率 | warmup + cosine，**每步 step()** |
| 断点续训 | `train.resume: auto` 默认开启，原子写入避免半个文件 |
| 分布式 | `torchrun` + 真 DDP（`NGPU>1`），梯度累积时跳过中间同步 |
| 验证 | 独立 val 切分；VQGAN 报 rec/PSNR/码本使用率，prior 报 val_loss/ppl |
| 采样监控 | VQGAN 定期落盘重建对比图 |
| 鲁棒性 | 损坏图片自动跳过（48 万张里必有坏文件），不中断训练 |
| 文件清单 | 缓存到 `~/.cache/ugenimage`（481k 文件 glob 一次约 7s） |
| EMA | `train.ema_decay` 设为 0.999 即启用；采样加 `--use_ema` |

## Token 约定

统一定义在 `constants.py`，不要在别处硬编码：

| ID | 含义 |
|---|---|
| 0 | PAD |
| 1 | BOS |
| 2 | MASK |
| 3 … 1026 | 码本条目（原始 id + 3） |

序列长度 `257 = BOS + 16×16`。

## 已知限制（方案 A 固有，非 bug）

- Tokenizer 无判别器、无感知损失、无残差块/归一化 → **重建模糊，构成全链路画质上限**
- 码本 1024×256 维、无 EMA、无死码复活 → 坍塌风险高，**务必盯 `codebook_usage`**
- 无条件生成，无 CFG
- 推理无 KV-cache，256 步全量重算
- 混合因果/掩码目标本身自相矛盾（见 `SCHEME_B_OPTIMIZED.md` 1.2）

详见 `SCHEME_A_BASELINE.md` A.8。
