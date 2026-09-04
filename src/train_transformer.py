"""阶段 3：Transformer prior 训练。

用法:
    python train_transformer.py --config configs/transformer.yaml
    python train_transformer.py --config configs/transformer.yaml --set train.max_steps=200 data.limit=1000
    torchrun --nproc_per_node=4 train_transformer.py --config configs/transformer.yaml
"""
import contextlib
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import base_parser, config_from_args, save_config
from datasets.token_dataset import TokenDataset
from models.transformer.model import VisualTransformer
from utils.checkpoint import (estimate_size_gb, load_checkpoint, resolve_resume,
                              save_checkpoint)
from utils.distributed import (all_reduce_mean, barrier, cleanup, init_distributed,
                               is_main)
from utils.ema import ModelEma
from utils.logging import AverageMeter, Logger, Throughput
from utils.schedule import build_scheduler, param_groups_no_decay


class LossWrapper(nn.Module):
    """把 compute_loss 暴露成 forward。

    DDP 只在 forward 上挂梯度同步钩子，直接调用 model.module.compute_loss
    会绕过同步，多卡训练静默退化成各练各的。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, mask_ratio, mask_prob, use_masked):
        return self.model.compute_loss(tokens, mask_ratio, mask_prob, use_masked)


def infinite(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def build_loader(cfg, split, batch_size, world_size, rank, shuffle):
    ds = TokenDataset(cfg.data.token_dir, cfg.model.seq_len, split=split,
                      val_size=cfg.data.val_size, limit=cfg.data.get('limit'),
                      seed=cfg.train.seed)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                     shuffle=shuffle, drop_last=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=(shuffle and sampler is None),
                        sampler=sampler, num_workers=cfg.data.num_workers, pin_memory=True,
                        # 验证集不能 drop_last：样本不足一个 batch 时会静默变成 0 个 batch，
                        # val_loss 直接报 0，看起来像“完美模型”
                        drop_last=(split == 'train'),
                        persistent_workers=cfg.data.num_workers > 0)
    if split == 'val' and len(loader) == 0:
        raise SystemExit(f"验证集为空（{len(ds)} 条，batch={batch_size}），请调大 data.val_size")
    return ds, loader, sampler


@torch.no_grad()
def evaluate(core, loader, device, use_amp, max_batches=50, amp_dtype=torch.float16):
    """验证只跑因果目标，指标才可比（训练时掩码分支是随机触发的）。"""
    core.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        tokens = batch.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            loss = core._causal_loss(tokens)
        total += loss.item()
        n += 1
    core.train()
    return total / max(n, 1)


def main():
    args = base_parser('训练 Transformer prior').parse_args()
    cfg = config_from_args(args)

    rank, world_size, local_rank = init_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.train.seed + rank)
    # 训练目标的分支选择必须各 rank 一致，用独立的同种子生成器
    branch_rng = torch.Generator().manual_seed(cfg.train.seed)

    run_dir = Path(cfg.train.run_dir)
    ckpt_dir = run_dir / 'ckpt'
    log = Logger(run_dir, cfg.train.get('tensorboard', False))
    if is_main():
        save_config(cfg, run_dir / 'config.yaml')

    # ---- 数据 ----
    accum = int(cfg.train.get('grad_accum', 1))
    per_rank_bs = cfg.data.batch_size // (world_size * accum)
    if per_rank_bs < 1:
        raise SystemExit(f"batch_size({cfg.data.batch_size}) 不足以拆分到 "
                         f"{world_size} 进程 x {accum} 累积步")
    train_ds, train_loader, train_sampler = build_loader(cfg, 'train', per_rank_bs, world_size, rank, True)
    val_ds, val_loader, _ = build_loader(cfg, 'val', per_rank_bs, 1, 0, False)
    log.info(f"训练集 {len(train_ds):,} 条 | 验证集 {len(val_ds):,} 条 | "
             f"world_size={world_size} 每卡 micro-batch={per_rank_bs} 累积={accum} "
             f"总 batch={per_rank_bs*world_size*accum}")

    # ---- 模型 ----
    model_cfg = dict(cfg.model.to_dict())
    core = VisualTransformer(model_cfg).to(device)
    n_params = sum(p.numel() for p in core.parameters())
    wo = cfg.train.get('save_weights_only', False)
    log.info(f"Transformer 参数量 {n_params/1e6:.1f}M ({n_params/1e9:.3f}B) | "
             f"init_std={model_cfg.get('init_std')} | 单个 checkpoint 约 "
             f"{estimate_size_gb(core, not wo):.1f}GB × keep_last={cfg.train.keep_last}")

    optimizer = torch.optim.AdamW(param_groups_no_decay(core, cfg.train.weight_decay),
                                  lr=cfg.train.lr, betas=(0.9, 0.95))
    scheduler = build_scheduler(optimizer, cfg.train, cfg.train.max_steps)
    use_amp = device.type == 'cuda' and cfg.train.get('amp', True)
    # dim>=2048 时 fp16 的残差流会溢出（实测 PR_A/PR_M 分别在 step 6500/11000 变 NaN，
    # 而 GradScaler 只能拦梯度溢出，拦不住前向激活溢出）。bf16 动态范围同 fp32，无此问题。
    amp_dtype = torch.bfloat16 if cfg.train.get('amp_dtype', 'bfloat16') == 'bfloat16' \
        else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)
    log.info(f"AMP: {'关闭' if not use_amp else str(amp_dtype).replace('torch.','')}")
    ema = ModelEma(core, cfg.train.ema_decay) if cfg.train.get('ema_decay') else None

    start_step = 0
    resume_path = resolve_resume(cfg.train.get('resume'), ckpt_dir)
    if resume_path:
        start_step = load_checkpoint(resume_path, core, optimizer, scheduler, scaler, ema,
                                     map_location=device)
        log.info(f"从 {resume_path} 恢复，step={start_step}")

    wrapped = LossWrapper(core)
    if world_size > 1:
        wrapped = DDP(wrapped, device_ids=[local_rank] if device.type == 'cuda' else None)

    # ---- 训练 ----
    loss_m, masked_m = AverageMeter(), AverageMeter()
    nan_streak = n_skipped = 0
    tp = Throughput(cfg.train.max_steps)
    data_iter = infinite(train_loader, train_sampler)
    t_start = time.time()

    wrapped.train()
    for step in range(start_step, cfg.train.max_steps):
        optimizer.zero_grad(set_to_none=True)
        use_masked = bool(torch.rand(1, generator=branch_rng).item() < cfg.train.mask_ratio)

        for micro in range(accum):
            tokens = next(data_iter).to(device, non_blocking=True)
            last = (micro == accum - 1)
            # 非最后一个累积步不做梯度同步，省掉 accum-1 次 all-reduce
            sync_ctx = wrapped.no_sync() if (world_size > 1 and not last) else contextlib.nullcontext()
            with sync_ctx:
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    loss = wrapped(tokens, cfg.train.mask_ratio,
                                   cfg.train.get('mask_prob', 0.15), use_masked) / accum
                scaler.scale(loss).backward()
            loss_m.update(loss.item() * accum)
        masked_m.update(1.0 if use_masked else 0.0)

        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(core.parameters(), cfg.train.grad_clip)
        if not torch.isfinite(gnorm):
            # 丢弃这一步而不是把 NaN 写进权重
            optimizer.zero_grad(set_to_none=True)
            nan_streak += 1
            n_skipped += 1
            scaler.update()
            scheduler.step()
            if nan_streak >= int(cfg.train.get('nan_abort_after', 50)):
                log.info(f"连续 {nan_streak} 步梯度非有限，训练已发散，主动中止于 step {step+1}")
                break
            continue
        nan_streak = 0
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(core, step)

        gstep = step + 1
        if gstep % cfg.train.log_every == 0:
            rate, eta = tp.update(gstep, per_rank_bs * world_size * accum)
            log.metrics(gstep, loss=loss_m.avg, ppl=math.exp(min(loss_m.avg, 20)),
                        lr=optimizer.param_groups[0]['lr'], grad_norm=float(gnorm),
                        masked_frac=masked_m.avg, skipped=n_skipped,
                        seq_per_s=rate, eta=eta)

        if gstep % cfg.train.eval_every == 0 or gstep == cfg.train.max_steps:
            val = evaluate(core, val_loader, device, use_amp, cfg.train.eval_batches, amp_dtype)
            val = all_reduce_mean(val, device)
            log.metrics(gstep, val_loss=val, val_ppl=math.exp(min(val, 20)))

        if gstep % cfg.train.save_every == 0 or gstep == cfg.train.max_steps:
            save_checkpoint(ckpt_dir / f"step_{gstep:08d}.pt", gstep, core, optimizer,
                            scheduler, scaler, ema, cfg, keep_last=cfg.train.keep_last,
                            weights_only=cfg.train.get('save_weights_only', False))
            log.info(f"已保存 checkpoint step={gstep}")

    barrier()
    if is_main():
        save_checkpoint(ckpt_dir / 'final.pt', cfg.train.max_steps, core, optimizer,
                        scheduler, scaler, ema, cfg, keep_last=99,
                        weights_only=cfg.train.get('save_weights_only', False))
        log.info(f"训练结束，耗时 {(time.time()-t_start)/3600:.2f}h")
    log.close()
    cleanup()


if __name__ == '__main__':
    main()
