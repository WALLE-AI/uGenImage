"""阶段 1：VQGAN(实为 VQ-VAE) 训练。

方案 A 的 models/vqgan/ 没有判别器与感知损失，因此损失只有 L1 重建 + VQ。
重建会偏模糊，这是方案 A 的已知画质上限（见 SCHEME_A_BASELINE.md A.8 #1）。

用法:
    python train_vqgan.py --config configs/vqgan.yaml
    python train_vqgan.py --config configs/vqgan.yaml --set train.max_steps=1000 data.limit=2000
    torchrun --nproc_per_node=4 train_vqgan.py --config configs/vqgan.yaml
"""
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image

from config import base_parser, config_from_args, save_config
from datasets.image_dataset import ImageDataset
from models.vqgan.vqgan import VQGAN
from utils.checkpoint import (estimate_size_gb, load_checkpoint, resolve_resume,
                              save_checkpoint)
from utils.distributed import (all_reduce_mean, barrier, cleanup, get_world_size,
                               init_distributed, is_main, unwrap)
from utils.ema import ModelEma
from utils.logging import AverageMeter, Logger, Throughput
from utils.schedule import build_scheduler


def infinite(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def build_loader(cfg, split, batch_size, world_size, rank, shuffle):
    ds = ImageDataset(
        cfg.data.image_dir, cfg.data.image_size, split=split,
        val_size=cfg.data.val_size, limit=cfg.data.get('limit'),
        recursive=cfg.data.get('recursive', False), seed=cfg.train.seed,
    )
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                     shuffle=shuffle, drop_last=True)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=(shuffle and sampler is None),
        sampler=sampler, num_workers=cfg.data.num_workers, pin_memory=True,
        # 验证集不能 drop_last：样本不足一个 batch 时会静默变成 0 个 batch，
        # 指标直接报 0，看起来像“完美模型”
        drop_last=(split == 'train'), persistent_workers=cfg.data.num_workers > 0,
        prefetch_factor=4 if cfg.data.num_workers > 0 else None,
    )
    if split == 'val' and len(loader) == 0:
        raise SystemExit(f"验证集为空（{len(ds)} 张，batch={batch_size}），请调大 data.val_size")
    return ds, loader, sampler


@torch.no_grad()
def evaluate(model, loader, device, use_amp, codebook_size, max_batches=50):
    model.eval()
    rec_sum = psnr_sum = n = 0.0
    used = torch.zeros(codebook_size, dtype=torch.bool, device=device)
    for i, imgs in enumerate(loader):
        if i >= max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            recon, indices, _ = model(imgs)
        recon = recon.float()
        rec_sum += F.l1_loss(recon, imgs).item()
        mse = F.mse_loss((recon + 1) / 2, (imgs + 1) / 2).item()
        psnr_sum += 10 * math.log10(1.0 / max(mse, 1e-10))
        used[indices.reshape(-1)] = True
        n += 1
    model.train()
    n = max(n, 1)
    return rec_sum / n, psnr_sum / n, used


@torch.no_grad()
def dump_samples(model, batch, path, device, use_amp):
    model.eval()
    imgs = batch[:8].to(device)
    with torch.amp.autocast('cuda', enabled=use_amp):
        recon, _, _ = model(imgs)
    grid = torch.cat([imgs, recon.float()], dim=0)
    save_image((grid + 1) / 2, path, nrow=8, value_range=(0, 1))
    model.train()


def main():
    args = base_parser('训练 VQGAN tokenizer').parse_args()
    cfg = config_from_args(args)

    rank, world_size, local_rank = init_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.train.seed + rank)

    run_dir = Path(cfg.train.run_dir)
    ckpt_dir = run_dir / 'ckpt'
    sample_dir = run_dir / 'samples'
    log = Logger(run_dir, cfg.train.get('tensorboard', False))
    if is_main():
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, run_dir / 'config.yaml')

    # ---- 数据 ----
    per_rank_bs = cfg.data.batch_size // world_size
    if per_rank_bs < 1:
        raise SystemExit(f"batch_size({cfg.data.batch_size}) 小于进程数({world_size})")
    train_ds, train_loader, train_sampler = build_loader(cfg, 'train', per_rank_bs, world_size, rank, True)
    val_ds, val_loader, _ = build_loader(cfg, 'val', per_rank_bs, 1, 0, False)
    log.info(f"训练集 {len(train_ds):,} 张 | 验证集 {len(val_ds):,} 张 | "
             f"world_size={world_size} 每卡 batch={per_rank_bs}")

    # ---- 模型 ----
    model = VQGAN(cfg.data.image_size, cfg.model.codebook_size, cfg.model.embedding_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    wo = cfg.train.get('save_weights_only', False)
    log.info(f"VQGAN 参数量 {n_params/1e6:.1f}M | 单个 checkpoint 约 "
             f"{estimate_size_gb(model, not wo):.2f}GB × keep_last={cfg.train.keep_last}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay, betas=(0.9, 0.95))
    scheduler = build_scheduler(optimizer, cfg.train, cfg.train.max_steps)
    use_amp = device.type == 'cuda' and cfg.train.get('amp', True)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    ema = ModelEma(model, cfg.train.ema_decay) if cfg.train.get('ema_decay') else None

    start_step = 0
    resume_path = resolve_resume(cfg.train.get('resume'), ckpt_dir)
    if resume_path:
        start_step = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, ema,
                                     map_location=device)
        log.info(f"从 {resume_path} 恢复，step={start_step}")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None)

    # ---- 训练 ----
    rec_m, vq_m = AverageMeter(), AverageMeter()
    tp = Throughput(cfg.train.max_steps)
    data_iter = infinite(train_loader, train_sampler)
    used_running = torch.zeros(cfg.model.codebook_size, dtype=torch.bool, device=device)
    vis_batch = None
    t_start = time.time()

    model.train()
    for step in range(start_step, cfg.train.max_steps):
        imgs = next(data_iter).to(device, non_blocking=True)
        if vis_batch is None:
            vis_batch = imgs.detach().cpu()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            recon, indices, vq_loss = model(imgs)
            rec_loss = F.l1_loss(recon, imgs)
            loss = rec_loss + vq_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model, step)

        rec_m.update(rec_loss.item())
        vq_m.update(vq_loss.item())
        used_running[indices.reshape(-1)] = True

        gstep = step + 1
        if gstep % cfg.train.log_every == 0:
            rate, eta = tp.update(gstep, cfg.data.batch_size)
            usage = used_running.sum().item() / cfg.model.codebook_size
            log.metrics(gstep,
                        rec=rec_m.avg, vq=vq_m.avg,
                        lr=optimizer.param_groups[0]['lr'],
                        grad_norm=float(gnorm),
                        codebook_usage=usage,
                        img_per_s=rate, eta=eta)
            used_running.zero_()  # 使用率按窗口统计，累计值会一直单调上升失去意义

        if gstep % cfg.train.eval_every == 0 or gstep == cfg.train.max_steps:
            eval_model = unwrap(model)
            rec, psnr, used = evaluate(eval_model, val_loader, device, use_amp,
                                       cfg.model.codebook_size, cfg.train.eval_batches)
            rec = all_reduce_mean(rec, device)
            psnr = all_reduce_mean(psnr, device)
            log.metrics(gstep, val_rec=rec, val_psnr=psnr,
                        val_codebook_usage=used.sum().item() / cfg.model.codebook_size)
            if is_main() and vis_batch is not None:
                dump_samples(eval_model, vis_batch, sample_dir / f"recon_{gstep:08d}.png",
                             device, use_amp)

        if gstep % cfg.train.save_every == 0 or gstep == cfg.train.max_steps:
            save_checkpoint(ckpt_dir / f"step_{gstep:08d}.pt", gstep, model, optimizer,
                            scheduler, scaler, ema, cfg, keep_last=cfg.train.keep_last,
                            weights_only=cfg.train.get('save_weights_only', False))
            log.info(f"已保存 checkpoint step={gstep}")

    barrier()
    if is_main():
        save_checkpoint(ckpt_dir / 'final.pt', cfg.train.max_steps, model, optimizer,
                        scheduler, scaler, ema, cfg, keep_last=99,
                        weights_only=cfg.train.get('save_weights_only', False))
        log.info(f"训练结束，耗时 {(time.time()-t_start)/3600:.2f}h")
    log.close()
    cleanup()


if __name__ == '__main__':
    main()
