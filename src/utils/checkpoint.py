"""Checkpoint 保存/恢复。

原实现只存 model.state_dict()，无法续训。这里存全量状态，
并维护一个 latest.pt 供 --resume auto 使用，同时按 keep_last 清理旧档。
"""
from pathlib import Path

import torch

from utils.distributed import is_main, unwrap


def estimate_size_gb(model, with_optimizer=True):
    """粗估 checkpoint 体积：模型 fp32 + AdamW 两个动量。

    1.32B 的 prior 一个全量 checkpoint 约 16GB，keep_last=3 就是 48GB ——
    磁盘紧张时应调小 keep_last 或开启 save_weights_only。
    """
    n = sum(p.numel() for p in unwrap(model).parameters())
    mult = 3 if with_optimizer else 1
    return n * 4 * mult / 1e9


def save_checkpoint(path, step, model, optimizer=None, scheduler=None, scaler=None,
                    ema=None, config=None, extra=None, keep_last=3, weights_only=False):
    if not is_main():
        return
    if weights_only:
        optimizer = scheduler = scaler = None  # 只存权重，体积降到 1/3，但无法续训
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        'step': step,
        'model': unwrap(model).state_dict(),
        'config': config.to_dict() if hasattr(config, 'to_dict') else config,
    }
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    if scaler is not None:
        state['scaler'] = scaler.state_dict()
    if ema is not None:
        state['ema'] = ema.state_dict()
    if extra:
        state.update(extra)

    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)  # 原子替换，避免训练被杀时留下半个文件

    latest = path.parent / 'latest.pt'
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path.name)

    ckpts = sorted(p for p in path.parent.glob('step_*.pt'))
    for old in ckpts[:-keep_last]:
        old.unlink()


def resolve_resume(resume, ckpt_dir):
    """'auto' -> ckpt_dir/latest.pt（不存在则返回 None）；否则原样返回路径。"""
    if not resume:
        return None
    if resume != 'auto':
        return resume
    latest = Path(ckpt_dir) / 'latest.pt'
    return str(latest) if latest.exists() else None


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    ema=None, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    unwrap(model).load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    if scaler is not None and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    if ema is not None and 'ema' in ckpt:
        ema.load_state_dict(ckpt['ema'])
    return ckpt.get('step', 0)
