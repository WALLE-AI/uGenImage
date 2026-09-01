"""按 step 计的学习率调度。

原实现建了 CosineAnnealingLR(T_max=epochs) 但全仓库没有一处 .step()，
学习率恒为初值且无 warmup。这里统一为按 step 的 warmup + cosine。
"""
import math

import torch.optim as optim


def build_warmup_cosine(optimizer, warmup_steps, total_steps, min_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_scheduler(optimizer, cfg, total_steps):
    """cfg 需含 warmup_steps / lr_schedule('cosine'|'constant') / min_lr_ratio。"""
    kind = cfg.get('lr_schedule', 'cosine')
    warmup = int(cfg.get('warmup_steps', 0))
    if kind == 'constant':
        return optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: (s + 1) / max(1, warmup) if s < warmup else 1.0
        )
    if kind == 'cosine':
        return build_warmup_cosine(optimizer, warmup, total_steps,
                                   float(cfg.get('min_lr_ratio', 0.1)))
    raise ValueError(f"未知 lr_schedule: {kind}")


def param_groups_no_decay(model, weight_decay):
    """norm / bias / embedding 不做权重衰减 —— 对它们衰减通常有害。"""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or 'norm' in name or 'embedding' in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}]
