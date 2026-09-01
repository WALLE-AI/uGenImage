"""参数指数滑动平均。采样时用 EMA 权重通常明显好于原始权重。"""
import copy

import torch

from utils.distributed import unwrap


class ModelEma:
    def __init__(self, model, decay=0.9999, warmup_steps=0):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.module = copy.deepcopy(unwrap(model)).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def _cur_decay(self, step):
        if self.warmup_steps and step < self.warmup_steps:
            # 训练初期参数变化剧烈，用较小的 decay 让 EMA 跟得上
            return min(self.decay, (1 + step) / (10 + step))
        return self.decay

    @torch.no_grad()
    def update(self, model, step):
        d = self._cur_decay(step)
        msd = unwrap(model).state_dict()
        for k, v in self.module.state_dict().items():
            src = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src.detach(), alpha=1 - d)
            else:
                v.copy_(src)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)
