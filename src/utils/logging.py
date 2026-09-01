"""训练日志：同时写终端与文件，可选 TensorBoard。"""
import json
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

from utils.distributed import is_main


class AverageMeter:
    """滑动窗口均值，避免整轮平均把早期的异常值一直带着。"""

    def __init__(self, window=100):
        self.vals = deque(maxlen=window)

    def update(self, v):
        self.vals.append(float(v))

    @property
    def avg(self):
        return sum(self.vals) / len(self.vals) if self.vals else 0.0


class Logger:
    def __init__(self, run_dir, use_tensorboard=False):
        self.run_dir = Path(run_dir)
        self.enabled = is_main()
        self.writer = None
        self.jsonl = None
        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = open(self.run_dir / 'log.txt', 'a', encoding='utf-8')
            self.jsonl = open(self.run_dir / 'metrics.jsonl', 'a', encoding='utf-8')
            if use_tensorboard:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self.writer = SummaryWriter(str(self.run_dir / 'tb'))
                except ImportError:
                    self.info('未安装 tensorboard，跳过 TB 日志')

    def info(self, msg):
        if not self.enabled:
            return
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        self.log_file.write(line + '\n')
        self.log_file.flush()

    def metrics(self, step, **kv):
        if not self.enabled:
            return
        parts = []
        for k, v in kv.items():
            parts.append(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}")
        self.info(f"step {step} | " + ' | '.join(parts))
        self.jsonl.write(json.dumps({'step': step, **kv}) + '\n')
        self.jsonl.flush()
        if self.writer:
            for k, v in kv.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, step)

    def close(self):
        if not self.enabled:
            return
        self.log_file.close()
        self.jsonl.close()
        if self.writer:
            self.writer.close()


class Throughput:
    """统计 样本/秒 与剩余时间估计。"""

    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.t0 = time.time()
        self.last_t = self.t0
        self.last_step = 0

    def update(self, step, samples):
        now = time.time()
        dt = max(1e-6, now - self.last_t)
        steps = max(1, step - self.last_step)
        rate = samples * steps / dt
        remaining = (self.total_steps - step) * dt / steps
        self.last_t, self.last_step = now, step
        return rate, str(timedelta(seconds=int(remaining)))
