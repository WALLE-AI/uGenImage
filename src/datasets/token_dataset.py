"""Token 序列数据集。

原实现每次启动都 glob 全目录，且没有 train/val 切分。
48 万个 .npy（每个约 1KB）的场景下，清单缓存与分片打包都是必要的，
这里先做清单缓存 + 切分；打包成 shard 见 PLAN.md P2-2。
"""
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from constants import PAD_ID, SEQ_LEN
from datasets.image_dataset import DEFAULT_CACHE

import hashlib
from pathlib import Path


def scan_tokens(data_dir, cache_dir=DEFAULT_CACHE, use_cache=True):
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"token 目录不存在: {data_dir}")
    key = hashlib.md5(str(data_dir.resolve()).encode()).hexdigest()[:16]
    cache_file = Path(cache_dir) / f"tokenlist_{key}.txt"
    if use_cache and cache_file.exists():
        paths = cache_file.read_text(encoding='utf-8').splitlines()
        if paths:
            return paths
    paths = sorted(str(p) for p in data_dir.glob('*.npy'))
    if use_cache and paths:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text('\n'.join(paths), encoding='utf-8')
    return paths


class TokenDataset(Dataset):
    def __init__(self, data_dir, seq_len=SEQ_LEN, split='train', val_size=2000,
                 limit=None, seed=0, use_cache=True):
        paths = scan_tokens(data_dir, use_cache=use_cache)
        if not paths:
            raise SystemExit(f"{data_dir} 中没有 .npy，请先运行 preencode.py")

        rng = random.Random(seed)
        rng.shuffle(paths)
        val_size = min(val_size, max(0, len(paths) // 10))
        if split == 'train':
            paths = paths[val_size:]
        elif split == 'val':
            paths = paths[:val_size]
        else:
            raise ValueError(f"未知 split: {split}")
        if limit:
            paths = paths[:limit]

        self.split = split
        self.files = paths
        self.seq_len = seq_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 必须显式越界报错：下面的坏文件回退用了取模，若不检查，
        # `for x in dataset` 这种基于 IndexError 的迭代会永不终止
        if not -len(self.files) <= idx < len(self.files):
            raise IndexError(idx)
        for offset in range(8):
            j = (idx + offset) % len(self.files)
            try:
                seq = np.load(self.files[j])
                break
            except Exception:
                print(f"[TokenDataset] 跳过损坏文件: {self.files[j]}", flush=True)
        else:
            seq = np.full(self.seq_len, PAD_ID, dtype=np.int32)

        if len(seq) > self.seq_len:
            seq = seq[:self.seq_len]
        elif len(seq) < self.seq_len:
            seq = np.pad(seq, (0, self.seq_len - len(seq)), constant_values=PAD_ID)
        return torch.from_numpy(seq.astype(np.int64))
