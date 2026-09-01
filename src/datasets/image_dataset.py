"""图像数据集。

相对原实现的改动，都是 48 万张规模下才会暴露的问题：
- 文件清单缓存（481k 张时每次启动 glob 约 7s，续训/多进程会重复付出）
- 损坏图片不再让整轮训练崩掉（48 万张里出现坏文件是常态）
- train/val 切分
- 支持递归扫描与 limit（快速试跑）
"""
import hashlib
import os
import random
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
DEFAULT_CACHE = Path(os.environ.get('UGEN_CACHE', Path.home() / '.cache' / 'ugenimage'))


def scan_images(root, recursive=False, cache_dir=DEFAULT_CACHE, use_cache=True):
    """扫描图片路径。结果按 (root, recursive) 缓存到本地，避免重复 glob。"""
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"图片目录不存在: {root}")

    key = hashlib.md5(f"{root.resolve()}|{recursive}".encode()).hexdigest()[:16]
    cache_file = Path(cache_dir) / f"filelist_{key}.txt"
    if use_cache and cache_file.exists():
        paths = cache_file.read_text(encoding='utf-8').splitlines()
        if paths:
            return paths

    it = root.rglob('*') if recursive else root.glob('*')
    paths = sorted(str(p) for p in it if p.suffix.lower() in IMAGE_EXTS)
    if use_cache and paths:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text('\n'.join(paths), encoding='utf-8')
    return paths


def build_transform(image_size, augment=True):
    ops = [T.Resize(image_size), T.CenterCrop(image_size)]
    if augment:
        # 仅水平翻转：离散 tokenizer 对颜色/几何增强很敏感，不要加过头
        ops = [T.Resize(image_size), T.RandomCrop(image_size, pad_if_needed=True),
               T.RandomHorizontalFlip()]
    ops += [T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    return T.Compose(ops)


class ImageDataset(Dataset):
    def __init__(self, root_dir, image_size=256, split='train', val_size=2000,
                 limit=None, recursive=False, augment=None, seed=0, use_cache=True):
        paths = scan_images(root_dir, recursive, use_cache=use_cache)
        if not paths:
            raise SystemExit(f"{root_dir} 下没有找到图片（支持 {', '.join(IMAGE_EXTS)}）")

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
        self.image_size = image_size
        self.image_paths = paths
        self.transform = build_transform(image_size, augment=(split == 'train') if augment is None else augment)
        self._bad = set()

    def __len__(self):
        return len(self.image_paths)

    def _load(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)

    def __getitem__(self, idx):
        # 必须显式越界报错：下面的坏图回退用了取模，若不检查，
        # `for x in dataset` 这种基于 IndexError 的迭代会永不终止
        if not -len(self.image_paths) <= idx < len(self.image_paths):
            raise IndexError(idx)
        # 48 万张里必然有坏文件；一张坏图不应该让训练在第 N 小时崩掉
        for offset in range(8):
            j = (idx + offset) % len(self.image_paths)
            try:
                return self._load(j)
            except Exception:
                if j not in self._bad:
                    self._bad.add(j)
                    print(f"[ImageDataset] 跳过损坏文件: {self.image_paths[j]}", flush=True)
        # 连续 8 张都读不出来时返回黑图，保证训练不中断（日志里已有告警）
        return torch.zeros(3, self.image_size, self.image_size)
