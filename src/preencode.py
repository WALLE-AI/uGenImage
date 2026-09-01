"""阶段 2：用训练好的 VQGAN 把图片预编码为 token 序列 (.npy)。

原实现在主进程里串行解码 JPEG（约 70 img/s），48 万张要近 2 小时；
这里改用 DataLoader 多进程解码，瓶颈回到 GPU。
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from constants import BOS_ID, CODEBOOK_OFFSET
from datasets.image_dataset import IMAGE_EXTS, build_transform, scan_images
from models.vqgan.vqgan import build_vqgan_from_config


class _EncodeDataset(Dataset):
    """返回 (图像张量, 输出文件名)。中心裁剪，不做任何增强。"""

    def __init__(self, paths, image_size):
        self.paths = paths
        self.transform = build_transform(image_size, augment=False)
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if not -len(self.paths) <= idx < len(self.paths):
            raise IndexError(idx)
        p = Path(self.paths[idx])
        try:
            return self.transform(Image.open(p).convert('RGB')), p.stem, True
        except Exception:
            print(f"[preencode] 跳过损坏文件: {p}", flush=True)
            return torch.zeros(3, self.image_size, self.image_size), p.stem, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqgan_ckpt', required=True)
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--overwrite', action='store_true',
                        help='默认跳过已存在的输出，便于断点续跑')
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.vqgan_ckpt, map_location='cpu', weights_only=False)
    # 模型结构直接取自 checkpoint 里的配置，避免与训练时的超参手工对不上
    ck_cfg = ckpt.get('config', {}) or {}
    image_size = ck_cfg.get('data', {}).get('image_size', args.image_size)
    model = build_vqgan_from_config(ck_cfg).to(device)
    # 原实现用 strict=False，checkpoint 不匹配时会静默地用随机码本编码整个数据集
    model.load_state_dict(ckpt.get('model', ckpt), strict=True)
    model.eval()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    paths = scan_images(args.image_dir, args.recursive)
    if not paths:
        raise SystemExit(f"{args.image_dir} 下没有找到图片（支持 {', '.join(IMAGE_EXTS)}）")
    if args.limit:
        paths = paths[:args.limit]
    if not args.overwrite:
        paths = [p for p in paths if not (out_path / (Path(p).stem + '.npy')).exists()]
    print(f"待编码图片: {len(paths):,} (image_size={image_size})")
    if not paths:
        return

    loader = DataLoader(_EncodeDataset(paths, image_size), batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)

    n_ok = n_bad = 0
    for imgs, stems, ok in tqdm(loader):
        with torch.no_grad():
            _, indices, _ = model.encode(imgs.to(device, non_blocking=True))  # [B,H,W]
        indices = indices.flatten(1).cpu().numpy()
        for stem, seq, good in zip(stems, indices, ok.tolist()):
            if not good:
                n_bad += 1
                continue
            # 序列头部写入 BOS，与推理起点保持一致（原实现不写 BOS，推理第一步 OOD）
            seq = np.concatenate([[BOS_ID], seq + CODEBOOK_OFFSET])
            np.save(out_path / (stem + '.npy'), seq.astype(np.int32))
            n_ok += 1
    print(f"完成: 写出 {n_ok:,} 个 .npy，跳过损坏 {n_bad}")


if __name__ == '__main__':
    main()
