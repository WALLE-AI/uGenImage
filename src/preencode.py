import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from constants import BOS_ID, CODEBOOK_OFFSET
from models.vqgan.vqgan import VQGAN

IMAGE_EXTS = ('*.jpg', '*.jpeg', '*.png')  # 原实现只收 *.jpg，与 ImageDataset 不一致


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqgan_ckpt', required=True)
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--overwrite', action='store_true', help='默认跳过已存在的输出，便于断点续跑')
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.vqgan_ckpt, map_location='cpu', weights_only=False)
    # 模型结构直接取自 checkpoint 里的配置，避免与训练时的超参手工对不上
    ck_cfg = ckpt.get('config', {}) or {}
    image_size = ck_cfg.get('data', {}).get('image_size', args.image_size)
    model = VQGAN(
        image_size,
        ck_cfg.get('model', {}).get('codebook_size', 1024),
        ck_cfg.get('model', {}).get('embedding_dim', 256),
    ).to(device)
    args.image_size = image_size
    state = ckpt.get('model', ckpt)
    # 原实现用 strict=False，checkpoint 不匹配时会静默地用随机码本编码整个数据集
    model.load_state_dict(state, strict=True)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(p for ext in IMAGE_EXTS for p in Path(args.image_dir).glob(ext))
    if not args.overwrite:
        img_paths = [p for p in img_paths if not (out_path / (p.stem + '.npy')).exists()]
    print(f"待编码图片: {len(img_paths)}")

    for i in tqdm(range(0, len(img_paths), args.batch_size)):
        chunk = img_paths[i:i + args.batch_size]
        batch = torch.stack([transform(Image.open(p).convert('RGB')) for p in chunk]).to(device)
        with torch.no_grad():
            _, indices, _ = model.encode(batch)          # [B, H, W]
        indices = indices.flatten(1).cpu().numpy()       # [B, H*W]
        for p, seq in zip(chunk, indices):
            # 序列头部写入 BOS，与推理起点保持一致（原实现不写 BOS，导致推理第一步 OOD）
            seq = np.concatenate([[BOS_ID], seq + CODEBOOK_OFFSET])
            np.save(out_path / (p.stem + '.npy'), seq.astype(np.int32))


if __name__ == '__main__':
    main()
