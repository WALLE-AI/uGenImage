"""采样入口。原方案第 5 步只说“参考 inference.py”，没有可执行入口。"""
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from inference import generate_autoregressive
from models.transformer.model import VisualTransformer
from models.vqgan.vqgan import VQGAN

DOWNSAMPLE = 16  # Encoder 4 级 stride-2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--transformer_ckpt', required=True)
    p.add_argument('--vqgan_ckpt', required=True)
    p.add_argument('--output', default='outputs/sample.png')
    p.add_argument('--n_samples', type=int, default=4)
    p.add_argument('--n_tokens', type=int, default=None,
                   help='默认按 VQGAN 的 image_size 推导 (image_size/16)^2')
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--use_ema', action='store_true', help='用 EMA 权重采样（若 checkpoint 中存在）')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device(args.device)

    t_ckpt = torch.load(args.transformer_ckpt, map_location='cpu', weights_only=False)
    t_cfg = t_ckpt['config']
    model = VisualTransformer(t_cfg.get('model', t_cfg)).to(device)
    state = t_ckpt['ema'] if (args.use_ema and 'ema' in t_ckpt) else t_ckpt['model']
    model.load_state_dict(state)
    model.eval()

    v_ckpt = torch.load(args.vqgan_ckpt, map_location='cpu', weights_only=False)
    v_cfg = v_ckpt.get('config', {}) or {}
    image_size = v_cfg.get('data', {}).get('image_size', 256)
    codebook_size = v_cfg.get('model', {}).get('codebook_size', 1024)
    vqgan = VQGAN(image_size, codebook_size,
                  v_cfg.get('model', {}).get('embedding_dim', 256)).to(device)
    vqgan.load_state_dict(v_ckpt.get('model', v_ckpt), strict=True)
    vqgan.eval()

    n_tokens = args.n_tokens or (image_size // DOWNSAMPLE) ** 2
    print(f"image_size={image_size} codebook={codebook_size} n_tokens={n_tokens}")

    imgs = [generate_autoregressive(model, vqgan, n_tokens, args.temperature,
                                    args.top_k, codebook_size)
            for _ in range(args.n_samples)]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.stack(imgs), out, nrow=min(4, args.n_samples))
    print(f"已保存: {out}")


if __name__ == '__main__':
    main()
