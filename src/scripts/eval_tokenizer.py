"""Tokenizer 评估：一次性输出对照实验需要的全部指标。

用法:
    python scripts/eval_tokenizer.py --ckpt runs/vqgan_full/ckpt/final.pt
    python scripts/eval_tokenizer.py --ckpt A.pt --ckpt B.pt --tag E0 --tag E1

指标（对应 OPTIMIZATION_PLAN.md 第 1 节诊断）:
  码字使用率 / 码本熵 / 每图信息量   -> 信息瓶颈
  码本范数(用/未用) / 量化误差       -> 是否存在死锁
  PSNR / SSIM / 低频高频误差分解      -> 重建质量与模糊程度
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision.utils import save_image  # noqa: E402

from datasets.image_dataset import ImageDataset  # noqa: E402
from models.vqgan.vqgan import build_vqgan_from_config  # noqa: E402


def ssim(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    """在 [0,1] 值域上按 11x11 均值窗口估算 SSIM（够用的近似，不引入额外依赖）。"""
    mu_x = F.avg_pool2d(x, 11, 1, 5)
    mu_y = F.avg_pool2d(y, 11, 1, 5)
    sxx = F.avg_pool2d(x * x, 11, 1, 5) - mu_x ** 2
    syy = F.avg_pool2d(y * y, 11, 1, 5) - mu_y ** 2
    sxy = F.avg_pool2d(x * y, 11, 1, 5) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sxy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2)
    return (num / den).mean().item()


@torch.no_grad()
def evaluate(ckpt_path, image_dir, n_images, batch_size, num_workers, device, sample_out=None):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {}) or {}
    model = build_vqgan_from_config(cfg).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()

    image_size = cfg.get('data', {}).get('image_size', 256)
    ds = ImageDataset(image_dir, image_size, split='val', val_size=n_images,
                      limit=n_images, augment=False)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

    q = model.codebook
    K = q.num_codewords
    cnt = Counter()
    psnr_s = ssim_s = lo_s = hi_s = l1_s = 0.0
    quant_err = 0.0
    n = 0
    first = None

    for imgs in dl:
        imgs = imgs.to(device, non_blocking=True)
        z_e = model.encoder(imgs)
        z_q, idx, _ = q(z_e)
        rec = model.decoder(z_q).float()
        if first is None:
            first = (imgs[:8].cpu(), rec[:8].cpu())

        cnt.update(idx.reshape(-1).cpu().tolist())
        l1_s += F.l1_loss(rec, imgs).item()

        x01, r01 = (imgs + 1) / 2, (rec + 1) / 2
        mse = F.mse_loss(r01, x01).item()
        psnr_s += 10 * math.log10(1.0 / max(mse, 1e-10))
        ssim_s += ssim(x01.clamp(0, 1), r01.clamp(0, 1))

        # 低频 = 8x 下采样再上采样；高频 = 残差
        lo_x = F.interpolate(F.avg_pool2d(imgs, 8), scale_factor=8, mode='nearest')
        lo_r = F.interpolate(F.avg_pool2d(rec, 8), scale_factor=8, mode='nearest')
        lo_s += F.l1_loss(lo_r, lo_x).item()
        hi_s += F.l1_loss(rec - lo_r, imgs - lo_x).item()

        zp = q.proj_in(z_e) if hasattr(q, 'proj_in') else z_e
        if getattr(q, 'l2_norm', False):
            zp = F.normalize(zp, dim=1)
        zq_code = q.codebook_weight()[idx.view(zp.shape[0], zp.shape[2], zp.shape[3])]
        zq_code = zq_code.permute(0, 3, 1, 2)
        if getattr(q, 'l2_norm', False):
            zq_code = F.normalize(zq_code, dim=1)
        quant_err += ((zq_code - zp).norm(dim=1).mean() / (zp.norm(dim=1).mean() + 1e-8)).item()
        n += 1

    tot = sum(cnt.values())
    H = -sum((c / tot) * math.log2(c / tot) for c in cnt.values())
    used = sorted(cnt)
    unused = [i for i in range(K) if i not in cnt]
    cb = q.codebook_weight().detach().cpu()
    norm_used = cb[used].norm(dim=1).mean().item() if used else 0.0
    norm_unused = cb[unused].norm(dim=1).mean().item() if unused else float('nan')
    n_tokens = (image_size // 16) ** 2

    if sample_out and first is not None:
        Path(sample_out).parent.mkdir(parents=True, exist_ok=True)
        save_image((torch.cat(first) + 1) / 2, sample_out, nrow=8, value_range=(0, 1))

    return {
        'ckpt': str(ckpt_path),
        'step': ckpt.get('step'),
        'codebook_size': K,
        'used': len(used),
        'usage': len(used) / K,
        'entropy_bits': H,
        'bytes_per_image': H * n_tokens / 8,
        'bytes_if_full': math.log2(K) * n_tokens / 8,
        'norm_used': norm_used,
        'norm_unused': norm_unused,
        'norm_ratio': (norm_used / norm_unused) if norm_unused and norm_unused > 0 else float('inf'),
        'quant_err_rel': quant_err / n,
        'l1': l1_s / n,
        'psnr': psnr_s / n,
        'ssim': ssim_s / n,
        'err_lo': lo_s / n,
        'err_hi': hi_s / n,
        'hi_frac': hi_s / (lo_s + hi_s),
        'n_images': len(ds),
    }


def show(r, tag):
    print(f"\n===== {tag} =====")
    print(f"  checkpoint      {r['ckpt']}  (step {r['step']})")
    print(f"  码字使用        {r['used']} / {r['codebook_size']} = {r['usage']:.2%}")
    print(f"  码本熵          {r['entropy_bits']:.2f} bit/token  "
          f"(满用上限 {math.log2(r['codebook_size']):.2f})")
    print(f"  每图信息量      {r['bytes_per_image']:.0f} 字节  "
          f"(满用可达 {r['bytes_if_full']:.0f})")
    if r['used'] == r['codebook_size']:
        print(f"  码本范数        用 {r['norm_used']:.3f} / 无未用码字   <- 无死锁")
    else:
        flag = "  <- 比值大 = 死锁" if r['norm_ratio'] > 10 else "  <- 比值接近 1 = 无死锁"
        print(f"  码本范数        用 {r['norm_used']:.3f} / 未用 {r['norm_unused']:.3f} "
              f"= {r['norm_ratio']:.1f}x{flag}")
    print(f"  量化误差        {r['quant_err_rel']:.1%} (量化空间内，相对输入范数)")
    print(f"  重建 L1         {r['l1']:.4f}")
    print(f"  PSNR / SSIM     {r['psnr']:.2f} dB / {r['ssim']:.4f}")
    print(f"  误差分解        低频 {r['err_lo']:.4f} | 高频 {r['err_hi']:.4f} "
          f"(高频占 {r['hi_frac']:.0%})  <- 高频占比大 = 模糊")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', action='append', required=True, help='可重复，用于多组对照')
    p.add_argument('--tag', action='append', default=None, help='可重复，与 --ckpt 对应')
    p.add_argument('--image_dir',
                   default='/home/dataset0/images/ALLaVA-4V/allava_laion/image_chunks/images')
    p.add_argument('--n_images', type=int, default=1024)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--sample_dir', default=None, help='落盘重建对比图的目录')
    p.add_argument('--json_out', default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    tags = args.tag or [Path(c).parent.parent.name for c in args.ckpt]
    results = []
    for ck, tag in zip(args.ckpt, tags):
        out = f"{args.sample_dir}/{tag}.png" if args.sample_dir else None
        r = evaluate(ck, args.image_dir, args.n_images, args.batch_size,
                     args.num_workers, torch.device(args.device), out)
        r['tag'] = tag
        results.append(r)
        show(r, tag)

    if len(results) > 1:
        print("\n===== 对照汇总 =====")
        print(f"{'tag':<10}{'usage':>9}{'熵(bit)':>10}{'字节/图':>10}"
              f"{'PSNR':>8}{'SSIM':>8}{'高频占比':>10}")
        for r in results:
            print(f"{r['tag']:<10}{r['usage']:>8.2%}{r['entropy_bits']:>10.2f}"
                  f"{r['bytes_per_image']:>10.0f}{r['psnr']:>8.2f}{r['ssim']:>8.4f}"
                  f"{r['hi_frac']:>9.0%}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\n已写出 {args.json_out}")


if __name__ == '__main__':
    main()
