"""Prior 对照评估：在同一份验证集上比较多个 prior checkpoint。

用法:
    python scripts/eval_prior.py --ckpt runs/A/ckpt/final.pt --tag A \
                                 --ckpt runs/P1-M/ckpt/final.pt --tag P1-M \
                                 --vqgan_ckpt runs/E4/ckpt/final.pt --sample_dir runs/_prior_eval

指标:
    val_loss / ppl      因果目标下的交叉熵与困惑度（唯一可直接横比的量）
    ppl_norm            ppl / 2^(码本熵)，消除 tokenizer 差异后的相对难度
    token 分布覆盖率     采样结果用到的码字比例，低说明生成塌缩
    参数量 / 吞吐        同等步数下的算力代价
"""
import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision.utils import save_image  # noqa: E402

from datasets.token_dataset import TokenDataset  # noqa: E402
from inference import generate_tokens  # noqa: E402
from models.transformer.model import VisualTransformer  # noqa: E402
from models.vqgan.vqgan import build_vqgan_from_config  # noqa: E402
from constants import CODEBOOK_OFFSET  # noqa: E402


@torch.no_grad()
def eval_one(ckpt_path, token_dir, device, n_batches, batch_size, vqgan=None,
             n_samples=0, sample_out=None, temperature=1.0, top_k=100, seed=0):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    mcfg = cfg.get('model', cfg)
    model = VisualTransformer(mcfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    ds = TokenDataset(token_dir, mcfg['seq_len'], split='val',
                      val_size=cfg.get('data', {}).get('val_size', 2000), seed=0)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4)

    tot, n = 0.0, 0
    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        tot += model._causal_loss(batch.to(device)).item()
        n += 1
    val_loss = tot / max(n, 1)

    n_params = sum(p.numel() for p in model.parameters())
    out = {
        'ckpt': str(ckpt_path), 'step': ckpt.get('step'),
        'params_B': n_params / 1e9,
        'dim': mcfg['dim'], 'n_layers': mcfg['n_layers'],
        'n_heads': mcfg['n_heads'], 'n_kv_heads': mcfg['n_kv_heads'],
        'head_dim': mcfg['head_dim'], 'rope': mcfg.get('rope', '1d'),
        'qk_norm': mcfg.get('qk_norm', False),
        'swiglu_expansion': mcfg.get('swiglu_expansion'),
        'tied': mcfg.get('tie_embeddings', True),
        'val_loss': val_loss, 'val_ppl': math.exp(min(val_loss, 20)),
        'n_val': len(ds),
    }

    if n_samples and vqgan is not None:
        torch.manual_seed(seed)
        t0 = time.time()
        toks = generate_tokens(model, n_tokens=mcfg['seq_len'] - 1,
                               temperature=temperature, top_k=top_k,
                               batch_size=n_samples,
                               codebook_size=vqgan.codebook.num_codewords)
        out['sample_sec'] = time.time() - t0
        cnt = Counter(toks.reshape(-1).cpu().tolist())
        out['sample_code_coverage'] = len(cnt) / vqgan.codebook.num_codewords
        if sample_out:
            codes = (toks - CODEBOOK_OFFSET).clamp_(0, vqgan.codebook.num_codewords - 1)
            g = int(round((mcfg['seq_len'] - 1) ** 0.5))
            img = vqgan.decode_code(codes.view(-1, g, g))
            Path(sample_out).parent.mkdir(parents=True, exist_ok=True)
            save_image(((img + 1) / 2).clamp(0, 1), sample_out, nrow=4)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', action='append', required=True)
    p.add_argument('--tag', action='append', default=None)
    p.add_argument('--token_dir', default='data/tokens_e4')
    p.add_argument('--vqgan_ckpt', default=None, help='给出后会额外采样并落盘图片')
    p.add_argument('--n_samples', type=int, default=8)
    p.add_argument('--n_batches', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=100)
    p.add_argument('--sample_dir', default=None)
    p.add_argument('--json_out', default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    vqgan = None
    if args.vqgan_ckpt:
        vck = torch.load(args.vqgan_ckpt, map_location='cpu', weights_only=False)
        vqgan = build_vqgan_from_config(vck.get('config', {})).to(device)
        vqgan.load_state_dict(vck['model'], strict=True)
        vqgan.eval()

    tags = args.tag or [Path(c).parent.parent.name for c in args.ckpt]
    rows = []
    for ck, tag in zip(args.ckpt, tags):
        out = f"{args.sample_dir}/{tag}.png" if args.sample_dir else None
        r = eval_one(ck, args.token_dir, device, args.n_batches, args.batch_size,
                     vqgan, args.n_samples if vqgan else 0, out,
                     args.temperature, args.top_k)
        r['tag'] = tag
        rows.append(r)
        print(f"\n===== {tag} =====")
        print(f"  {r['params_B']:.3f}B | dim {r['dim']} x {r['n_layers']}层 | "
              f"heads {r['n_heads']}/{r['n_kv_heads']} x {r['head_dim']} | "
              f"rope {r['rope']} | qk_norm {r['qk_norm']} | "
              f"swiglu {r['swiglu_expansion'] or '8/3'} | tied {r['tied']}")
        print(f"  val_loss {r['val_loss']:.4f} | val_ppl {r['val_ppl']:.2f}")
        if 'sample_code_coverage' in r:
            print(f"  采样码字覆盖率 {r['sample_code_coverage']:.1%} | "
                  f"采样耗时 {r['sample_sec']:.1f}s")

    print("\n===== 对照汇总 =====")
    print(f"{'tag':<10}{'参数':>9}{'dim/层':>10}{'val_loss':>10}{'val_ppl':>10}{'采样覆盖':>10}")
    for r in sorted(rows, key=lambda x: x['val_loss']):
        print(f"{r['tag']:<10}{r['params_B']:>8.3f}B{r['dim']:>7}/{r['n_layers']:<3}"
              f"{r['val_loss']:>10.4f}{r['val_ppl']:>10.2f}"
              f"{r.get('sample_code_coverage', float('nan')):>9.1%}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"\n已写出 {args.json_out}")


if __name__ == '__main__':
    main()
