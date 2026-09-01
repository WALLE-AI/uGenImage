"""打印 Transformer prior 的真实参数量分解。

方案 A 标称 0.39B，实测约 1.32B（偏差 3.4x），原因见 SCHEME_A_BASELINE.md A.3。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from constants import SEQ_LEN, VOCAB_SIZE  # noqa: E402
from models.transformer.model import VisualTransformer  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dim', type=int, default=2048)
    p.add_argument('--n_layers', type=int, default=20)
    p.add_argument('--n_heads', type=int, default=24)
    p.add_argument('--n_kv_heads', type=int, default=6)
    p.add_argument('--head_dim', type=int, default=128)
    p.add_argument('--vocab_size', type=int, default=VOCAB_SIZE)
    p.add_argument('--seq_len', type=int, default=SEQ_LEN)
    args = p.parse_args()

    with torch.device('meta'):
        model = VisualTransformer(vars(args))

    groups = {'embedding': 0, 'attention': 0, 'ffn': 0, 'norm': 0, 'other': 0}
    seen = set()
    for name, prm in model.named_parameters():
        if id(prm) in seen:      # lm_head 与 embedding 绑定，避免重复计数
            continue
        seen.add(id(prm))
        if 'token_embedding' in name or 'lm_head' in name:
            groups['embedding'] += prm.numel()
        elif '.attn.' in name:
            groups['attention'] += prm.numel()
        elif '.ffn.' in name:
            groups['ffn'] += prm.numel()
        elif 'norm' in name:
            groups['norm'] += prm.numel()
        else:
            groups['other'] += prm.numel()

    total = sum(groups.values())
    print(f"dim={args.dim} layers={args.n_layers} heads={args.n_heads} "
          f"kv_heads={args.n_kv_heads} head_dim={args.head_dim} vocab={args.vocab_size}")
    print(f"注意: n_heads*head_dim = {args.n_heads*args.head_dim}, dim = {args.dim}")
    for k, v in groups.items():
        print(f"  {k:10s} {v/1e6:9.2f}M  ({v/total:5.1%})")
    print(f"  {'TOTAL':10s} {total/1e6:9.2f}M  ({total/1e9:.3f}B)")


if __name__ == '__main__':
    main()
