import torch.nn as nn
import torch.nn.functional as F  # 原实现缺失，SwiGLU 中的 F.silu 会 NameError

from models.transformer.attention import GroupedQueryAttention
from models.transformer.norm import RMSNorm

__all__ = ['RMSNorm', 'SwiGLU', 'TransformerBlock']


def swiglu_hidden(dim, expansion=None, multiple_of=256):
    """SwiGLU 的标准隐藏维是 8/3·dim（三个矩阵合计 8d²）。

    方案 A 用 expansion=4 → 三个 dim×4dim 矩阵 = 12d²，比标准多 50% 的 FLOPs，
    并把 FFN 参数占比从 67% 推到 76%。expansion 显式给出时按原样计算，
    否则用 8/3 并对齐到 multiple_of。
    """
    if expansion is not None:
        return int(dim * expansion)
    hidden = int(8 * dim / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class SwiGLU(nn.Module):
    def __init__(self, dim, expansion=None, multiple_of=256):
        super().__init__()
        hidden = swiglu_hidden(dim, expansion, multiple_of)
        self.hidden = hidden
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, head_dim, swiglu_expansion=None,
                 qk_norm=False, multiple_of=256):
        super().__init__()
        self.attn = GroupedQueryAttention(dim, n_heads, n_kv_heads, head_dim, qk_norm)
        self.ffn = SwiGLU(dim, swiglu_expansion, multiple_of)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x, attn_mask=None, causal=True, rope_cos=None, rope_sin=None,
                cache=None):
        x = x + self.attn(self.norm1(x), attn_mask, causal, rope_cos, rope_sin, cache)
        x = x + self.ffn(self.norm2(x))
        return x
