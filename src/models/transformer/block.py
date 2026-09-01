import torch
import torch.nn as nn
import torch.nn.functional as F  # 原实现缺失，SwiGLU 中的 F.silu 会 NameError

from models.transformer.attention import GroupedQueryAttention


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 在 fp32 下计算，避免 AMP fp16 时 x.pow(2) 溢出
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * expansion, bias=False)
        self.w2 = nn.Linear(dim, dim * expansion, bias=False)
        self.w3 = nn.Linear(dim * expansion, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, head_dim, swiglu_expansion=4):
        super().__init__()
        self.attn = GroupedQueryAttention(dim, n_heads, n_kv_heads, head_dim)
        self.ffn = SwiGLU(dim, swiglu_expansion)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x, attn_mask=None, causal=True, rope_cos=None, rope_sin=None):
        x = x + self.attn(self.norm1(x), attn_mask, causal, rope_cos, rope_sin)
        x = x + self.ffn(self.norm2(x))
        return x
