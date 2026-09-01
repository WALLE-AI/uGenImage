import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope(head_dim, max_seq_len, theta=10000.0, device=None, dtype=torch.float32):
    """预计算 RoPE 的 cos/sin 表。

    原实现只定义了 apply_rotary_pos_emb，却没有任何地方生成 cos/sin，
    导致位置编码实际从未生效。

    返回: cos, sin —— 形状均为 [1, 1, max_seq_len, head_dim]
    """
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)               # [T, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)        # [T, head_dim]，与 rotate_half 的半分切法一致
    return emb.cos()[None, None].to(dtype), emb.sin()[None, None].to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, n_heads=24, n_kv_heads=6, head_dim=128):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads 必须能被 n_kv_heads 整除"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x, attn_mask=None, causal=True, rope_cos=None, rope_sin=None):
        """causal 与 attn_mask 解耦。

        原实现为 is_causal=(mask is None)，导致 mask=None 恒触发因果掩码，
        “双向掩码预测”分支实际上仍是单向的。
        """
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if rope_cos is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        # GQA repeat
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=(causal and attn_mask is None),
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(attn)
