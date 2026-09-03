import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer.norm import RMSNorm


def precompute_rope(head_dim, max_seq_len, theta=10000.0, device=None, dtype=torch.float32):
    """1D RoPE 的 cos/sin 表。

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


def precompute_rope_2d(head_dim, grid, theta=10000.0, n_prefix=1, device=None,
                       dtype=torch.float32):
    """2D RoPE：head_dim 按 h/w 两轴对半分解。

    图像 token 是 grid×grid 的二维网格，用一维序号做 RoPE 是错误的先验 ——
    (0,15) 和 (1,0) 在一维上相邻，在图像上却隔了一整行。

    n_prefix 个前缀位置（BOS / 条件 token）用零角度，即不施加旋转。
    返回 cos, sin —— [1, 1, n_prefix + grid*grid, head_dim]
    """
    assert head_dim % 4 == 0, "2D RoPE 需要 head_dim 能被 4 整除"
    quarter = head_dim // 4
    inv_freq = 1.0 / (theta ** (torch.arange(0, quarter, device=device,
                                             dtype=torch.float32) / quarter))
    pos = torch.arange(grid, device=device, dtype=torch.float32)
    f = torch.outer(pos, inv_freq)                              # [grid, head_dim/4]
    fh = f[:, None, :].expand(grid, grid, quarter).reshape(-1, quarter)
    fw = f[None, :, :].expand(grid, grid, quarter).reshape(-1, quarter)
    freqs = torch.cat([fh, fw], dim=-1)                          # [grid^2, head_dim/2]
    if n_prefix:
        freqs = torch.cat([freqs.new_zeros(n_prefix, freqs.shape[1]), freqs], dim=0)
    emb = torch.cat((freqs, freqs), dim=-1)                      # [T, head_dim]
    return emb.cos()[None, None].to(dtype), emb.sin()[None, None].to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class GroupedQueryAttention(nn.Module):
    """n_kv_heads == n_heads 时即为标准 MHA。

    seq_len 只有 257 时 KV-cache 微不足道，GQA 是纯损质量而不省资源；
    视频阶段序列变长后再把 n_kv_heads 调小即可，无需改代码。
    """

    def __init__(self, dim, n_heads=24, n_kv_heads=6, head_dim=128, qk_norm=False):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads 必须能被 n_kv_heads 整除"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
        # QK-norm：视觉 token + 低精度训练时 attention logit 爆炸是已知高发问题
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

    def forward(self, x, attn_mask=None, causal=True, rope_cos=None, rope_sin=None,
                cache=None):
        """cache: 可选的 dict(k=..., v=...)，用于增量解码；会被就地更新。

        原实现为 is_causal=(mask is None)，导致 mask=None 恒触发因果掩码，
        “双向掩码预测”分支实际上仍是单向的。
        """
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope_cos is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        if cache is not None:
            if cache.get('k') is not None:
                k = torch.cat([cache['k'], k], dim=2)
                v = torch.cat([cache['v'], v], dim=2)
            cache['k'], cache['v'] = k, v
            causal = False          # 增量解码时 query 只有当前若干位，天然只能看到前缀

        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=(causal and attn_mask is None and T > 1),
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(attn)
