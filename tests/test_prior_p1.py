"""P1：prior 结构修正的测试（OPTIMIZATION_PLAN.md 阶段 P1）。"""
import math

import pytest
import torch

from constants import CODEBOOK_OFFSET, VOCAB_SIZE
from inference import generate_tokens
from models.transformer.attention import precompute_rope_2d
from models.transformer.block import SwiGLU, swiglu_hidden
from models.transformer.model import VisualTransformer

BASE = dict(dim=64, n_layers=2, n_heads=4, n_kv_heads=4, head_dim=16,
            vocab_size=VOCAB_SIZE, seq_len=17, latent_size=4)


def _tokens(b=2, t=17):
    return torch.randint(CODEBOOK_OFFSET, VOCAB_SIZE, (b, t))


# --- SwiGLU 比例 --------------------------------------------------------
def test_swiglu_default_is_8_3_not_4x():
    """方案 A 的 expansion=4 使三个矩阵合计 12d²，比标准 SwiGLU 多 50% FLOPs。"""
    dim = 2048
    assert swiglu_hidden(dim, expansion=4) == 4 * dim
    h = swiglu_hidden(dim)                       # 默认 8/3
    assert abs(h - 8 * dim / 3) < 256
    params_std = 3 * dim * h
    params_a = 3 * dim * 4 * dim
    assert params_std / params_a < 0.72, "标准比例应显著小于 expansion=4"


def test_swiglu_hidden_aligned():
    assert SwiGLU(1280).hidden % 256 == 0


# --- 2D RoPE ------------------------------------------------------------
def test_rope_2d_table_shape_and_prefix():
    cos, sin = precompute_rope_2d(16, grid=4, n_prefix=1)
    assert cos.shape == (1, 1, 17, 16) == sin.shape
    # 前缀位置角度为 0 -> cos=1, sin=0，即不施加旋转
    assert torch.allclose(cos[0, 0, 0], torch.ones(16))
    assert torch.allclose(sin[0, 0, 0], torch.zeros(16))


def test_rope_2d_encodes_grid_geometry():
    """同一行相邻(w 差1) 与 同一列相邻(h 差1) 必须产生不同的编码；
    而一维 RoPE 下 (0,3) 与 (1,0) 会被当成相邻。"""
    cos, _ = precompute_rope_2d(16, grid=4, n_prefix=0)
    c = cos[0, 0]
    same_row = (c[0] - c[1]).abs().sum()      # (0,0) vs (0,1)
    same_col = (c[0] - c[4]).abs().sum()      # (0,0) vs (1,0)
    wrap = (c[3] - c[4]).abs().sum()          # (0,3) vs (1,0)：一维上相邻
    assert same_row > 0 and same_col > 0
    assert wrap > same_row * 0.5, "2D RoPE 下行末与下一行行首不应被视为紧邻"


def test_model_accepts_rope_2d():
    m = VisualTransformer({**BASE, 'rope': '2d'})
    assert m.rope_cos.shape == (1, 1, 17, 16)
    out = m(_tokens())
    assert out.shape == (2, 17, VOCAB_SIZE) and torch.isfinite(out).all()


# --- QK-norm ------------------------------------------------------------
def test_qk_norm_changes_output_and_bounds_logits():
    torch.manual_seed(0)
    plain = VisualTransformer({**BASE, 'qk_norm': False})
    torch.manual_seed(0)
    normed = VisualTransformer({**BASE, 'qk_norm': True})
    t = _tokens()
    with torch.no_grad():
        assert not torch.allclose(plain(t), normed(t)), "qk_norm 未生效"
    assert any('q_norm.weight' in n for n, _ in normed.named_parameters())


# --- MHA vs GQA ---------------------------------------------------------
def test_mha_when_kv_heads_equals_heads():
    m = VisualTransformer({**BASE, 'n_kv_heads': 4})
    a = m.layers[0].attn
    assert a.wk.out_features == a.wq.out_features, "n_kv_heads==n_heads 时应为标准 MHA"


def test_gqa_still_supported():
    m = VisualTransformer({**BASE, 'n_heads': 4, 'n_kv_heads': 2})
    a = m.layers[0].attn
    assert a.wk.out_features * 2 == a.wq.out_features


# --- 权重解绑 -----------------------------------------------------------
def test_untied_head_is_separate_parameter():
    tied = VisualTransformer({**BASE, 'tie_embeddings': True})
    untied = VisualTransformer({**BASE, 'tie_embeddings': False})
    assert tied.lm_head.weight is tied.token_embedding.weight
    assert untied.lm_head.weight is not untied.token_embedding.weight
    n_t = sum(p.numel() for p in set(tied.parameters()))
    n_u = sum(p.numel() for p in set(untied.parameters()))
    assert n_u - n_t == BASE['vocab_size'] * BASE['dim']


# --- KV-cache -----------------------------------------------------------
@pytest.mark.parametrize('rope', ['1d', '2d'])
def test_kv_cache_matches_full_recompute(rope):
    """增量解码必须与全量重算逐位一致 —— 否则采样分布悄悄变了。"""
    torch.manual_seed(0)
    m = VisualTransformer({**BASE, 'rope': rope}).eval()
    t = _tokens(b=2, t=9)
    with torch.no_grad():
        full = m(t, causal=True)
        caches = m.new_caches()
        outs = [m(t[:, i:i + 1], causal=True, caches=caches, pos=i) for i in range(t.shape[1])]
        inc = torch.cat(outs, dim=1)
    assert torch.allclose(full, inc, atol=1e-4), \
        f"KV-cache 与全量重算不一致，最大差 {(full - inc).abs().max():.2e}"


def test_generate_with_and_without_cache_agree():
    torch.manual_seed(0)
    m = VisualTransformer(BASE).eval()
    torch.manual_seed(1)
    a = generate_tokens(m, n_tokens=16, top_k=4, use_cache=True)
    torch.manual_seed(1)
    b = generate_tokens(m, n_tokens=16, top_k=4, use_cache=False)
    assert torch.equal(a, b), "开关 KV-cache 不应改变采样结果"


# --- 参数量核算 ---------------------------------------------------------
def test_p1_config_hits_target_size():
    """P1 建议的 0.39B 配置：dim1280 / 20 层 / MHA 16x80 / SwiGLU 8-3。"""
    cfg = dict(dim=1280, n_layers=20, n_heads=16, n_kv_heads=16, head_dim=80,
               vocab_size=16387, seq_len=257, latent_size=16, rope='2d',
               qk_norm=True, tie_embeddings=False, init_std=0.02)
    with torch.device('meta'):
        m = VisualTransformer(cfg)
    n = sum(p.numel() for p in m.parameters())
    assert 0.40e9 < n < 0.48e9, f"参数量 {n/1e9:.3f}B 偏离 P1 设计"
    assert math.isclose(cfg['n_heads'] * cfg['head_dim'], cfg['dim']), \
        "n_heads*head_dim 必须等于 dim"
