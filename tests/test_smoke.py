"""方案 A 冒烟测试 —— 全部在 CPU、极小配置下运行。

每个用例直接对应 SCHEME_A_BASELINE.md A.9 中的一项修复。
"""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from config import load_config
from constants import BOS_ID, CODEBOOK_OFFSET, CODEBOOK_SIZE, MASK_ID, PAD_ID, VOCAB_SIZE
from datasets.image_dataset import ImageDataset
from datasets.token_dataset import TokenDataset
from inference import generate_autoregressive, generate_tokens
from models.transformer.model import VisualTransformer
from models.vqgan.vqgan import VQGAN

TINY = dict(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16,
            vocab_size=VOCAB_SIZE, seq_len=17)


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    return VisualTransformer(TINY)


# --- A.9 #8: token 约定 --------------------------------------------------
def test_special_tokens_are_distinct():
    assert len({PAD_ID, BOS_ID, MASK_ID}) == 3, "BOS 与 MASK 不能共用同一个 id"
    assert CODEBOOK_OFFSET >= 3
    assert VOCAB_SIZE == CODEBOOK_OFFSET + CODEBOOK_SIZE


# --- A.9 #1/#2: 导入缺失 -------------------------------------------------
def test_transformer_forward_shape(model):
    tokens = torch.randint(CODEBOOK_OFFSET, VOCAB_SIZE, (2, 17))
    out = model(tokens)
    assert out.shape == (2, 17, VOCAB_SIZE)
    assert torch.isfinite(out).all()


def test_vqgan_roundtrip():
    vq = VQGAN()
    x = torch.randn(2, 3, 64, 64)
    recon, indices, vq_loss = vq(x)
    assert recon.shape == x.shape
    assert indices.shape == (2 * 4 * 4,)
    assert torch.isfinite(vq_loss)

    _, idx2d, _ = vq.encode(x)
    assert idx2d.shape == (2, 4, 4), "encode 必须返回 [B,H,W]，否则 batch>1 时下游错位"
    assert idx2d.min() >= 0 and idx2d.max() < CODEBOOK_SIZE
    assert vq.decode_code(idx2d).shape == x.shape


# --- A.9 #6: RoPE 必须真正生效 -------------------------------------------
def test_rope_is_actually_applied(model):
    """打乱 token 顺序必须改变输出；若 RoPE 未接入，双向模式下输出只是被置换。"""
    torch.manual_seed(1)
    tokens = torch.randint(CODEBOOK_OFFSET, VOCAB_SIZE, (1, 17))
    perm = torch.randperm(17)
    with torch.no_grad():
        a = model(tokens, causal=False)[0]
        b = model(tokens[:, perm], causal=False)[0]
    # 无位置编码时 b 应等于 a 的行置换；有 RoPE 时不成立
    assert not torch.allclose(a[perm], b, atol=1e-4), "RoPE 未生效：模型对 token 顺序不敏感"


def test_rope_table_registered(model):
    assert model.rope_cos.shape == (1, 1, TINY['seq_len'], TINY['head_dim'])
    assert model.rope_sin.shape == model.rope_cos.shape


# --- A.9 #7: causal 与 attn_mask 解耦 ------------------------------------
def test_causal_and_bidirectional_differ(model):
    torch.manual_seed(2)
    tokens = torch.randint(CODEBOOK_OFFSET, VOCAB_SIZE, (1, 17))
    with torch.no_grad():
        causal = model(tokens, causal=True)
        bidir = model(tokens, causal=False)
    assert not torch.allclose(causal, bidir, atol=1e-5), "掩码分支并非真正双向"


def test_bidirectional_sees_future(model):
    """双向模式下，改动最后一个 token 必须影响第 0 个位置的输出。"""
    torch.manual_seed(3)
    a = torch.randint(CODEBOOK_OFFSET, VOCAB_SIZE, (1, 17))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 7 - CODEBOOK_OFFSET) % CODEBOOK_SIZE + CODEBOOK_OFFSET
    with torch.no_grad():
        assert not torch.allclose(model(a, causal=False)[:, 0], model(b, causal=False)[:, 0], atol=1e-5)
        # 因果模式下则必须不受影响
        assert torch.allclose(model(a, causal=True)[:, 0], model(b, causal=True)[:, 0], atol=1e-5)


# --- 两个训练分支都要能反传 ----------------------------------------------
@pytest.mark.parametrize('mask_ratio', [0.0, 1.0])
def test_backward_both_branches(mask_ratio):
    torch.manual_seed(4)
    m = VisualTransformer(TINY)
    tokens = torch.randint(CODEBOOK_OFFSET, VOCAB_SIZE, (2, 17))
    loss = m.compute_loss(tokens, mask_ratio=mask_ratio)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name} 无梯度"
        assert torch.isfinite(p.grad).all(), f"{name} 梯度含 NaN/Inf"


def test_empty_mask_does_not_nan():
    """mask_prob=0 时掩码分支会选中 0 个位置，原实现会产生 NaN。"""
    torch.manual_seed(5)
    m = VisualTransformer(TINY)
    tokens = torch.randint(CODEBOOK_OFFSET, VOCAB_SIZE, (2, 17))
    loss = m.compute_loss(tokens, mask_ratio=1.0, mask_prob=0.0)
    assert torch.isfinite(loss)


# --- A.9 #4: 推理 off-by-one --------------------------------------------
def test_generate_token_count(model):
    toks = generate_tokens(model, n_tokens=16, top_k=8)
    assert toks.shape == (1, 16), "自回归采样必须产出恰好 n_tokens 个图像 token"
    assert toks.min() >= CODEBOOK_OFFSET, "不应采样出 PAD/BOS/MASK"


def test_generate_and_decode(model):
    vq = VQGAN()
    img = generate_autoregressive(model, vq, n_tokens=16, top_k=8)
    assert img.shape == (3, 64, 64)
    assert 0.0 <= img.min() and img.max() <= 1.0


# --- 数据集 --------------------------------------------------------------
def test_token_dataset_pad_and_truncate(tmp_path):
    np.save(tmp_path / 'short.npy', np.array([BOS_ID, 5, 6], dtype=np.int32))
    np.save(tmp_path / 'long.npy', np.arange(50, dtype=np.int32))
    ds = TokenDataset(tmp_path, seq_len=17, use_cache=False)
    assert len(ds) == 2
    for item in ds:
        assert item.shape == (17,)
        assert item.dtype == torch.int64
    short = ds[[Path(f).stem for f in ds.files].index('short')]
    assert short[0].item() == BOS_ID
    assert short[-1].item() == PAD_ID


def test_token_dataset_train_val_disjoint(tmp_path):
    for i in range(40):
        np.save(tmp_path / f's{i:03d}.npy', np.full(17, BOS_ID, dtype=np.int32))
    tr = TokenDataset(tmp_path, 17, split='train', val_size=4, use_cache=False)
    va = TokenDataset(tmp_path, 17, split='val', val_size=4, use_cache=False)
    assert len(va) == 4 and len(tr) == 36
    assert set(tr.files).isdisjoint(va.files), "train/val 必须无重叠"


def test_image_dataset_skips_corrupt(tmp_path):
    from PIL import Image
    Image.new('RGB', (80, 80)).save(tmp_path / 'good.png')
    (tmp_path / 'broken.png').write_bytes(b'not an image')
    ds = ImageDataset(tmp_path, image_size=64, val_size=0, use_cache=False)
    assert len(ds) == 2
    for i in range(len(ds)):
        assert ds[i].shape == (3, 64, 64), "损坏图片不应让训练中断"


# --- 配置系统 ----------------------------------------------------------
def test_config_override_and_typing(tmp_path):
    cfg_file = tmp_path / 'c.yaml'
    # YAML 里写 3e-4（无小数点）在 YAML 1.1 下会被解析成字符串，必须被纠正
    cfg_file.write_text("train:\n  lr: 3e-4\n  amp: true\ndata:\n  limit: 5\n  name: e2e\n")
    cfg = load_config(cfg_file, [])
    assert cfg.train.lr == 3e-4 and isinstance(cfg.train.lr, float)
    assert cfg.data.name == 'e2e', "普通字符串不能被误转成数字"

    cfg = load_config(cfg_file, ['train.lr=1e-5', 'train.amp=false', 'data.limit=null'])
    assert cfg.train.lr == 1e-5 and isinstance(cfg.train.lr, float)
    assert cfg.train.amp is False
    assert cfg.data.limit is None


def test_config_rejects_unknown_key(tmp_path):
    cfg_file = tmp_path / 'c.yaml'
    cfg_file.write_text("train:\n  lr: 1.0\n")
    with pytest.raises(SystemExit):
        load_config(cfg_file, ['train.lrr=2.0'])  # 拼错键名必须报错而非静默忽略


# --- 初始化 ------------------------------------------------------------
def test_init_std_controls_initial_loss():
    """init_std=0.02 时初始 CE 应接近 ln(vocab)；None 复现原实现的爆炸行为。"""
    torch.manual_seed(0)
    good = VisualTransformer({**TINY, 'init_std': 0.02})
    torch.manual_seed(0)
    orig = VisualTransformer({**TINY, 'init_std': None})
    tokens = torch.randint(CODEBOOK_OFFSET, VOCAB_SIZE, (2, 17))
    with torch.no_grad():
        ce_good = good._causal_loss(tokens).item()
        ce_orig = orig._causal_loss(tokens).item()
    uniform = math.log(VOCAB_SIZE)
    assert ce_good < 2 * uniform, f"init_std=0.02 初始 CE 异常: {ce_good}"
    assert ce_orig > 4 * ce_good, f"init_std=None 应复现原实现的高初始 CE，实测 {ce_orig}"
