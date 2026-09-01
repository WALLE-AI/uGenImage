"""量化器测试 —— 逐条对应 OPTIMIZATION_PLAN.md 第 1.1 节诊断出的死锁机制。"""
import torch

from models.vqgan.codebook import VectorQuantize
from models.vqgan.quantize import VectorQuantizer
from models.vqgan.vqgan import VQGAN, build_vqgan_from_config


def _fake_encoder_output(b=4, d=32, h=8, w=8, scale=13.85):
    """模拟真实编码器输出：范数约 13.85（实测值）。"""
    z = torch.randn(b, d, h, w)
    return z / z.norm(dim=1, keepdim=True) * scale


def test_legacy_init_norm_is_far_below_encoder_output():
    """基线初始化 uniform(-1/K, 1/K) 在高维下范数极小，是死锁的起点。"""
    q = VectorQuantize(1024, 256)
    assert q.codebook_weight().norm(dim=1).mean() < 0.05, "基线码本初始范数本应极小"


def _converged_system(n_alive=32, K=256, d=32, n=2048, radius=13.4, resid=4.1):
    """模拟收敛后的真实系统（数值取自 runs/vqgan_full 实测）：

    编码器输出已经聚拢到 n_alive 个码字附近（到胜出码字距离 4.1，自身范数 13.4），
    其余 K-n_alive 个码字仍停在 uniform(-1/K,1/K) 的范数 ~0.009。
    返回 (码本, 编码器输出)。
    """
    centers = torch.randn(n_alive, d)
    centers = centers / centers.norm(dim=1, keepdim=True) * radius
    w = torch.randn(K, d) * (0.009 / d ** 0.5)
    w[:n_alive] = centers
    noise = torch.randn(n, d)
    noise = noise / noise.norm(dim=1, keepdim=True) * resid
    z = centers[torch.randint(0, n_alive, (n,))] + noise
    return w, z.view(1, n, 1, d).permute(0, 3, 1, 2).contiguous()


def test_legacy_deadlock_dead_codes_never_win():
    """复现实测到的死锁（OPTIMIZATION_PLAN 1.1）。

    真机实测：最近的"未命中"码字在 1024 个中平均排第 18 名，
    胜出比例 0.000%，距离比胜出者远 930%。
    """
    torch.manual_seed(0)
    w, z = _converged_system()
    q = VectorQuantize(256, 32)
    with torch.no_grad():
        q.codebook.weight.copy_(w)
    _, idx, _ = q(z)
    assert idx.max().item() < 32, \
        "范数 ~0.009 的死码本应完全无法胜出 —— 这正是基线坍塌的机制"


def test_l2_norm_alone_does_not_resurrect_dead_codes():
    """L2 归一化是"预防"不是"治疗"。

    它消除了范数悬殊带来的锁定，但已经死掉的码字方向是随机的，
    仍然赢不过对齐到数据的活码。救活它们要靠 revive。
    这条断言存在是为了防止把 L2 当成完整解法。
    """
    torch.manual_seed(0)
    w, z = _converged_system()
    q = VectorQuantizer(num_codewords=256, input_dim=32, code_dim=32,
                        l2_norm=True, ema=False)
    with torch.no_grad():
        q.embedding.copy_(w)
    _, idx, _ = q(z)
    assert (idx >= 32).float().mean() < 0.2, \
        "若 L2 单独就能救活死码，则本项目对 revive 的必要性判断需要重新评估"


def test_revive_guarantees_no_code_stays_dead():
    """复活机制提供的核心保证：没有任何码字可以永久不被使用。

    基线的失败正是"961 个码字从初始化起就再没被选中过"。这里断言
    在开启 revive 后，连续未命中步数被硬性限制在 revive_after 以内。
    """
    torch.manual_seed(0)
    q = VectorQuantizer(num_codewords=2048, input_dim=32, code_dim=8,
                        ema=True, revive=True, revive_after=5)
    q.train()
    for _ in range(40):
        q(_fake_encoder_output(b=2, d=32, h=4, w=4))   # 每步只有 32 个 token
        assert q.unused_steps.max().item() <= 5, \
            f"存在连续 {q.unused_steps.max().item()} 步未命中的码字，超过 revive_after"
    assert q.n_revived.item() > 0


def test_revived_codes_land_on_the_data_manifold():
    """复活是用当前 batch 的真实编码器输出重置，不是随机重初始化 ——
    这样码字落点在数据流形上，下一步就能参与竞争。"""
    torch.manual_seed(0)
    q = VectorQuantizer(num_codewords=512, input_dim=32, code_dim=8,
                        l2_norm=True, ema=True, revive=True, revive_after=1)
    q.train()
    z = _fake_encoder_output(b=4, d=32, h=4, w=4)
    q(z), q(z)                                  # 第二步触发复活
    zf = torch.nn.functional.normalize(
        q.proj_in(z).permute(0, 2, 3, 1).reshape(-1, 8), dim=1)
    cos = (q._normed_codebook() @ zf.t()).max(dim=1).values
    assert cos.min() > 0.5, \
        f"复活后仍有码字远离数据（最差余弦 {cos.min():.2f}），可能是随机重初始化"


def test_v2_uses_far_more_codewords():
    """低维 + L2 归一化后，同样的输入应激活远多于基线的码字。"""
    torch.manual_seed(0)
    q = VectorQuantizer(num_codewords=512, input_dim=32, code_dim=8, ema=False)
    z = _fake_encoder_output(b=16, d=32)
    _, idx, _ = q(z)
    assert idx.unique().numel() > 100, f"仅用到 {idx.unique().numel()} 个码字，L2 归一化未生效"


def test_l2_norm_removes_norm_advantage():
    """L2 归一化后所有码字范数相同，不存在'范数大的永远赢'。"""
    q = VectorQuantizer(num_codewords=64, input_dim=32, code_dim=8, l2_norm=True)
    norms = q._normed_codebook().norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_ema_update_changes_codebook_without_grad():
    q = VectorQuantizer(num_codewords=64, input_dim=32, code_dim=8, ema=True, revive=False)
    before = q.embedding.clone()
    q.train()
    q(_fake_encoder_output(b=8, d=32))
    assert not torch.allclose(before, q.embedding), "EMA 未更新码本"
    assert not q.embedding.requires_grad, "EMA 模式下码本不应是可训练参数"


def test_revive_triggers_after_n_unused_steps():
    """复活以"连续未命中步数"为准，不是 EMA 阈值 ——
    后者的合理值依赖 batch token 数与码本大小之比，极易设成每步全量复活。"""
    torch.manual_seed(0)
    q = VectorQuantizer(num_codewords=1024, input_dim=32, code_dim=8,
                        ema=True, revive=True, revive_after=3)
    q.train()
    for step in range(3):
        q(_fake_encoder_output(b=2, d=32, h=4, w=4))
        if step < 2:
            assert q.n_revived.item() == 0, f"第 {step+1} 步就复活了，过于激进"
    assert q.n_revived.item() > 0, "达到 revive_after 后仍未复活"


def test_no_revive_leaves_dead_codes():
    q = VectorQuantizer(num_codewords=1024, input_dim=32, code_dim=8,
                        ema=True, revive=False, revive_after=1)
    q.train()
    for _ in range(5):
        q(_fake_encoder_output(b=2, d=32, h=4, w=4))
    assert q.n_revived.item() == 0


def test_shapes_and_gradients():
    q = VectorQuantizer(num_codewords=64, input_dim=32, code_dim=8)
    z = _fake_encoder_output(b=2, d=32, h=4, w=4)
    z.requires_grad_(True)
    zq, idx, loss = q(z)
    assert zq.shape == z.shape
    assert idx.shape == (2 * 4 * 4,)
    assert torch.isfinite(loss)
    (zq.sum() + loss).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all(), "straight-through 未打通梯度"


def test_decode_indices_roundtrip():
    for kw in ({'quantizer': 'legacy'}, {'quantizer': 'v2', 'code_dim': 8}):
        m = VQGAN(64, 64, 32, **kw)
        x = torch.randn(2, 3, 64, 64)
        _, idx, _ = m.encode(x)
        assert idx.shape == (2, 4, 4)
        assert m.decode_code(idx).shape == x.shape


def test_build_from_config_matches_checkpoint():
    cfg = {'data': {'image_size': 64},
           'model': {'quantizer': 'v2', 'codebook_size': 128, 'embedding_dim': 32,
                     'code_dim': 8, 'l2_norm': True, 'ema': True, 'revive': True}}
    m = build_vqgan_from_config(cfg)
    assert m.codebook.num_codewords == 128
    assert m.codebook.code_dim == 8
    sd = m.state_dict()
    assert build_vqgan_from_config(cfg).load_state_dict(sd) is not None


def test_stats_reports_perplexity():
    q = VectorQuantizer(num_codewords=64, input_dim=32, code_dim=8, ema=True)
    q.train()
    q(_fake_encoder_output(b=8, d=32))
    ppl, revived = q.stats()
    assert 1.0 <= ppl <= 64.0
    assert isinstance(revived, int)
