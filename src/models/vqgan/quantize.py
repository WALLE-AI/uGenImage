"""向量量化器。

对照 OPTIMIZATION_PLAN.md 第 1.1 节的诊断：原实现
`codebook.weight.uniform_(-1/1024, 1/1024)` 使 256 维码字的初始范数约 0.009，
而编码器输出范数约 13.85。最近邻竞争中这些码字永远赢不了，也就永远拿不到梯度，
961/1024 个码字被冻结在初始化值上 —— 这是机械性的死锁，不是训练不足。

本模块用四项改动打破死锁，每项都可单独开关，便于做 E1/E2/E3 对照：

  code_dim     低维投影（256 -> 8），低维空间里码字更容易覆盖数据分布
  l2_norm      码字与输入都做 L2 归一化，距离退化为余弦相似度，范数不再影响竞争
  ema          码本用 EMA 更新而非梯度，天然稳定
  revive       长期未命中的码字用当前 batch 的编码器输出重置，杜绝永久死亡
"""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _all_reduce(t):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


class VectorQuantizer(nn.Module):
    def __init__(self, num_codewords=16384, input_dim=256, code_dim=8,
                 commitment_cost=0.25, l2_norm=True, ema=True, decay=0.99,
                 eps=1e-5, revive=True, revive_after=100, init_std=1.0):
        """revive_after: 连续多少步未被命中就复活该码字。

        用显式的"连续未命中步数"而不是 EMA cluster_size 阈值 —— 后者的合理阈值
        依赖 batch 内 token 数与码本大小之比，很容易设成"每步复活全部未命中码字"。
        """
        super().__init__()
        self.num_codewords = num_codewords
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.l2_norm = l2_norm
        self.use_ema = ema
        self.decay = decay
        self.eps = eps
        self.revive = revive
        self.revive_after = revive_after

        # 低维投影。code_dim == input_dim 时退化为恒等，便于复现原行为
        if code_dim != input_dim:
            self.proj_in = nn.Conv2d(input_dim, code_dim, 1, bias=False)
            self.proj_out = nn.Conv2d(code_dim, input_dim, 1, bias=False)
        else:
            self.proj_in = nn.Identity()
            self.proj_out = nn.Identity()

        # 初始化尺度对齐单位方差，而不是原来的 1/K（在 256 维下小到无法参与竞争）
        emb = torch.randn(num_codewords, code_dim) * init_std
        if l2_norm:
            emb = F.normalize(emb, dim=1)
        if ema:
            self.register_buffer('embedding', emb)
            self.register_buffer('cluster_size', torch.ones(num_codewords))
            self.register_buffer('embed_avg', emb.clone())
        else:
            self.embedding = nn.Parameter(emb)

        self.register_buffer('unused_steps', torch.zeros(num_codewords, dtype=torch.long))
        self.register_buffer('n_revived', torch.zeros((), dtype=torch.long))

    # ---- 供外部（评估脚本 / 解码）使用 -------------------------------------
    def codebook_weight(self):
        return self.embedding

    def _normed_codebook(self):
        return F.normalize(self.embedding, dim=1) if self.l2_norm else self.embedding

    @torch.no_grad()
    def decode_indices(self, idx):
        """idx: [B,H,W] -> [B, input_dim, H, W]"""
        z = self._normed_codebook()[idx]              # [B,H,W,code_dim]
        z = z.permute(0, 3, 1, 2).contiguous()
        return self.proj_out(z)

    # ---- 前向 --------------------------------------------------------------
    def _lookup(self, z_flat):
        cb = self._normed_codebook()
        if self.l2_norm:
            # 两边都已归一化，余弦相似度最大 == 欧氏距离最小
            return (z_flat @ cb.t()).argmax(dim=1)
        d = (z_flat.pow(2).sum(1, keepdim=True)
             - 2 * z_flat @ cb.t()
             + cb.pow(2).sum(1))
        return d.argmin(dim=1)

    @torch.no_grad()
    def _ema_update(self, z_flat, idx):
        onehot = F.one_hot(idx, self.num_codewords).type(z_flat.dtype)
        cluster = onehot.sum(0)
        embed_sum = onehot.t() @ z_flat
        _all_reduce(cluster)
        _all_reduce(embed_sum)

        self.cluster_size.mul_(self.decay).add_(cluster, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
        n = self.cluster_size.sum()
        cluster_norm = (self.cluster_size + self.eps) / (n + self.num_codewords * self.eps) * n
        embed = self.embed_avg / cluster_norm.unsqueeze(1)
        self.embedding.copy_(F.normalize(embed, dim=1) if self.l2_norm else embed)

        # 连续未命中步数：命中清零，未命中 +1
        hit = cluster > 0
        self.unused_steps[hit] = 0
        self.unused_steps[~hit] += 1

        if self.revive:
            dead = self.unused_steps >= self.revive_after
            k = int(dead.sum().item())
            if k > 0:
                # 用当前 batch 的真实编码器输出重置死码，保证它立刻能参与竞争。
                # 这是"治疗"：L2 归一化只能预防范数悬殊，救不活方向已经偏离数据的死码。
                pick = torch.randint(0, z_flat.shape[0], (k,), device=z_flat.device)
                new = z_flat[pick]
                if self.l2_norm:
                    new = F.normalize(new, dim=1)
                self.embedding[dead] = new
                self.embed_avg[dead] = new
                self.cluster_size[dead] = 1.0
                self.unused_steps[dead] = 0
                self.n_revived += k

    def forward(self, z_e):
        """z_e: [B, input_dim, H, W] -> (z_q[B,input_dim,H,W], indices[B*H*W], loss)"""
        z = self.proj_in(z_e)
        B, D, H, W = z.shape
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z_perm.view(-1, D)
        if self.l2_norm:
            z_flat = F.normalize(z_flat, dim=1)

        idx = self._lookup(z_flat)
        zq_flat = self._normed_codebook()[idx]

        # 量化损失在低维空间计算
        commit = F.mse_loss(zq_flat.detach(), z_flat)
        if self.use_ema:
            loss = self.commitment_cost * commit
            if self.training:
                self._ema_update(z_flat.detach(), idx)
        else:
            loss = self.commitment_cost * commit + F.mse_loss(zq_flat, z_flat.detach())

        zq_flat = z_flat + (zq_flat - z_flat).detach()   # straight-through
        z_q = zq_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        return self.proj_out(z_q), idx, loss

    @torch.no_grad()
    def stats(self):
        """返回 (perplexity, 复活计数)。perplexity 是有效码字数的连续版本。"""
        if not self.use_ema:
            return float('nan'), int(self.n_revived.item())
        p = self.cluster_size / self.cluster_size.sum().clamp(min=1e-8)
        ppl = torch.exp(-(p * (p + 1e-10).log()).sum()).item()
        return ppl, int(self.n_revived.item())
