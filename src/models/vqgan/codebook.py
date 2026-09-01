"""原始量化器（方案 A 基线，实验 E0）。

保留是为了复现基线与做对照，新实验请用 quantize.VectorQuantizer。
已知缺陷见 OPTIMIZATION_PLAN.md 1.1：
`uniform(-1/1024, 1/1024)` 在 256 维下初始范数仅约 0.009，
而编码器输出范数约 13.85，961/1024 个码字从初始化起就永远赢不了最近邻竞争。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantize(nn.Module):
    def __init__(self, num_codewords=1024, embedding_dim=256, commitment_cost=0.25):
        super().__init__()
        self.num_codewords = num_codewords
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(num_codewords, embedding_dim)
        self.codebook.weight.data.uniform_(-1 / num_codewords, 1 / num_codewords)

    # ---- 与 quantize.VectorQuantizer 对齐的接口 ----------------------------
    def codebook_weight(self):
        return self.codebook.weight

    @torch.no_grad()
    def decode_indices(self, idx):
        """idx: [B,H,W] -> [B, embedding_dim, H, W]"""
        return self.codebook(idx).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def stats(self):
        return float('nan'), 0

    def forward(self, z_e):
        # z_e: [B, D, H, W]
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        distances = torch.cdist(z_e_flat, self.codebook.weight)
        indices = torch.argmin(distances, dim=1)
        z_q_flat = self.codebook(indices)
        z_q = z_q_flat.reshape(z_e.shape[0], z_e.shape[2], z_e.shape[3], z_e.shape[1])
        z_q = z_q.permute(0, 3, 1, 2)

        commitment_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z_e)
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        z_q = z_e + (z_q - z_e).detach()
        return z_q, indices, commitment_loss + codebook_loss
