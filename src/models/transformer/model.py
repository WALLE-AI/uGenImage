import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import LATENT_SIZE, MASK_ID, PAD_ID
from models.transformer.attention import precompute_rope, precompute_rope_2d
from models.transformer.block import RMSNorm, TransformerBlock  # 原实现漏掉 RMSNorm


class VisualTransformer(nn.Module):
    """图像 token 的自回归 / 掩码先验。

    P1 阶段（OPTIMIZATION_PLAN.md）相对方案 A 的结构修正，全部可由 config 开关：
      rope='2d'         图像是二维网格，一维序号是错误先验
      qk_norm=True      视觉 token + 低精度训练下 attention logit 易爆炸
      swiglu_expansion  置 null 时用标准 8/3 比例，而非方案 A 的 4
      n_kv_heads=n_heads 即 MHA；seq 257 下 GQA 纯损质量
      tie_embeddings    码本扩到 16384 后解绑代价上升，可配置
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config['dim']
        self.token_embedding = nn.Embedding(config['vocab_size'], dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, config['n_heads'], config['n_kv_heads'],
                             config['head_dim'],
                             swiglu_expansion=config.get('swiglu_expansion'),
                             qk_norm=config.get('qk_norm', False))
            for _ in range(config['n_layers'])
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, config['vocab_size'], bias=False)
        if config.get('tie_embeddings', True):
            self.lm_head.weight = self.token_embedding.weight

        # RoPE cos/sin 表（原实现只有 apply 函数，从未生成过表，位置编码实际未生效）
        if config.get('rope', '1d') == '2d':
            grid = config.get('latent_size', LATENT_SIZE)
            n_prefix = config['seq_len'] - grid * grid
            assert n_prefix >= 0, f"seq_len={config['seq_len']} 小于 {grid}x{grid}"
            cos, sin = precompute_rope_2d(config['head_dim'], grid, n_prefix=n_prefix)
        else:
            cos, sin = precompute_rope(config['head_dim'], config['seq_len'])
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

        # 原实现没有任何显式初始化，nn.Embedding 默认 N(0,1)；因为 lm_head 与之绑定，
        # 会把 logits 的 std 推到 71，初始交叉熵约 1742（理论值 ln(vocab)=6.93），
        # 且 AMP fp16 下前若干步梯度直接溢出。init_std=None 可复现原行为。
        init_std = config.get('init_std', 0.02)
        if init_std is not None:
            self.init_weights(float(init_std))

    def init_weights(self, std=0.02):
        def _init(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)
        # 残差分支的输出投影按深度缩放，避免多层累积后残差流方差爆炸
        scale = std / math.sqrt(2 * len(self.layers))
        for blk in self.layers:
            nn.init.normal_(blk.attn.wo.weight, mean=0.0, std=scale)
            nn.init.normal_(blk.ffn.w3.weight, mean=0.0, std=scale)

    # ---- 前向 --------------------------------------------------------------
    def _rope(self, start, length, dtype):
        cos = self.rope_cos[:, :, start:start + length].to(dtype=dtype)
        sin = self.rope_sin[:, :, start:start + length].to(dtype=dtype)
        return cos, sin

    def forward(self, tokens, attn_mask=None, causal=True, caches=None, pos=0):
        """caches: 每层一个 dict，用于增量解码；pos 是当前片段在序列中的起始位置。"""
        T = tokens.shape[1]
        assert pos + T <= self.rope_cos.shape[2], \
            f"位置 {pos + T} 超过 RoPE 预计算长度 {self.rope_cos.shape[2]}"
        cos, sin = self._rope(pos, T, self.token_embedding.weight.dtype)

        x = self.token_embedding(tokens)
        for i, layer in enumerate(self.layers):
            x = layer(x, attn_mask, causal, cos, sin,
                      caches[i] if caches is not None else None)
        x = self.norm(x)
        return self.lm_head(x)

    def new_caches(self):
        return [{'k': None, 'v': None} for _ in self.layers]

    # ---- 损失 --------------------------------------------------------------
    def compute_loss(self, tokens, mask_ratio=0.2, mask_prob=0.15, use_masked=None):
        """方案 A 的混合目标：以 mask_ratio 的概率走双向掩码预测，否则走因果预测。

        mask_ratio 控制的是“哪个目标生效”，不是被掩 token 的比例（后者是 mask_prob）。

        use_masked 显式指定走哪个分支。DDP 下必须由调用方统一决定，
        否则各 rank 可能走不同分支，梯度语义不一致。
        """
        if use_masked is None:
            use_masked = torch.rand(1).item() < mask_ratio

        if use_masked:
            masked_tokens = tokens.clone()
            sel = (torch.rand_like(tokens, dtype=torch.float) < mask_prob) & (tokens != PAD_ID)
            if not sel.any():
                # 空掩码会让 cross_entropy 作用于 0 个元素并产生 NaN，回退到因果分支
                return self._causal_loss(tokens)
            masked_tokens[sel] = MASK_ID
            logits = self.forward(masked_tokens, causal=False)  # 真·双向
            return F.cross_entropy(logits[sel], tokens[sel])

        return self._causal_loss(tokens)

    def _causal_loss(self, tokens):
        logits = self.forward(tokens, causal=True)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tokens[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=PAD_ID,
        )
