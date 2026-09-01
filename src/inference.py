import torch
import torch.nn.functional as F

from constants import BOS_ID, CODEBOOK_OFFSET, CODEBOOK_SIZE, LATENT_TOKENS


@torch.no_grad()
def generate_tokens(model, n_tokens=LATENT_TOKENS, temperature=0.9, top_k=50,
                    batch_size=1, codebook_size=CODEBOOK_SIZE):
    """从 BOS 起自回归采样 n_tokens 个图像 token，返回 [B, n_tokens]（不含 BOS）。

    注意：本实现没有 KV-cache，每步都重算整个前缀，复杂度 O(n^3)。
    """
    model.eval()
    device = next(model.parameters()).device
    tokens = torch.full((batch_size, 1), BOS_ID, dtype=torch.long, device=device)
    hi = CODEBOOK_OFFSET + codebook_size

    for _ in range(n_tokens):  # 原实现为 seq_len-1，导致最终只有 255 个 token
        logits = model(tokens, causal=True)[:, -1, :] / temperature
        # 特殊 token（PAD/BOS/MASK）与码本范围之外的 id 都不应被采样出来
        logits[:, :CODEBOOK_OFFSET] = -float('inf')
        if hi < logits.size(-1):
            logits[:, hi:] = -float('inf')
        k = min(top_k, hi - CODEBOOK_OFFSET)
        thresh = torch.topk(logits, k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(logits < thresh, -float('inf'))
        next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens[:, 1:]


@torch.no_grad()
def generate_autoregressive(model, vqgan, n_tokens=LATENT_TOKENS, temperature=0.9,
                            top_k=50, codebook_size=None):
    """采样并解码为图像，返回 [3, H, W]，值域 [0, 1]。"""
    if codebook_size is None:
        codebook_size = vqgan.codebook.num_codewords
    tokens = generate_tokens(model, n_tokens, temperature, top_k,
                             batch_size=1, codebook_size=codebook_size)
    codes = (tokens - CODEBOOK_OFFSET).clamp_(0, codebook_size - 1)
    h = w = int(round(n_tokens ** 0.5))
    assert h * w == n_tokens, f"n_tokens={n_tokens} 不是完全平方数，无法还原成方形潜码网格"
    img = vqgan.decode_code(codes.view(1, h, w))
    return ((img + 1) * 0.5).clamp_(0, 1).squeeze(0).cpu()
