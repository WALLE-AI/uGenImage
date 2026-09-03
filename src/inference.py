import torch
import torch.nn.functional as F

from constants import BOS_ID, CODEBOOK_OFFSET, CODEBOOK_SIZE, LATENT_TOKENS


@torch.no_grad()
def generate_tokens(model, n_tokens=LATENT_TOKENS, temperature=0.9, top_k=50,
                    top_p=None, batch_size=1, codebook_size=CODEBOOK_SIZE, use_cache=True):
    """从 BOS 起自回归采样 n_tokens 个图像 token，返回 [B, n_tokens]（不含 BOS）。

    use_cache=True 时用 KV-cache 增量解码：每步只前向 1 个 token，
    整体复杂度从 O(n^3)（原实现每步重算整个前缀）降到 O(n^2)。
    """
    model.eval()
    device = next(model.parameters()).device
    hi = CODEBOOK_OFFSET + codebook_size

    tokens = torch.full((batch_size, 1), BOS_ID, dtype=torch.long, device=device)
    caches = model.new_caches() if use_cache else None
    step_in = tokens

    for i in range(n_tokens):  # 原实现为 seq_len-1，导致最终只有 255 个 token
        if use_cache:
            logits = model(step_in, causal=True, caches=caches, pos=i)[:, -1, :]
        else:
            logits = model(tokens, causal=True)[:, -1, :]
        logits = logits / temperature
        # 特殊 token（PAD/BOS/MASK）与码本范围之外的 id 都不应被采样出来
        logits[:, :CODEBOOK_OFFSET] = -float('inf')
        if hi < logits.size(-1):
            logits[:, hi:] = -float('inf')

        if top_k:
            k = min(top_k, hi - CODEBOOK_OFFSET)
            thresh = torch.topk(logits, k, dim=-1)[0][..., -1, None]
            logits = logits.masked_fill(logits < thresh, -float('inf'))
        if top_p:
            srt, idx = torch.sort(logits, descending=True, dim=-1)
            cum = torch.softmax(srt, dim=-1).cumsum(dim=-1)
            drop = cum - torch.softmax(srt, dim=-1) > top_p
            srt = srt.masked_fill(drop, -float('inf'))
            logits = torch.full_like(logits, -float('inf')).scatter_(-1, idx, srt)

        next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
        tokens = torch.cat([tokens, next_token], dim=1)
        step_in = next_token

    return tokens[:, 1:]


@torch.no_grad()
def generate_autoregressive(model, vqgan, n_tokens=LATENT_TOKENS, temperature=0.9,
                            top_k=50, codebook_size=None, top_p=None, batch_size=1,
                            use_cache=True):
    """采样并解码为图像，返回 [B, 3, H, W]（B=1 时返回 [3,H,W]），值域 [0,1]。"""
    if codebook_size is None:
        codebook_size = vqgan.codebook.num_codewords
    tokens = generate_tokens(model, n_tokens, temperature, top_k, top_p,
                             batch_size, codebook_size, use_cache)
    codes = (tokens - CODEBOOK_OFFSET).clamp_(0, codebook_size - 1)
    h = w = int(round(n_tokens ** 0.5))
    assert h * w == n_tokens, f"n_tokens={n_tokens} 不是完全平方数，无法还原成方形潜码网格"
    img = vqgan.decode_code(codes.view(-1, h, w))
    img = ((img + 1) * 0.5).clamp_(0, 1).cpu()
    return img.squeeze(0) if batch_size == 1 else img
