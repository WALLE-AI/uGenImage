"""T2：编解码器主干（对应 OPTIMIZATION_PLAN.md 阶段 T2）。

方案 A 的主干是纯 Conv+ReLU 堆叠，缺三样 VQGAN 系列的标准配置：
残差块、归一化层、瓶颈自注意力。这里补齐：

  ResBlock ×2/级 + GroupNorm(32) + SiLU
  16×16 分辨率处 1 层自注意力（提供全局一致性）
  上采样用 nearest + Conv3x3（避免 ConvTranspose 的棋盘伪影）

保留 backbone='legacy' 以复现基线，便于做 E4 单变量对照。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(ch, groups=32):
    return nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch, eps=1e-6, affine=True)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        out_ch = out_ch or in_ch
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    """瓶颈处的单头自注意力。16×16=256 个位置，开销可忽略。"""

    def __init__(self, ch):
        super().__init__()
        self.norm = group_norm(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(1)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return x + self.proj(out.transpose(1, 2).reshape(B, C, H, W))


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode='nearest'))


class EncoderV2(nn.Module):
    def __init__(self, in_ch=3, z_ch=256, ch=128, ch_mult=(1, 1, 2, 2, 4),
                 num_res_blocks=2, attn_resolutions=(16,), image_size=256):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.levels = nn.ModuleList()
        cur, res = ch, image_size
        for i, mult in enumerate(ch_mult):
            blocks = nn.ModuleList()
            out = ch * mult
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(cur, out))
                cur = out
                if res in attn_resolutions:
                    blocks.append(AttnBlock(cur))
            last = i == len(ch_mult) - 1
            self.levels.append(nn.ModuleDict({
                'blocks': blocks,
                'down': Downsample(cur) if not last else nn.Identity(),
            }))
            if not last:
                res //= 2
        self.mid = nn.Sequential(ResBlock(cur), AttnBlock(cur), ResBlock(cur))
        self.norm_out = group_norm(cur)
        self.conv_out = nn.Conv2d(cur, z_ch, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for lvl in self.levels:
            for b in lvl['blocks']:
                h = b(h)
            h = lvl['down'](h)
        h = self.mid(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class DecoderV2(nn.Module):
    def __init__(self, out_ch=3, z_ch=256, ch=128, ch_mult=(1, 1, 2, 2, 4),
                 num_res_blocks=2, attn_resolutions=(16,), image_size=256):
        super().__init__()
        cur = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(z_ch, cur, 3, padding=1)
        self.mid = nn.Sequential(ResBlock(cur), AttnBlock(cur), ResBlock(cur))
        self.levels = nn.ModuleList()
        res = image_size // 2 ** (len(ch_mult) - 1)
        for i, mult in enumerate(reversed(ch_mult)):
            blocks = nn.ModuleList()
            out = ch * mult
            for _ in range(num_res_blocks + 1):
                blocks.append(ResBlock(cur, out))
                cur = out
                if res in attn_resolutions:
                    blocks.append(AttnBlock(cur))
            last = i == len(ch_mult) - 1
            self.levels.append(nn.ModuleDict({
                'blocks': blocks,
                'up': Upsample(cur) if not last else nn.Identity(),
            }))
            if not last:
                res *= 2
        self.norm_out = group_norm(cur)
        self.conv_out = nn.Conv2d(cur, out_ch, 3, padding=1)

    def forward(self, z):
        h = self.mid(self.conv_in(z))
        for lvl in self.levels:
            for b in lvl['blocks']:
                h = b(h)
            h = lvl['up'](h)
        return torch.tanh(self.conv_out(F.silu(self.norm_out(h))))
