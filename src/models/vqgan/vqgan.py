import torch
import torch.nn as nn
from models.vqgan.codebook import VectorQuantize
from models.vqgan.quantize import VectorQuantizer
from models.vqgan.backbone import EncoderV2, DecoderV2

class Encoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=256, ch=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, ch, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch*2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch*2, ch*2, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch*2, ch*4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch*4, ch*4, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch*4, ch*4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch*4, out_ch, 3, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.conv4(self.conv3(self.conv2(self.conv1(x))))

class Decoder(nn.Module):
    def __init__(self, in_ch=256, out_ch=3, ch=128):
        super().__init__()
        self.deconv1 = nn.Sequential(
            nn.Conv2d(in_ch, ch*4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*4, ch*4, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(ch*4, ch*2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*2, ch*2, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(ch*2, ch, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.deconv4 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.deconv4(self.deconv3(self.deconv2(self.deconv1(x))))

class VQGAN(nn.Module):
    def __init__(self, image_size=256, codebook_size=1024, embedding_dim=256, channels=3,
                 quantizer='legacy', code_dim=8, l2_norm=True, ema=True, decay=0.99,
                 revive=True, revive_after=100, commitment_cost=0.25,
                 backbone='legacy', ch=128, ch_mult=(1, 1, 2, 2, 4),
                 num_res_blocks=2, attn_resolutions=(16,)):
        """quantizer: 'legacy'(方案 A 原实现, E0) | 'v2'(低维+L2+EMA+复活)
        backbone : 'legacy'(纯 Conv+ReLU) | 'v2'(ResBlock+GroupNorm+SiLU+瓶颈注意力)"""
        super().__init__()
        if backbone == 'legacy':
            self.encoder = Encoder(channels, embedding_dim)
            self.decoder = Decoder(embedding_dim, channels)
        elif backbone == 'v2':
            kw = dict(ch=ch, ch_mult=tuple(ch_mult), num_res_blocks=num_res_blocks,
                      attn_resolutions=tuple(attn_resolutions), image_size=image_size)
            self.encoder = EncoderV2(channels, embedding_dim, **kw)
            self.decoder = DecoderV2(channels, embedding_dim, **kw)
        else:
            raise ValueError(f"未知 backbone: {backbone}")
        if quantizer == 'legacy':
            self.codebook = VectorQuantize(codebook_size, embedding_dim, commitment_cost)
        elif quantizer == 'v2':
            self.codebook = VectorQuantizer(
                num_codewords=codebook_size, input_dim=embedding_dim, code_dim=code_dim,
                commitment_cost=commitment_cost, l2_norm=l2_norm, ema=ema, decay=decay,
                revive=revive, revive_after=revive_after)
        else:
            raise ValueError(f"未知 quantizer: {quantizer}")


    def forward(self, x):
        z_e = self.encoder(x)
        z_q, indices, vq_loss = self.codebook(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, indices, vq_loss
    
    def encode(self, x):
        """返回 [B, H, W] 形状的码本索引。

        原实现直接返回展平的 [B*H*W]，只有 batch=1 时下游 squeeze/flatten 才凑巧正确。
        """
        z_e = self.encoder(x)
        _, indices, _ = self.codebook(z_e)
        B, _, H, W = z_e.shape
        return z_e, indices.view(B, H, W), None

    def get_last_layer(self):
        """解码器最后一层的权重，用于对抗损失的自适应权重计算。"""
        if hasattr(self.decoder, 'conv_out'):
            return self.decoder.conv_out.weight
        return self.decoder.deconv4[-2].weight   # legacy: ConvTranspose2d(在 Tanh 之前)

    @torch.no_grad()
    def decode_code(self, indices):
        """indices: [B, H, W] 码本索引 -> 图像 [B, 3, H*16, W*16]"""
        return self.decoder(self.codebook.decode_indices(indices))


def build_vqgan_from_config(cfg):
    """从 checkpoint 里存的完整配置重建 VQGAN，避免手工对齐超参。"""
    data = cfg.get('data', {}) if isinstance(cfg, dict) else {}
    m = cfg.get('model', {}) if isinstance(cfg, dict) else {}
    return VQGAN(
        image_size=data.get('image_size', 256),
        codebook_size=m.get('codebook_size', 1024),
        embedding_dim=m.get('embedding_dim', 256),
        quantizer=m.get('quantizer', 'legacy'),
        code_dim=m.get('code_dim', 8),
        l2_norm=m.get('l2_norm', True),
        ema=m.get('ema', True),
        decay=m.get('decay', 0.99),
        revive=m.get('revive', True),
        revive_after=m.get('revive_after', 100),
        commitment_cost=m.get('commitment_cost', 0.25),
        backbone=m.get('backbone', 'legacy'),
        ch=m.get('ch', 128),
        ch_mult=m.get('ch_mult', (1, 1, 2, 2, 4)),
        num_res_blocks=m.get('num_res_blocks', 2),
        attn_resolutions=m.get('attn_resolutions', (16,)),
    )
