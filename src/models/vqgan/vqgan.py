import torch
import torch.nn as nn
from models.vqgan.codebook import VectorQuantize

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
    def __init__(self, image_size=256, codebook_size=1024, embedding_dim=256, channels=3):
        super().__init__()
        self.encoder = Encoder(channels, embedding_dim)
        self.decoder = Decoder(embedding_dim, channels)
        self.codebook = VectorQuantize(codebook_size, embedding_dim)
    
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

    @torch.no_grad()
    def decode_code(self, indices):
        """indices: [B, H, W] 码本索引 -> 图像 [B, 3, H*16, W*16]"""
        z_q = self.codebook.codebook(indices)      # [B, H, W, D]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]
        return self.decoder(z_q)
