# ============================================================
# ⚠️  已过时：本文件仅作历史留档，请勿运行。
#
# 它生成的是 **修复前** 的代码，与当前的 src/ 已经不同步。
# 运行它会在 ./visual_transformer_0.39B/ 下产生一份陈旧副本
# （并对该目录执行 shutil.rmtree），不会影响 src/，但会造成混淆。
#
# 唯一真相是 src/。见 PLAN.md P0-1。
# ============================================================
import os
import shutil
from pathlib import Path

# ============================================================
# 项目根目录名称
# ============================================================
PROJECT_NAME = "visual_transformer_0.39B"

# ============================================================
# 所有需要生成的文件及内容 (字典: 文件路径 -> 文件内容)
# ============================================================
FILES = {}

# ---------- 1. 配置文件 ----------
FILES["configs/vqgan_config.yaml"] = """
# VQGAN 训练配置
image_size: 256
codebook_size: 1024
embedding_dim: 256
channels: 3
lr: 4.5e-6
weight_decay: 0.01
epochs: 100
batch_size: 32
"""

FILES["configs/transformer_config.yaml"] = """
# Transformer 训练配置 (0.39B)
dim: 2048
n_layers: 20
n_heads: 24
n_kv_heads: 6  # GQA 4:1 比例
head_dim: 128
vocab_size: 1028  # 1024 codebook + 4 special tokens
seq_len: 256
lr: 3e-4
weight_decay: 0.1
epochs: 200
mask_ratio: 0.2  # 20% 掩码预测
"""

# ---------- 2. 数据集模块 ----------
FILES["datasets/__init__.py"] = ""
FILES["datasets/image_dataset.py"] = """
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from pathlib import Path

class ImageDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.root_dir = Path(root_dir)
        self.image_paths = list(self.root_dir.glob('*.jpg')) + list(self.root_dir.glob('*.png'))
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
    
    def __len__(self): return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)
"""

FILES["datasets/token_dataset.py"] = """
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class TokenDataset(Dataset):
    def __init__(self, data_dir, seq_len=256):
        self.data_dir = Path(data_dir)
        self.files = list(self.data_dir.glob('*.npy'))
        self.seq_len = seq_len
    
    def __len__(self): return len(self.files)
    
    def __getitem__(self, idx):
        seq = np.load(self.files[idx])
        if len(seq) > self.seq_len:
            seq = seq[:self.seq_len]
        else:
            pad_len = self.seq_len - len(seq)
            seq = np.pad(seq, (0, pad_len), constant_values=0)  # PAD_ID = 0
        return torch.LongTensor(seq)
"""

# ---------- 3. VQGAN 模型 ----------
FILES["models/__init__.py"] = ""
FILES["models/vqgan/__init__.py"] = ""
FILES["models/vqgan/codebook.py"] = """
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
        self.codebook.weight.data.uniform_(-1/num_codewords, 1/num_codewords)
    
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
"""

FILES["models/vqgan/vqgan.py"] = """
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
        z_e = self.encoder(x)
        _, indices, _ = self.codebook(z_e)
        return z_e, indices, None
"""

# ---------- 4. Transformer 核心 (RoPE + GQA + 20层) ----------
FILES["models/transformer/__init__.py"] = ""
FILES["models/transformer/attention.py"] = """
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, n_heads=24, n_kv_heads=6, head_dim=128):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
    
    def forward(self, x, mask=None, rope_cos=None, rope_sin=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        if rope_cos is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
        
        # GQA repeat
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None))
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(attn)
"""

FILES["models/transformer/block.py"] = """
import torch
import torch.nn as nn
from models.transformer.attention import GroupedQueryAttention

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return self.weight * (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

class SwiGLU(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * expansion, bias=False)
        self.w2 = nn.Linear(dim, dim * expansion, bias=False)
        self.w3 = nn.Linear(dim * expansion, dim, bias=False)
    
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, head_dim, swiglu_expansion=4):
        super().__init__()
        self.attn = GroupedQueryAttention(dim, n_heads, n_kv_heads, head_dim)
        self.ffn = SwiGLU(dim, swiglu_expansion)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
    
    def forward(self, x, mask=None, rope_cos=None, rope_sin=None):
        x = x + self.attn(self.norm1(x), mask, rope_cos, rope_sin)
        x = x + self.ffn(self.norm2(x))
        return x
"""

FILES["models/transformer/model.py"] = """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.block import TransformerBlock

class VisualTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config['vocab_size'], config['dim'])
        self.layers = nn.ModuleList([
            TransformerBlock(config['dim'], config['n_heads'], config['n_kv_heads'], config['head_dim'])
            for _ in range(config['n_layers'])
        ])
        self.norm = RMSNorm(config['dim'])
        self.lm_head = nn.Linear(config['dim'], config['vocab_size'], bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying
    
    def forward(self, tokens, mask=None):
        x = self.token_embedding(tokens)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.lm_head(x)
    
    def compute_loss(self, tokens, mask_ratio=0.2):
        B, T = tokens.shape
        use_masked = torch.rand(1).item() < mask_ratio
        
        if use_masked:
            # 20% 掩码预测
            masked_tokens = tokens.clone()
            mask_prob = torch.rand_like(tokens.float())
            mask = (mask_prob < 0.15) & (tokens != 0)
            masked_tokens[mask] = 1  # MASK_ID
            logits = self.forward(masked_tokens, mask=None)  # 双向
            loss = F.cross_entropy(logits[mask].view(-1, logits.size(-1)), tokens[mask].view(-1), ignore_index=0)
        else:
            # 80% 因果预测
            logits = self.forward(tokens, mask=None)  # Flash attention uses is_causal
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tokens[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=0)
        return loss
"""

# ---------- 5. 训练器 ----------
FILES["trainers/__init__.py"] = ""
FILES["trainers/transformer_trainer.py"] = """
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

class TransformerTrainer:
    def __init__(self, model, config):
        self.model = model.cuda()
        self.config = config
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        self.scaler = GradScaler()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['epochs'])
    
    def train_step(self, batch):
        tokens = batch.cuda()
        self.optimizer.zero_grad()
        with autocast():
            loss = self.model.compute_loss(tokens, mask_ratio=0.2)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
"""

# ---------- 6. 推理生成代码 ----------
FILES["inference.py"] = """
import torch
import torch.nn.functional as F
from models.transformer.model import VisualTransformer

@torch.no_grad()
def generate_autoregressive(model, vqgan_decoder, seq_len=256, temperature=0.9, top_k=50):
    model.eval()
    device = next(model.parameters()).device
    tokens = torch.full((1, 1), 1, dtype=torch.long, device=device)  # BOS
    
    for _ in range(seq_len - 1):
        logits = model(tokens)  # causal
        logits = logits[:, -1, :] / temperature
        # top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_token], dim=1)
    
    # Decode to image
    raw_indices = tokens[:, 1:] - 2
    raw_indices = raw_indices.clamp(min=0, max=1023)
    h = w = int(seq_len ** 0.5)
    indices_2d = raw_indices.view(1, h, w)
    
    # 假设 vqgan_decoder 有 decode_code 方法
    z_q = vqgan_decoder.codebook.codebook(indices_2d) 
    z_q = z_q.permute(0, 3, 1, 2)
    img = vqgan_decoder.decoder(z_q)
    img = (img + 1) * 0.5
    return img.squeeze(0).cpu()
"""

# ---------- 7. 训练脚本 (Shell) ----------
FILES["scripts/train_transformer.sh"] = """#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4
DIM=2048
N_LAYERS=20
N_HEADS=24
N_KV_HEADS=6
SEQ_LEN=256
BATCH_SIZE=16
EPOCHS=200
LR=3e-4

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29501 \\
    train_transformer.py \\
    --token_dir data/tokens_train \\
    --dim $DIM --n_layers $N_LAYERS --n_heads $N_HEADS --n_kv_heads $N_KV_HEADS \\
    --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR
"""

FILES["scripts/preencode_dataset.sh"] = """#!/bin/bash
# 使用 VQGAN 预编码图片为 Token
python preencode.py \\
    --vqgan_ckpt checkpoints/vqgan_final.pth \\
    --image_dir data/raw/train \\
    --output_dir data/tokens_train
"""

# ---------- 8. 入口脚本 ----------
FILES["train_transformer.py"] = """
import argparse
import torch
from torch.utils.data import DataLoader
from datasets.token_dataset import TokenDataset
from models.transformer.model import VisualTransformer
from trainers.transformer_trainer import TransformerTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token_dir', type=str, required=True)
    parser.add_argument('--dim', type=int, default=2048)
    parser.add_argument('--n_layers', type=int, default=20)
    parser.add_argument('--n_heads', type=int, default=24)
    parser.add_argument('--n_kv_heads', type=int, default=6)
    parser.add_argument('--head_dim', type=int, default=128)
    parser.add_argument('--vocab_size', type=int, default=1028)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()
    
    config = vars(args)
    dataset = TokenDataset(args.token_dir, args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    model = VisualTransformer(config).cuda()
    trainer = TransformerTrainer(model, config)
    
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in loader:
            loss = trainer.train_step(batch)
            total_loss += loss
        print(f"Epoch {epoch}: Loss {total_loss/len(loader):.4f}")
        if epoch % 10 == 0:
            trainer.save(f"checkpoint_{epoch}.pth")

if __name__ == '__main__':
    main()
"""

FILES["preencode.py"] = """
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from models.vqgan.vqgan import VQGAN

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqgan_ckpt', required=True)
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    
    device = 'cuda'
    model = VQGAN().to(device)
    model.load_state_dict(torch.load(args.vqgan_ckpt), strict=False)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for img_path in tqdm(list(Path(args.image_dir).glob('*.jpg'))):
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            _, indices, _ = model.encode(tensor)
        indices = indices.squeeze(0).flatten().cpu().numpy()
        indices = indices + 2  # offset for special tokens
        np.save(out_path / (img_path.stem + '.npy'), indices.astype(np.int32))

if __name__ == '__main__':
    main()
"""

# ============================================================
# 执行创建
# ============================================================
if __name__ == "__main__":
    # 1. 创建目录
    base_dir = Path(PROJECT_NAME)
    if base_dir.exists():
        print(f"删除已存在的目录: {base_dir}")
        shutil.rmtree(base_dir)
    
    print(f"正在生成项目: {base_dir}")
    
    for filepath, content in FILES.items():
        full_path = base_dir / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content.lstrip())  # 去除多余缩进但保留内部格式
        print(f"  ✅ 创建: {filepath}")
    
    # 2. 创建空 __init__ 补全
    (base_dir / "models/vqgan/__init__.py").touch()
    (base_dir / "models/transformer/__init__.py").touch()
    (base_dir / "trainers/__init__.py").touch()
    (base_dir / "datasets/__init__.py").touch()
    
    # 3. 生成启动说明
    readme = """
# Visual Transformer 0.39B (VQGAN + 20层 LLaMA式)
## 快速开始
1. 准备数据: `mkdir -p data/raw/train` 并将图片放入。
2. 训练 VQGAN: `python train_vqgan.py` (需自行实现简单循环，或参考 vqgan_config.yaml)
3. 预编码 Token: `bash scripts/preencode_dataset.sh`
4. 训练 Transformer: `bash scripts/train_transformer.sh`
5. 生成图片: 参考 `inference.py`
"""
    with open(base_dir / "README.md", "w") as f:
        f.write(readme)
    
    print(f"\n🎉 项目生成成功！请执行: cd {PROJECT_NAME}")