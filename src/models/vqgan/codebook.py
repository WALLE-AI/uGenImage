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
