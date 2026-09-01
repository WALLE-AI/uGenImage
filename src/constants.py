"""全局 Token 约定 —— 唯一真相。

方案 A 原实现将 offset 硬编码在 preencode.py 与 inference.py 两处，
且 BOS 与 MASK 共用 id 1；此处集中定义，任何一处改动自动同步。
"""

PAD_ID = 0
BOS_ID = 1
MASK_ID = 2          # 与 BOS 分离（原实现两者共用 id 1）
CODEBOOK_OFFSET = 3  # 真实码本条目从此处开始

CODEBOOK_SIZE = 1024
VOCAB_SIZE = CODEBOOK_OFFSET + CODEBOOK_SIZE  # 1027

# 256x256 图像经 f=16 下采样 -> 16x16 潜码网格
LATENT_SIZE = 16
LATENT_TOKENS = LATENT_SIZE * LATENT_SIZE     # 256
SEQ_LEN = LATENT_TOKENS + 1                   # 257 = BOS + 256 个图像 token
