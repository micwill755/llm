import numpy as np
from attention import CausalAttention, ScaledDotProductAttention
from linear import Linear

### Ma

### LAYER NORMALIZATION


### LAYER NORMALIZATION

class LayerNorm():
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = np.ones(emb_dim)
        self.shift = np.zeros(emb_dim)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / np.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

### LAYER NORMALIZATION

### BLOCKS

class Block:
    def __init__(self, cfg):
        pass
    
    def forward(self, x):
        return x

class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        pass
    
    def forward(self, x):
        return x

class Embedding:
    def __init__(self, vocab_size, emb_dim):
        self.weight = np.random.randn(vocab_size, emb_dim)
    
    def forward(self, idx):
        return self.weight[idx]

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

class GPT2Model:
    def __init__(self, cfg):
        self.tok_emb = Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = Embedding(cfg["context_length"], cfg["emb_dim"])
        self.blocks = [Block(cfg) for _ in range(cfg["n_layers"])]
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb.forward(in_idx)
        pos_embeds = self.pos_emb.forward(np.arange(seq_len))
        x = tok_embeds + pos_embeds
        for block in self.blocks:
            x = block.forward(x)
        x = self.final_norm.forward(x)
        logits = self.out_head.forward(x)
        return logits

### MODEL

### Attention test

np.random.seed(42)
emd_dim = 5
x = np.random.randn(1, 3, emd_dim)  # (batch_size=1, num_tokens=3, emb_dim=5)

attention = CausalAttention(emd_dim, emd_dim, 5, 2)
res = attention.forward(x)
print('CausalAttention')
print(res)

attention = ScaledDotProductAttention(emd_dim, emd_dim, 5, 2)
res = attention.forward(x)
print('ScaledDotProductAttention')
print(res)

### MAIN -------

'''### Step 1: tokenzier

import tiktoken
import torch
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

### Step 1: tokenzier

### Step 2: initialize a model

torch.manual_seed(123)
model = GPT2Model(GPT_CONFIG_124M)
logits = model(batch)
print(f'Output shape {logits.shape}')
print(logits)

### Step 2: initialize a model

### MAIN -------'''