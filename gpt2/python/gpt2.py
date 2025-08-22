import numpy as np

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = np.random.randn(out_features, in_features)
        self.bias = np.random.randn(out_features) if bias else None
    
    # y = mx + b
    def forward(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out += self.bias
        return out

### ATTENTION

class MultiHeadAttention:
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.dropout = dropout

        self.W_query = Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = Linear(d_out, d_out)
        self.mask = np.triu(np.ones((context_length, context_length)), k=1)

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key.forward(x)
        queries = self.W_query.forward(x)
        values = self.W_value.forward(x)

        keys = keys.reshape(b, num_tokens, self.num_heads, self.head_dim)
        values = values.reshape(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.reshape(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask[:num_tokens, :num_tokens].astype(bool)
        attn_scores = np.where(mask_bool, -np.inf, attn_scores)

        attn_weights = self.softmax(attn_scores / np.sqrt(keys.shape[-1]))
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj.forward(context_vec)

        return context_vec
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

### ATTENTION

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

### MAIN -------

### Step 1: tokenzier

import tiktoken

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

### MAIN -------