import numpy as np

class RMSNorm():
    def __init__(self, emb_dim, eps=1e-5):
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = np.ones(emb_dim)

    def forward(self, x):
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * np.sqrt(means + self.eps)
        return (x_normed * self.weight).to(dtype=x.dtype)

class LayerNorm():
    def __init__(self, emb_dim):
        self.eps = 1e-5
        self.scale = np.ones(emb_dim)
        self.shift = np.zeros(emb_dim)

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        norm_x = (x - mean) / np.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift