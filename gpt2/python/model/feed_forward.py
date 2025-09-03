import numpy as np
from attention import CausalAttention, ScaledDotProductAttention, MultiHeadAttention
from linear import Linear
from activation_functions import GELU, SwiGLU

class FeedForwardGPT():
    def __init__(self, cfg):
        self.linear1 = Linear(cfg["emb_dim"], 4 * cfg["emb_dim"])
        self.gelu = GELU()
        self.linear2 = Linear(4 * cfg["emb_dim"], cfg["emb_dim"])

    def forward(self, x):
        x = self.linear1.forward(x)
        x = self.gelu.forward(x)
        x = self.linear2.forward(x)
        return x

class FeedForwardLlama():
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)
        self.silu = SwiGLU()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)