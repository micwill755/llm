import numpy as np
from linear import Linear

# GELU (Gaussian Error Linear Unit)
# Used in: GPT-2, BERT, many transformers
# Formula: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
class GELU():
    def forward(self, x):
        return 0.5 * x * (1 + np.tanh(
            np.sqrt(2.0 / np.pi) *
            (x + 0.044715 * np.power(x, 3))
        ))

# SiLU (Sigmoid Linear Unit) is the base activation function used in SwiGLU!
# SiLU (also called Swish)
# Formula: x * sigmoid(x) = x * (1 / (1 + e^(-x)))
class SiLU:
    def forward(self, x):
        return x * (1 / (1 + np.exp(-x)))

#SwiGLU (Swish-Gated Linear Unit)
#Used in: LLaMA, PaLM, modern large models
#Formula: Swish(xW + b) ⊙ (xV + c) where Swish(x) = x * sigmoid(x)  
class SwiGLU:
    def __init__(self, d_in, d_hidden):
        self.w_gate = Linear(d_in, d_hidden)
        self.w_up = Linear(d_in, d_hidden) 
        self.w_down = Linear(d_hidden, d_in)
    
    def forward(self, x):
        gate = self.w_gate.forward(x)
        up = self.w_up.forward(x)
        # Swish activation: x * sigmoid(x)
        swish_gate = gate * (1 / (1 + np.exp(-gate)))
        # Element-wise multiply (gating)
        gated = swish_gate * up
        return self.w_down.forward(gated)