import numpy as np
from matrix_helper import transpose, mat_mul

class Linear:
    def __init__(self, d_in, d_out, bias=True):
        self.weight = np.random.randn(d_out, d_in)
        self.bias = np.random.randn(d_out) if bias else None
    
    # y = mx + b
    def forward(self, x):
        out = mat_mul(x, transpose(self.weight))
        if self.bias is not None:
            out += self.bias
        return out
