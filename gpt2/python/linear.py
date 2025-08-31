import numpy as np
from matrix_helper import transpose, mat_mul, reshape

class Linear:
    def __init__(self, d_in, d_out, bias=True):
        self.weight = np.random.randn(d_out, d_in)
        self.bias = np.random.randn(d_out) if bias else None
    
    # y = mx + b
    def forward(self, x):
        batch_size, num_tokens, emb_dim = x.shape
        d_out = self.weight.shape[0]
        out = np.zeros((batch_size, num_tokens, d_out))
        
        weight_t = transpose(self.weight)
        
        for batch in range(batch_size):
            # Apply mat_mul to each batch's 2D slice
            out[batch] = mat_mul(x[batch], weight_t)
            
            if self.bias is not None:
                out[batch] = out[batch] + self.bias
        
        return out
