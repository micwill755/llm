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
        
        for batch in range(batch_size):
            for token in range(num_tokens):
                for out_dim in range(d_out):
                    sum_val = 0.0
                    for in_dim in range(emb_dim):
                        sum_val += x[batch][token][in_dim] * self.weight[out_dim][in_dim]
                    
                    if self.bias is not None:
                        sum_val += self.bias[out_dim]
                    
                    out[batch][token][out_dim] = sum_val
        
        return out
