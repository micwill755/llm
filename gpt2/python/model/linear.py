import numpy as np
from matrix_helper import transpose, mat_mul, reshape

class Linear:
    def __init__(self, d_in, d_out, bias=True):
        self.weight = np.random.randn(d_out, d_in)
        self.bias = np.random.randn(d_out) if bias else None
    
    # y = mx + b
    def forward(self, x):
        if len(x.shape) == 2:  # 2D input (tokens, emb_dim)
            out = mat_mul(x, transpose(self.weight))
            if self.bias is not None:
                out = out + self.bias
            return out
        elif len(x.shape) == 3:  # 3D input (batch, tokens, emb_dim)
            batch_size, num_tokens, emb_dim = x.shape
            d_out = self.weight.shape[0]
            out = np.zeros((batch_size, num_tokens, d_out))
            
            weight_t = transpose(self.weight)
            
            for batch in range(batch_size):
                out[batch] = mat_mul(x[batch], weight_t)
                if self.bias is not None:
                    out[batch] = out[batch] + self.bias
            
            return out
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
