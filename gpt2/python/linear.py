import numpy as np
from matrix_helper import transpose, mat_mul, reshape

class Linear:
    def __init__(self, d_in, d_out, bias=True):
        self.weight = np.random.randn(d_out, d_in)
        self.bias = np.random.randn(d_out) if bias else None
    
    # y = mx + b
    def forward(self, x):
        batch_size, num_tokens, emb_dim = x.shape

        # Flatten to 2D: (batch_size * num_tokens, emb_dim)
        x_2d = x.reshape(batch_size * num_tokens, emb_dim)
        
        # Apply linear transformation
        out_2d = mat_mul(x_2d, transpose(self.weight))
        
        # Reshape back: (batch_size, num_tokens, d_out)
        out = out_2d.reshape(batch_size, num_tokens, self.weight.shape[0])
        
        if self.bias is not None:
            out = out + self.bias
            
        return out
