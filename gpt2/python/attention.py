import numpy as np
from matrix_helper import transpose, mat_mul, create_mask, apply_mask, split_mat, combine_mat
from linear import Linear

def softmax(m):
    n_tokens_d1, n_tokens_d2 = m.shape
    out = np.zeros((n_tokens_d1, n_tokens_d2))

    for i in range(n_tokens_d1):
        sum = 0
        # first calculate the sum
        for j in range(n_tokens_d2):
            sum += np.exp(m[i][j])
        
        # then divide each to get weight of 1
        for j in range(n_tokens_d2):
            out[i][j] = np.exp(m[i][j]) / sum

    return out

'''
Self Attention:

- Each token can attend to ALL other tokens (including future ones)
- No masking applied
- Used in encoders (like BERT)
'''
class SelfAttention:
    def __init__ (self, d_in, d_out, context_length, dropout, qkv_bias=False):
        self.d_out = d_out
        self.d_in = d_in
        self.droput = dropout

        self.query = Linear(d_in, d_out, bias=qkv_bias)
        self.key = Linear(d_in, d_out, bias=qkv_bias)
        self.value = Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = Linear(d_out, d_out)

    def forward(self, x):
        b, num_tokens, emd_dim = x.shape
        results = []

        # temp until we handle parallel
        for i in range(b):
            x_batch = x[i]

            query_w = self.query.forward(x_batch)
            key_w = self.key.forward(x_batch)
            value_w = self.value.forward(x_batch)

            att_scores = mat_mul(query_w, transpose(key_w))
            attn_weights = softmax(att_scores)
            context = mat_mul(attn_weights, value_w)
            context = self.out_proj.forward(context)
            results.append(context)
        
        return np.stack(results, axis=0)
            
'''
Causal Attention:

- Each token can only attend to previous tokens and itself
- Uses causal mask to block future tokens
- Used in decoders (like GPT)

'''

class CausalAttention:
    def __init__ (self, d_in, d_out, context_length, dropout, qkv_bias=False):
        self.d_out = d_out
        self.d_in = d_in
        self.droput = dropout

        self.query = Linear(d_in, d_out, bias=qkv_bias)
        self.key = Linear(d_in, d_out, bias=qkv_bias)
        self.value = Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = Linear(d_out, d_out)
        self.mask = create_mask(context_length, context_length)

    def forward(self, x):
        b, num_tokens, emd_dim = x.shape
        results = []

        # temp until we handle parallel
        for i in range(b):
            x_batch = x[i]

            query_w = self.query.forward(x_batch)
            key_w = self.key.forward(x_batch)
            value_w = self.value.forward(x_batch)

            att_scores = mat_mul(query_w, transpose(key_w))
            apply_mask(att_scores, self.mask)
            attn_weights = softmax(att_scores)
            context = mat_mul(attn_weights, value_w)
            context = self.out_proj.forward(context)
            results.append(context)
        
        return np.stack(results, axis=0)

'''
Scaled dot product Attention:

Scaled Dot-Product Attention is the mathematical formula:
Attention(Q,K,V) = softmax(QK^T / √d_k)V

The key feature is scaling by √d_k (square root of key dimension) to prevent extremely large dot products.

Difference from Causal Attention:

Scaled Dot-Product: Refers to the mathematical operation (with scaling)
Causal Attention: Refers to the masking pattern (blocking future tokens)

'''
class ScaledDotProductAttention:
    def __init__ (self, d_in, d_out, context_length, dropout, qkv_bias=False):
        self.d_out = d_out
        self.d_in = d_in
        self.droput = dropout

        self.query = Linear(d_in, d_out, bias=qkv_bias)
        self.key = Linear(d_in, d_out, bias=qkv_bias)
        self.value = Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = Linear(d_out, d_out)
        self.mask = create_mask(context_length, context_length)

    def forward(self, x):
        b, num_tokens, emd_dim = x.shape
        results = []

        # temp until we handle parallel
        for i in range(b):
            x_batch = x[i]

            query_w = self.query.forward(x_batch)
            key_w = self.key.forward(x_batch)
            value_w = self.value.forward(x_batch)

            # we only add
            att_scores = mat_mul(query_w, transpose(key_w)) / np.sqrt(self.d_out)
            apply_mask(att_scores, self.mask)
            attn_weights = softmax(att_scores)
            context = mat_mul(attn_weights, value_w)
            context = self.out_proj.forward(context)
            results.append(context)
        
        return np.stack(results, axis=0)
    

# temp for demonstration
class MultiHeadCausualAttention:
    def __init__ (self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        self.num_heads = num_heads
        self.heads = [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for i in range(num_heads)]

    def forward(self, x):
        results = []
        for i in range(self.num_heads):
            results.append(self.heads[i].forward(x))

        return results
    
'''

Multi-head attention splits the embedding dimensions across multiple parallel attention heads, each 
learning different relationships. For example, with input shape (batch, 4_tokens, 512_dims) and 8_heads: 
each head gets 512/8 = 64 dimensions per token, computes its own (4×4) attention matrix on those 64 dimensions, 
outputs (4_tokens, 64_dims), then all 8 head outputs are concatenated back to (4_tokens, 512_dims). 

This allows the model to simultaneously capture different types of relationships (like syntax in head 1, semantics in head 2) 
while maintaining the same overall dimensionality.

    Input: (batch, 4 tokens, 512 dims)
                    |
            Split into 8 heads
                    |
    ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
    │Head1│Head2│Head3│Head4│Head5│Head6│Head7│Head8│
    │ 64  │ 64  │ 64  │ 64  │ 64  │ 64  │ 64  │ 64  │
    │dims │dims │dims │dims │dims │dims │dims │dims │
    └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
         |     |     |     |     |     |     |     |
    Each head computes (4×4) attention matrix
         |     |     |     |     |     |     |     |
    ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
    │ 4×64│ 4×64│ 4×64│ 4×64│ 4×64│ 4×64│ 4×64│ 4×64│
    └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                    |
                Concatenate
                    |
        Output: (batch, 4 tokens, 512 dims)

'''

class MultiHeadAttention:
    def __init__ (self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        self.d_out = d_out
        self.d_in = d_in
        self.droput = dropout
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.query = Linear(d_in, d_out, bias=qkv_bias)
        self.key = Linear(d_in, d_out, bias=qkv_bias)
        self.value = Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = Linear(d_out, d_out)

        self.mask = create_mask(context_length, context_length)

    def forward(self, x):
        b, num_tokens, emd_dim = x.shape
        results = []

        # temp until we handle parallel
        for i in range(b):
            x_batch = x[i]

            query_w = self.query.forward(x_batch)
            key_w = self.key.forward(x_batch)
            value_w = self.value.forward(x_batch)

            all_context = np.zeros((num_tokens, self.d_out))

            for j in range(self.num_heads):
                q = split_mat(query_w, self.head_dim, self.head_dim * j)
                k = split_mat(key_w, self.head_dim, self.head_dim * j)
                v = split_mat(value_w, self.head_dim, self.head_dim * j)

                att_scores = mat_mul(q, transpose(k)) / np.sqrt(self.head_dim)
                apply_mask(att_scores, self.mask)
                attn_weights = softmax(att_scores)
                head_context = mat_mul(attn_weights, v)
                combine_mat(all_context, head_context,  self.head_dim, self.head_dim * j)

            final_context = self.out_proj.forward(all_context)
            results.append(final_context)
        
        return np.stack(results, axis=0)