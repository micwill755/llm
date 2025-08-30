import numpy as np

# takes 2d np array and flattens to 1d
def flatten(m):
    d_in, d_out = m.shape
    arr = np.zeros((d_in * d_out))
    count = 0

    for row in range(d_in):
        for col in range(d_out):
            arr[count] = float(m[row, col])
            count += 1
    
    return arr

# transpose matrix as a numpy.ndarray
def transpose(m):
    d_in, d_out = m.shape
    out = np.random.randn(d_out, d_in)

    for row in range(d_in):
        for col in range(d_out):
            out[col, row] = m[row, col]

    return out

# transpose matrix as 1d flattened array
def transpose_1d(arr, d_in, d_out):
    out = np.zeros((d_in * d_out))
    
    for row in range(d_in):
        for col in range(d_out):
            out[col * d_in + row] = arr[row * d_out + col]

    return out

# mat mul should handle batches
def mat_mul(m1, m2):
    m_d_in, m_d_out = m1.shape
    m2_d_in, m2_d_out = m2.shape
    # out will be square the smallest dimension of m2
    out = np.zeros((m_d_in, m2_d_out))

    for row in range(m_d_in):
        for col2 in range(m2_d_out):
            s = 0.0
            for col in range(m_d_out):
                s += float(m1[row][col] * m2[col][col2])
            out[row][col2] = s

    return out

def create_mask(height, width):
    out = np.ones((height, width))
    s_in = 1

    for row in range(height):
        for col in range(s_in, width):
            out[row][col] = 0
        s_in += 1
    
    return out

def apply_mask(m, mask):
    d_in, d_out = m.shape

    for row in range(d_in):
        for col in range(d_out):
            if mask[row][col] == 0:
                m[row][col] = -np.inf

def split_mat (m, emd_dim, i):
    d_in, d_out = m.shape
    out = np.zeros((d_in, emd_dim))
    out_i = 0

    for row in range(d_in):
        out_i = 0
        # we are taking elements from a start and end index
        for col in range(i, emd_dim):
            out[row][out_i] = m[row][col]
            out_i += 1

    return out

def combine_mat (m1, m2, emd_dim, i):
    d_in, d_out = m1.shape
    m2_col_i = 0

    for row in range(d_in):
        m2_col_i = 0
        for col in range(i, emd_dim):
            m1[row][col] = m2[row][m2_col_i]
            m2_col_i += 1

'''

For learning purposes we will create sepearte functions for handling 3D tensors so we can 
process attnetion heads in parallel

'''

# mat mul should handle batches
def mat_mul_nd(m1, m2):
    dims1 = list(m1.shape)
    dims2 = list(m2.shape)
    # output shape: (heads, m1_rows, m2_cols)
    out = np.zeros((dims1[0], dims1[1], dims2[2]))

    # do we simply need to step down until last 2 dimentions? 
    #def mat_mul_depth(depth):
        #for i in range(dims1):
    # TEMP focus on 3d tensor for now
    
    # for each head
    for h in range(dims1[0]):
        out[h] = mat_mul(m1[h], m2[h])

    return out

def transpose_nd(m, dim1, dim2):
    dims = list(m.shape)
    # swap specified dimensions
    dims[dim1], dims[dim2] = dims[dim2], dims[dim1] 
    out = np.zeros((dims))

    def fill_transposed(indices, depth):
        if depth == len(m.shape):
            # Create swapped indices for output
            out_indices = indices[:]
            out_indices[dim1], out_indices[dim2] = out_indices[dim2], out_indices[dim1]
            out[tuple(out_indices)] = m[tuple(indices)]
            return
        
        for i in range(m.shape[depth]):
            indices[depth] = i
            fill_transposed(indices, depth + 1)
    
    fill_transposed([0] * len(m.shape), 0)
    return out

def reshape (m, heads, emd_dim):
    num_tokens, d_out = m.shape
    out = np.zeros((heads, num_tokens, emd_dim))

    for head in range(heads):
        for token in range(num_tokens):
            start_dim = head * emd_dim
            for dim in range(emd_dim):
                out[head][token][dim] = m[token][start_dim + dim]

    return out

def apply_mask_nd(m, mask):
    heads, tokens1, tokens2 = m.shape
    
    for h in range(heads):
        apply_mask(m[h], mask)  # Apply 2D mask to each head

def combine_heads(m):
    heads, num_tokens, head_dim = m.shape
    out = np.zeros((num_tokens, heads * head_dim))

    for token in range(num_tokens):
        for head in range(heads):
            start_dim = head * head_dim
            for dim in range(head_dim):
                out[token][start_dim + dim] = m[head][token][dim]

    return out

'''# create a 2d np array
m = np.random.randn(2, 3)
print (m)
# transpose np array in np format
m_tranposed = transpose(m)
print(m_tranposed)

# working with matrices in 1d
# now flatten m to 1d array, this is how we represent matrices in C/C++
m_flattened_1d = flatten(m)
print (m_flattened_1d)
m_tranposed_1d = transpose_1d(m_flattened_1d, 2, 3)
print(m_tranposed_1d)

m2 = np.random.randn(2, 3)
m2_t = transpose(m2)
res = mat_mul(m, m2_t)
print(res.shape)
print(res)
print(m @ m2_t)'''