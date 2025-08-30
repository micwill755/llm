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


# create a 2d np array
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
print(m @ m2_t)