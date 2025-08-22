# C Matrix Representation and Linear Layers

## Matrix Storage in C

2D matrices are stored as 1D arrays in **row-major order**:

```c
// Matrix (3x2):        1D Array:
// [a b]             →  [a, b, c, d, e, f]
// [c d]
// [e f]
```

### Accessing Elements
For a matrix of shape `(rows, cols)`:
```c
// Access element at row i, column j:
matrix[i * cols + j]

// Example: weight matrix (features_out, features_in)
weight[row * features_in + col]
```

## Linear Layer Dimensions

**d_in** = number of input features (columns in weight matrix)
**d_out** = number of output features (rows in weight matrix)

### Example: Transform 3D input to 2D output
```c
// Input: [x1, x2, x3] (3 features)
// Output: [y1, y2] (2 features)
// d_in = 3, d_out = 2

// Weight matrix shape: (d_out, d_in) = (2, 3)
weight = [[w11, w12, w13],  // row 0 (for output y1)
          [w21, w22, w23]]  // row 1 (for output y2)

// Matrix multiplication:
// y1 = w11*x1 + w12*x2 + w13*x3
// y2 = w21*x1 + w22*x2 + w23*x3
```

### In Code:
- `weight[i * d_in + j]` accesses weight[i][j]
- `i` loops over d_out (output features/rows)
- `j` loops over d_in (input features/columns)
- Weight matrix is always (d_out × d_in), transforming d_in features to d_out features