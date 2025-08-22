# C Matrix Representation and Function Pointers

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