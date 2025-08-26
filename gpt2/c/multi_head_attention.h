#include <stdint.h>
#include <ctype.h>
#include <assert.h>
#include <stdlib.h>

typedef struct MultiHeadAttention {
    int d_out;
    int num_heads;
    int head_dim;

    void (*init) (int d_in, int d_out, int context_length, int dropout, bool qkv_bias);
    void (*forward) (MultiHeadAttention *attention, char *x);
} MultiHeadAttention;

