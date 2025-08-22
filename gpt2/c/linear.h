#include <stdint.h>
#include <ctype.h>
#include <assert.h>
#include <stdlib.h>

//the lienar layer. 
typedef struct Linear {
    float *weight; 
    int vocab_size;
    int emb_dim;
    
    void (* init)(struct Linear *linear, int vocab_size, int emb_dim);
    float* (* forward)(struct Linear *linear, char *x);
} Embedding;