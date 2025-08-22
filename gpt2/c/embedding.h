#include <stdint.h>
#include <ctype.h>
#include <assert.h>
#include <stdlib.h>

typedef struct Embedding {
    float *weight; 
    int vocab_size;
    int emb_dim;
    
    void (* init)(struct Embedding *emb, int vocab_size, int emb_dim);
    float* (* forward)(struct Embedding *emb, int *input_ids, int num_tokens);
} Embedding;

void embedding_init(Embedding *emb, int vocab_size, int emb_dim) { 
    emb->vocab_size = vocab_size;
    emb->emb_dim = emb_dim;

    // Allocate weight matrix (vocab_size x emb_dim)
    emb->weight = (float*)malloc(vocab_size * emb_dim * sizeof(float));
    // Initialize with random values 
    for (int i = 0; i < vocab_size * emb_dim; i++) {
        emb->weight[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Random [-1, 1]
    }
}

float* embedding_forward(Embedding *emb, int *input_ids, int num_tokens) {
    float *output = (float*)malloc(num_tokens * emb->emb_dim * sizeof(float));
    
    for (int i = 0; i < num_tokens; i++) {
        int token_id = input_ids[i];
        // Copy embedding for this token
        for (int j = 0; j < emb->emb_dim; j++) {
            output[i * emb->emb_dim + j] = emb->weight[token_id * emb->emb_dim + j];
        }
    }
    
    return output;
}
