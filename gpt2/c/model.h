#include <stdbool.h>
#include "embedding.h"

typedef struct {
    int vocab_size;
    int context_length;
    int emb_dim;
    int n_heads;
    int n_layers;
    float drop_rate;
    bool qkv_bias;
} GPTConfig;

typedef struct GPTModel {
    int emb_dim;
    Embedding tok_embeds;
    Embedding pos_embeds;
    
    // Method pointers
    void (*init)(GPTConfig *config);
    int (*forward)(char *text, int *tokens, int in_idx);
} GPTModel;

void GPTModel_init(GPTModel *model, GPTConfig *config) {
    embedding_init(&model->tok_embeds, config->vocab_size, model->emb_dim);
    embedding_init(&model->pos_embeds, config->context_length, model->emb_dim);
}