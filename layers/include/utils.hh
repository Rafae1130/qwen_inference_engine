#pragma once
#include <iostream>
#include "layers_include.cuh"
#include "tensor_parser.hh"
#include <unordered_map>

#include <cuda.h>

#define TILE_SIZE 16
#define WARP_SIZE 32

using TensorTable = std::unordered_map<std::string, std::vector<tensor>>;

struct ModelBuffers {
    // Token IDs
    int *d_token_ids;
    size_t sequence_len;
    
    // Model dimensions
    size_t number_of_layers;
    size_t head_dim;
    size_t hidden_dim;
    size_t hidden_dim_kv;
    size_t num_of_qheads;
    size_t num_of_kvheads;
    size_t context_size;
    size_t vocab_size;
    size_t up_dim;
    
    // Embeddings
    __nv_bfloat16 *embeddings_h;
    __nv_bfloat16 *embeddings_d;
    __nv_bfloat16 *embeddings_out;
    
    // RoPE cos/sin
    float *cos_values_h;
    float *sin_values_h;
    float *cos_values_d;
    float *sin_values_d;
    
    // KV cache
    __nv_bfloat16 *k_cache;
    __nv_bfloat16 *v_cache;
    
    // Layer weights and outputs
    __nv_bfloat16 *norm_weights_h;
    __nv_bfloat16 *norm_weights_d;
    __nv_bfloat16 *rms_out;
    
    __nv_bfloat16 *qk_norm_weights_h;
    __nv_bfloat16 *qk_norm_weights_d;
    
    __nv_bfloat16 *q_proj_weights_h;
    __nv_bfloat16 *q_proj_weights_d;
    __nv_bfloat16 *Q;
    size_t q_proj_size;
    
    __nv_bfloat16 *kv_proj_weights_h;
    __nv_bfloat16 *kv_proj_weights_d;
    size_t kv_proj_size;
    __nv_bfloat16 *K;
    __nv_bfloat16 *V;
    
    __nv_bfloat16 *atten_out;
    
    __nv_bfloat16 *o_proj_weights_h;
    __nv_bfloat16 *o_proj_weights_d;
    __nv_bfloat16 *O;
    __nv_bfloat16 *out_proj;
    size_t o_proj_size;
    
    // MLP
    __nv_bfloat16 *mlp_up_proj_weights_h;
    __nv_bfloat16 *mlp_up_proj_weights_d;
    __nv_bfloat16 *MLP_UP;
    __nv_bfloat16 *MLP_GATE;
    __nv_bfloat16 *MLP_GATE_OUT;
    __nv_bfloat16 *MLP_DOWN;
    size_t mlp_up_proj_size;
    
    __nv_bfloat16 *test_out;

    __nv_bfloat16* last_x;
    __nv_bfloat16* prefill_output_d;
    __nv_bfloat16* logits_weights_h;
    __nv_bfloat16* logits_weights_d;
    size_t logtis_shape;  
};

void initialize_model_buffers(ModelBuffers &buf, int *h_token_ids, 
                              TensorTable &tensors,
                              std::ifstream &weights, size_t sequence_len);






static inline void tryCudaFree(void* p);

static inline void tryCudaFreeHost(void* p);

static inline void tryFree(void* p);

void destroy_model_buffers(ModelBuffers& buf);



