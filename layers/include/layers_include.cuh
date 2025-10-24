#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <fstream>

#include "iengine.cuh"
#include "utils.hh"


struct ModelBuffers;
struct page_table_struct;
typedef struct page_table_struct page_table;
struct tensor;
__global__ void embedding_matrix_func(__nv_bfloat16 *embeddings_out, __nv_bfloat16 *embeddings_matrix,int *token_ids, size_t embedding_dim, size_t sequence_len);
// __global__ void matrix_mul(const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16*C, int M, int N, int K, int lda, int ldb, int ldc);
__global__ void matrix_mul(__nv_bfloat16 *A, __nv_bfloat16 *B, __nv_bfloat16 *C, int M, int N, int K);

__global__ void rmsNorm(__nv_bfloat16*embedding_matrix, __nv_bfloat16*norm_weigths, __nv_bfloat16 *rms_out, size_t hidden_dim, size_t sequence_len);
__global__ void qkNorm(__nv_bfloat16*qk, __nv_bfloat16*norm_weigths ,int head_dim, int sequence_len, int hidden_dim);


__global__ void RoPE(float *cos_values, float *sin_values, __nv_bfloat16*embedded_matrix, int seq_len, int head_dim, int embedding_dim, int num_heads);
__global__ void selfattention(__nv_bfloat16*query_vec, __nv_bfloat16*output, size_t seq_len_q, size_t seq_len_kv, size_t head_dim, size_t hidden_dim, size_t kv_dim, int causal, size_t q_abs_base, int layer_id, page_table* kv_cache_seq1, int page_size);
__global__ void activation( __nv_bfloat16 *matrix, size_t size);

__global__ void residual_add(__nv_bfloat16* __restrict__ mat_a, __nv_bfloat16* __restrict__ mat_b, size_t num_of_elements);

__global__ void append_kv(__nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ k_cache, __nv_bfloat16* __restrict__ v_cache, size_t current_seq_ind ,size_t sequence_len, size_t kv_dim);
__global__ void element_mul(__nv_bfloat16 *a, __nv_bfloat16 *b, __nv_bfloat16 *c, int size);
__global__ void topk_temperature_softmax_sampling_kernel_bf16( const __nv_bfloat16* __restrict__ logits, int* __restrict__ output_token, float temperature, int k, size_t vocab_size, size_t seed, size_t subseq); // e.g., sampling step

__global__ void apply_repetition_penalty_kernel(__nv_bfloat16* logits,const int* context_tokens, size_t context_len,int vocab_size, float penalty);

__global__ void dump_top_logits(const __nv_bfloat16* logits, size_t vocab, int k);

void weights_read(tensor t, std::ifstream& weights_file, __nv_bfloat16* weights_buffer);

// ---------- Tensor Size ----------
size_t calculate_tensor_size(const tensor& t);

// ---------- Memory Initializers ----------
void init_weights_memory(__nv_bfloat16** host_ptr, __nv_bfloat16** device_ptr_in, __nv_bfloat16** device_ptr_out, const tensor& t);
void init_weights_memory(__nv_bfloat16** host_ptr, __nv_bfloat16** device_ptr, const tensor& t);
void init_weights_memory(__nv_bfloat16** device_ptr, const tensor& t);

// ---------- Utility ----------
int find_max(const __nv_bfloat16* array, int size);

inline void log_cache_host_range( std::ofstream& out,  const __nv_bfloat16* d_cache_layer_base, int hidden_dim_kv, int layer_idx,  const char* tag,  int start_pos,  int end_pos,  int max_cols);

void kv_copy_layer_to_cache_prefill(ModelBuffers* buffer, int i, page_table* kv_cache_seq1, int page_size);

void kv_copy_layer_to_cache_decode(ModelBuffers* buffer, int i, page_table* kv_cache_seq1, int page_size);
