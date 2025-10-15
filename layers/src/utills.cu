#include "utils.hh"
#include <cuda.h>

void initialize_model_buffers(ModelBuffers &buf, int *h_token_ids, 
                              TensorTable &tensors,
                              std::ifstream &weights, size_t sequence_len) {
    // Model dimensions
    buf.number_of_layers = 40;
    buf.head_dim = 128;
    buf.hidden_dim = 5120;
    buf.hidden_dim_kv = 1024;
    buf.num_of_qheads = 40;
    buf.num_of_kvheads = 8;
    buf.context_size = 32786;
    buf.vocab_size = 151936;
    buf.up_dim = 17408;
    
    // Token IDs
    buf.sequence_len = sequence_len;
    cudaMalloc((void **)&buf.d_token_ids, buf.sequence_len * sizeof(int));
    cudaMemcpy(buf.d_token_ids, h_token_ids, buf.sequence_len * sizeof(int), cudaMemcpyHostToDevice);
    
    // Embeddings
    tensor embedding_weights = tensors["embed_tokens.weight"][0];
    int total_embedding_size = embedding_weights.shape[0] * embedding_weights.shape[1];
    int total_embedding_size_in_bytes = total_embedding_size * sizeof(__nv_bfloat16);
    int required_embedding_size = buf.sequence_len * buf.hidden_dim;

    
    buf.embeddings_h = ( __nv_bfloat16 *)malloc(total_embedding_size_in_bytes);
    init_weights_memory(&buf.embeddings_d, &buf.embeddings_out, embedding_weights);
    weights_read(embedding_weights, weights, buf.embeddings_h);
    cudaMemcpy(buf.embeddings_d, buf.embeddings_h, total_embedding_size_in_bytes, cudaMemcpyHostToDevice);
    
    // RoPE cos/sin
    size_t cos_sin_size_in_bytes = buf.context_size * (buf.head_dim / 2) * sizeof(float);
    buf.cos_values_h = (float *)malloc(cos_sin_size_in_bytes);
    buf.sin_values_h = (float *)malloc(cos_sin_size_in_bytes);
    precompute_cos_sin(buf.cos_values_h, buf.sin_values_h, buf.context_size, buf.head_dim);
    
    cudaMalloc((void **)&buf.cos_values_d, cos_sin_size_in_bytes);
    cudaMalloc((void **)&buf.sin_values_d, cos_sin_size_in_bytes);
    cudaMemcpy(buf.cos_values_d, buf.cos_values_h, cos_sin_size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(buf.sin_values_d, buf.sin_values_h, cos_sin_size_in_bytes, cudaMemcpyHostToDevice);
    
    // KV cache
    size_t kv_cache_size = size_t(buf.context_size) * size_t(buf.hidden_dim_kv) * size_t(buf.number_of_layers);
    cudaMalloc((void **)&buf.k_cache, kv_cache_size * sizeof(__nv_bfloat16));
    cudaMalloc((void **)&buf.v_cache, kv_cache_size * sizeof(__nv_bfloat16));
    std::cout << "kv cache size: " <<kv_cache_size * sizeof(__nv_bfloat16) << std::endl;
    // Launch embedding kernel
    int threads = 256;
    int blocks = (buf.sequence_len + threads - 1) / threads;
    embedding_matrix_func<<<blocks, threads>>>(buf.embeddings_out, buf.embeddings_d, 
                                               buf.d_token_ids, buf.hidden_dim, buf.sequence_len);
    
    // RMS norm
    tensor norm_tensor_temp = tensors["input_layernorm.weight"][0];
    init_weights_memory(&buf.norm_weights_h, &buf.norm_weights_d, norm_tensor_temp);
    cudaMalloc((void **)&buf.rms_out, required_embedding_size * sizeof(__nv_bfloat16));
    
    // QK norm
    tensor qk_norm_tensor_temp = tensors["self_attn.q_norm.weight"][0];
    init_weights_memory(&buf.qk_norm_weights_h, &buf.qk_norm_weights_d, qk_norm_tensor_temp);
    
    // Q projection
    tensor q_proj_tensor_temp = tensors["self_attn.q_proj.weight"][0];
    init_weights_memory(&buf.q_proj_weights_h, &buf.q_proj_weights_d, q_proj_tensor_temp);
    int Q_size = buf.sequence_len * buf.hidden_dim;                // activation size
    buf.q_proj_size = q_proj_tensor_temp.shape[0] *                // **weight size**
                      q_proj_tensor_temp.shape[1];
    cudaMalloc((void **)&buf.Q, Q_size * sizeof(__nv_bfloat16));
    
    // K projection
    tensor kv_proj_tensor_temp = tensors["self_attn.k_proj.weight"][0];
    init_weights_memory(&buf.kv_proj_weights_h, &buf.kv_proj_weights_d, kv_proj_tensor_temp);
    int KV_size = buf.sequence_len * buf.hidden_dim_kv;            // activation size
    buf.kv_proj_size = kv_proj_tensor_temp.shape[0] *              // **weight size**
                       kv_proj_tensor_temp.shape[1];
    cudaMalloc((void **)&buf.K, KV_size * sizeof(__nv_bfloat16));
    
    // V projection
    cudaMalloc((void **)&buf.V, KV_size * sizeof(__nv_bfloat16));
    
    // Attention output
    int atten_out_size = buf.sequence_len * buf.hidden_dim;
    cudaMalloc((void **)&buf.atten_out, atten_out_size * sizeof(__nv_bfloat16));
    
    // O projection
    tensor o_proj_tensor_temp = tensors["self_attn.o_proj.weight"][0];
    init_weights_memory(&buf.o_proj_weights_h, &buf.o_proj_weights_d, o_proj_tensor_temp);
    int O_size = buf.sequence_len * buf.hidden_dim;                // activation size
    buf.o_proj_size = o_proj_tensor_temp.shape[0] *                // **weight size**
                      o_proj_tensor_temp.shape[1];
    cudaMalloc((void **)&buf.O, O_size * sizeof(__nv_bfloat16));
    cudaMalloc((void **)&buf.out_proj, atten_out_size * sizeof(__nv_bfloat16));
    
    // MLP
    tensor mlp_up_proj_tensor_temp = tensors["mlp.up_proj.weight"][0];
    init_weights_memory(&buf.mlp_up_proj_weights_h, &buf.mlp_up_proj_weights_d, mlp_up_proj_tensor_temp);
    
    int MLP_UP_size = buf.sequence_len * buf.up_dim;               // activation size
    buf.mlp_up_proj_size = mlp_up_proj_tensor_temp.shape[0] *      // **weight size**
                           mlp_up_proj_tensor_temp.shape[1];
    cudaMalloc((void **)&buf.MLP_UP, MLP_UP_size * sizeof(__nv_bfloat16));
    cudaMalloc((void **)&buf.MLP_GATE, MLP_UP_size * sizeof(__nv_bfloat16));
    cudaMalloc((void **)&buf.MLP_GATE_OUT, MLP_UP_size * sizeof(__nv_bfloat16));
    
    int MLP_DOWN_size = buf.sequence_len * buf.hidden_dim;         // activation size
    cudaMalloc((void **)&buf.MLP_DOWN, MLP_DOWN_size * sizeof(__nv_bfloat16));
    
    // Test output
    buf.test_out = (__nv_bfloat16 *)malloc(buf.vocab_size * sizeof(__nv_bfloat16));


        // --- Final logits projection buffers
    buf.logtis_shape = static_cast<size_t>(buf.vocab_size) * static_cast<size_t>(buf.hidden_dim);

    // host weights (malloc, not pinned)
    buf.logits_weights_h = (__nv_bfloat16*) malloc(buf.logtis_shape * sizeof(__nv_bfloat16));

    // device weights
    cudaMalloc((void**)&buf.logits_weights_d, buf.logtis_shape * sizeof(__nv_bfloat16));

    // last token activation (1 x hidden_dim)
    cudaMalloc((void**)&buf.last_x, buf.hidden_dim * sizeof(__nv_bfloat16));

    // output logits buffer (vocab_size)
    cudaMalloc((void**)&buf.prefill_output_d, buf.vocab_size * sizeof(__nv_bfloat16));
}


static inline void tryCudaFree(void* p) {
    if (p) cudaFree(p);
}
static inline void tryCudaFreeHost(void* p) {
    if (p) cudaFreeHost(p);
}
static inline void tryFree(void* p) {
    if (p) free(p);
}

void destroy_model_buffers(ModelBuffers& buf)
{
    // Make sure all work is done
    cudaDeviceSynchronize();

    // --- Device frees (cudaMalloc)
    tryCudaFree(buf.d_token_ids);
    tryCudaFree(buf.embeddings_d);
    tryCudaFree(buf.embeddings_out);   // allocated by init_weights_memory
    tryCudaFree(buf.cos_values_d);
    tryCudaFree(buf.sin_values_d);
    tryCudaFree(buf.k_cache);
    tryCudaFree(buf.v_cache);

    tryCudaFree(buf.norm_weights_d);
    tryCudaFree(buf.rms_out);

    tryCudaFree(buf.qk_norm_weights_d);

    tryCudaFree(buf.q_proj_weights_d);
    tryCudaFree(buf.Q);

    tryCudaFree(buf.kv_proj_weights_d);
    tryCudaFree(buf.K);
    tryCudaFree(buf.V);

    tryCudaFree(buf.atten_out);

    tryCudaFree(buf.o_proj_weights_d);
    tryCudaFree(buf.O);
    tryCudaFree(buf.out_proj);

    tryCudaFree(buf.mlp_up_proj_weights_d);
    tryCudaFree(buf.MLP_UP);
    tryCudaFree(buf.MLP_GATE);
    tryCudaFree(buf.MLP_GATE_OUT);
    tryCudaFree(buf.MLP_DOWN);

    // --- Host frees
    // Pinned host (allocated by init_weights_memory via cudaMallocHost)
    // tryCudaFreeHost(buf.embeddings_h);
    tryCudaFreeHost(buf.norm_weights_h);
    tryCudaFreeHost(buf.qk_norm_weights_h);
    tryCudaFreeHost(buf.q_proj_weights_h);
    tryCudaFreeHost(buf.kv_proj_weights_h);
    tryCudaFreeHost(buf.o_proj_weights_h);
    tryCudaFreeHost(buf.mlp_up_proj_weights_h);

    // Regular host allocations (malloc/new)
    tryFree(buf.cos_values_h);   
    tryFree(buf.sin_values_h);   
    tryFree(buf.test_out);       
    tryFree(buf.embeddings_h);

    tryCudaFree(buf.last_x);
    tryCudaFree(buf.prefill_output_d);
    tryCudaFree(buf.logits_weights_d);

    // host
    tryFree(buf.logits_weights_h);

    // --- Null out everything to avoid dangling pointers
    buf = ModelBuffers{};
}
