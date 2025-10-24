#pragma once

#include<cuda_bf16.h>
#include<cuda.h>
#include "layers_include.cuh"
// #include "utils.hh"
#include "iengine.cuh"

static inline dim3 grid2D(int m, int k, int bx=16, int by=16) {
    return dim3((k + bx - 1)/bx, (m + by - 1)/by);
}
static inline dim3 block2D(int bx=16, int by=16){ return dim3(bx, by); }




// Assign pointer to specific tensor location in the unified GPU buffer
template<class T>
void assign_weight_pointer(const tensor& t, T*& d, __nv_bfloat16* g_gpu_weights_buffer ) {
    if (g_gpu_weights_buffer == nullptr) {
        std::cout << "Error: GPU weights buffer not initialized. Call load_all_weights_to_gpu first.\n";
        return;
    }
    
    size_t offset_bytes = t.data_offsets[0];
    
    // Calculate pointer offset
    d = reinterpret_cast<T*>(reinterpret_cast<char*>(g_gpu_weights_buffer) + offset_bytes);
}

// Optional: Original load_weight function updated to use the new approach
template<class T>
void load_weight(const tensor& t, std::ifstream& f, T* h, T*& d, size_t elems, __nv_bfloat16* g_gpu_weights_buffer) {
    assign_weight_pointer(t, d, g_gpu_weights_buffer);
}



// template<class T>
// void load_weight(const tensor& t, std::ifstream& f, T* h, T* d, size_t elems) {
//     weights_read(t, f, h);
//     cudaMemcpy(d, h, elems*sizeof(T), cudaMemcpyHostToDevice);
// }

void launch_rms(__nv_bfloat16* x, __nv_bfloat16* w, __nv_bfloat16* y,
                              size_t hidden, size_t seqlen) {
    int threads = 256, blocks = (seqlen + threads - 1)/threads;
    rmsNorm<<<blocks, threads>>>(x, w, y, hidden, seqlen);
}

 void launch_rope(float* cos_d, float* sin_d, __nv_bfloat16* x,
                               size_t seqlen, size_t head_dim, size_t hidden_dim, size_t nheads) {
    int threads = 256, blocks = (seqlen + threads - 1)/threads;
    RoPE<<<blocks, threads>>>(cos_d, sin_d, x, seqlen, head_dim, hidden_dim, nheads);
}

//  void launch_matmul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C,
//                                  int m, int n, int k) {
//     auto g = grid2D(m, k); auto b = block2D();
//     matrix_mul<<<g, b>>>(A, B, C, m, n, k);
// }

// Launch wrapper function
// void launch_matmul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C,
//                    int M, int N, int K) {
//     // Tiles along M and K (output dimensions)
//     int tilesM = (M + TILE_SIZE - 1) / TILE_SIZE;
//     int tilesK = (K + TILE_SIZE - 1) / TILE_SIZE;
//     int totalTiles = tilesM * tilesK;
    
//     int warpsPerBlock = 4; // 128 threads = 4 warps
//     int threadsPerBlock = warpsPerBlock * WARP_SIZE;
//     int blocksNeeded = (totalTiles + warpsPerBlock - 1) / warpsPerBlock;
    
//     dim3 blockSize(threadsPerBlock);
//     dim3 gridSize(blocksNeeded);
    
//     matrix_mul<<<gridSize, blockSize>>>(A, B, C, M, N, K);
// }

void launch_matmul( __nv_bfloat16* A,  __nv_bfloat16* B, __nv_bfloat16* C,
                        int M, int N, int K)
{
    // tiles along output dims M and K
    int tilesM = (M + TILE_SIZE - 1) / TILE_SIZE;
    int tilesK = (K + TILE_SIZE - 1) / TILE_SIZE;
    int totalTiles = tilesM * tilesK;

    int warpsPerBlock = 4; // 128 threads -> 4 warps
    int threadsPerBlock = warpsPerBlock * WARP_SIZE;
    int blocksNeeded = (totalTiles + warpsPerBlock - 1) / warpsPerBlock;

    // Dynamic shared memory size:
    // per warp: 2 * TILE_SIZE*TILE_SIZE __nv_bfloat16 (for A and B tiles) = 2 * 256 = 512 bf16 per warp
    // plus per-warp float buffer for storing cFrag: TILE_SIZE*TILE_SIZE floats = 256 floats per warp
    // To simplify, we allocate: (warpsPerBlock * 2 * perWarpElems) bf16 + (warpsPerBlock * perWarpElems) float
    int perWarpElems = TILE_SIZE * TILE_SIZE; // 256
    size_t bytes_bf16 = sizeof(__nv_bfloat16) * (warpsPerBlock * 2 * perWarpElems);
    size_t bytes_float = sizeof(float) * (warpsPerBlock * perWarpElems);
    size_t dynamicSharedBytes = bytes_bf16 + bytes_float;

    dim3 blockSize(threadsPerBlock);
    dim3 gridSize(blocksNeeded);
    // kernel expects shm sized exactly to dynamicSharedBytes
    matrix_mul<<<gridSize, blockSize, dynamicSharedBytes>>>(A, B, C, M, N, K);
}

 void launch_elem(__nv_bfloat16* a, __nv_bfloat16* b, __nv_bfloat16* out, int n) {
    int threads = 256, blocks = (n + threads - 1)/threads;
    element_mul<<<blocks, threads>>>(a, b, out, n);
}
 void launch_act(__nv_bfloat16* x, size_t n) {
    int threads = 256, blocks = (n + threads - 1)/threads;
    activation<<<blocks, threads>>>(x, n);
}
 void launch_resadd(__nv_bfloat16* x, __nv_bfloat16* y, size_t n) {
    int threads = 256, blocks = (n + threads - 1)/threads;
    residual_add<<<blocks, threads>>>(x, y, n);
}

void launch_attn(__nv_bfloat16* Q,
                               __nv_bfloat16* out, size_t mq, size_t mkv, size_t head_dim,
                               size_t hidden, size_t hidden_kv, int causal, size_t q_abs_base, int layer_id, page_table* kv_cache_seq1, int page_size) {
    int blocks = hidden / head_dim; // nheads
    int threads = head_dim;
    size_t smem = mkv * sizeof(float);
    selfattention<<<blocks, threads, smem>>>(Q, out, mq, mkv, head_dim, hidden, hidden_kv, causal, q_abs_base, layer_id, kv_cache_seq1, page_size);
            cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "Kernel embedding decode: %s\n", cudaGetErrorString(e)); } cudaDeviceSynchronize();

}

 void proj(const tensor& t, std::ifstream& f,
                        __nv_bfloat16* w_h, __nv_bfloat16* w_d, size_t w_elems,
                        __nv_bfloat16* x, __nv_bfloat16* y,
                        int m, int n, int k, __nv_bfloat16* g_gpu_weights_buffer) {
    load_weight(t, f, w_h, w_d, w_elems, g_gpu_weights_buffer);
    launch_matmul(x, w_d, y, m, n, k);
}

void launch_qknorm(__nv_bfloat16* X, __nv_bfloat16* w, int head_dim, int seqlen, int hidden, int nheads){
    int blocks = nheads, threads = head_dim; qkNorm<<<blocks, threads>>>(X, w, head_dim, seqlen, hidden);
}
 void launch_rope_single(float* cos_d, float* sin_d, __nv_bfloat16* x,
                                      size_t pos, size_t head_dim, int hidden_dim, int nheads){
    RoPE<<<1,1>>>(&cos_d[size_t(pos)*(head_dim/2)], &sin_d[size_t(pos)*(head_dim/2)],
                  x, /*seqlen=*/1, head_dim, hidden_dim, nheads);
}

static inline void copy_last_vocab_vec(__nv_bfloat16* seq, __nv_bfloat16* dst, int hidden, int seqlen) {
    cudaMemcpy(dst, &seq[(seqlen - 1) * hidden], hidden * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
}

static inline void copy_first_token(__nv_bfloat16* seq, __nv_bfloat16* dst, int hidden) {
    cudaMemcpy(dst, seq, hidden * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
}

static inline int sample_topk_bf16(__nv_bfloat16* logits_d, int vocab, float temperature, int topk,
                                   unsigned long long seed, int step) {
    int* d_token = nullptr; int h_token = -1;
    cudaMalloc(&d_token, sizeof(int));
    topk_temperature_softmax_sampling_kernel_bf16<<<1, 256>>>(logits_d, d_token, temperature, topk, vocab, seed, step);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_token, d_token, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_token);
    return h_token;
}