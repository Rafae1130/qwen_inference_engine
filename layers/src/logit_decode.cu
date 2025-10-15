#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <float.h>
#include <stdio.h>


#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

struct Pair { float val; int idx; };

__device__ __forceinline__ Pair better(Pair a, Pair b) {
    return (a.val > b.val) ? a : b;
}

__device__ Pair blockArgMax(Pair local) {
    __shared__ Pair s[BLOCK_SIZE];
    int tid = threadIdx.x;
    s[tid] = local;
    __syncthreads();

    // More robust reduction that handles non-power-of-two sizes
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim.x) {
            s[tid] = better(s[tid], s[tid + stride]);
        }
        __syncthreads();
    }
    return s[0];
}

// __global__ void topk_temperature_softmax_sampling_kernel_bf16(
//     const __nv_bfloat16* __restrict__ logits,
//     int* __restrict__ output_token,
//     float temperature,
//     int k,
//     size_t vocab_size,
//     size_t seed,
//     size_t subseq
// ) {
//     if (blockIdx.x != 0 || gridDim.x != 1) return;
    
//     int tid = threadIdx.x;
    
//     // Input validation
//     if (k <= 0) { 
//         if (tid == 0) *output_token = -1; 
//         return; 
//     }
//     k = min(k, BLOCK_SIZE); // Clamp k to block size
//     if (!(temperature > 0.0f)) temperature = 1.0f;

//     // Initialize RNG for all threads that might need it
//     __shared__ curandState rng_state;
//     if (tid == 0) {
//         curand_init(seed, subseq, 0, &rng_state);
//     }
//     __syncthreads();

//     // Shared memory for top-k results and masking
//     __shared__ int chosen[BLOCK_SIZE];
//     __shared__ float topk_vals[BLOCK_SIZE];
//     __shared__ int topk_idxs[BLOCK_SIZE];
    
//     // Initialize chosen indices
//     if (tid < k) chosen[tid] = -1;
//     __syncthreads();

//     // Find top-k elements
//     for (int sel = 0; sel < k; ++sel) {
//         Pair local = { -INFINITY, -1 };
        
//         // Each thread searches its segment for maximum not in chosen
//         for (int idx = tid; idx < vocab_size; idx += blockDim.x) {
//             // Check if this index is already chosen
//             bool skip = false;
//             for (int t = 0; t < sel; ++t) {
//                 if (idx == chosen[t]) {
//                     skip = true;
//                     break;
//                 }
//             }
//             if (skip) continue;
            
//             float v = __bfloat162float(logits[idx]) / temperature;
//             if (v > local.val) {
//                 local.val = v;
//                 local.idx = idx;
//             }
//         }

//         // Find global maximum
//         Pair global_max = blockArgMax(local);
        
//         if (tid == 0) {
//             // Early exit if no valid candidate found
//             if (global_max.idx == -1) {
//                 k = sel; // Adjust k to number actually found
//             } else {
//                 chosen[sel] = global_max.idx;
//                 topk_vals[sel] = global_max.val;
//                 topk_idxs[sel] = global_max.idx;
//             }
//         }
//         __syncthreads();
        
//         if (sel >= k) break; // Early exit
//     }

//     // Softmax and sampling (only thread 0)
//     if (tid == 0) {
//         if (k == 0) {
//             *output_token = -1;
//             return;
//         }

//         // Compute softmax
//         float max_val = topk_vals[0];
//         for (int i = 1; i < k; ++i) {
//             if (topk_vals[i] > max_val) max_val = topk_vals[i];
//         }

//         float sum = 0.0f;
//         for (int i = 0; i < k; ++i) {
//             topk_vals[i] = expf(topk_vals[i] - max_val);
//             sum += topk_vals[i];
//         }

//         // Sample from distribution
//         float u = curand_uniform(&rng_state);
//         float cum_prob = 0.0f;
//         int picked = topk_idxs[k-1]; // Default to last element
        
//         for (int i = 0; i < k; ++i) {
//             cum_prob += topk_vals[i] / sum;
//             if (u <= cum_prob || i == k-1) {
//                 picked = topk_idxs[i];
//                 break;
//             }
//         }
        
//         *output_token = picked;
//     }
// }

__global__ void topk_temperature_softmax_sampling_kernel_bf16(
    const __nv_bfloat16* __restrict__ logits,
    int* __restrict__ output_token,
    float temperature,
    int k,
    size_t vocab_size,
    size_t seed,
    size_t subseq
) {
    if (blockIdx.x != 0 || gridDim.x != 1) return;
    
    int tid = threadIdx.x;
    
    // Input validation
    if (k <= 0) { 
        if (tid == 0) *output_token = -1; 
        return; 
    }
    k = min(k, (int)vocab_size);
    k = min(k, BLOCK_SIZE);
    if (!(temperature > 0.0f)) temperature = 1.0f;

    __shared__ float topk_vals[BLOCK_SIZE];
    __shared__ int topk_idxs[BLOCK_SIZE];
    
    // Initialize
    if (tid < BLOCK_SIZE) {
        topk_vals[tid] = -INFINITY;
        topk_idxs[tid] = -1;
    }
    __syncthreads();

    // Find top-k using iterative masking
    for (int sel = 0; sel < k; ++sel) {
        Pair local = { -INFINITY, -1 };
        
        // Each thread finds local max (excluding already selected)
        for (int idx = tid; idx < vocab_size; idx += blockDim.x) {
            float v = __bfloat162float(logits[idx]);
            
            // Skip if already selected
            bool already_chosen = false;
            for (int t = 0; t < sel; ++t) {
                if (idx == topk_idxs[t]) {
                    already_chosen = true;
                    break;
                }
            }
            
            if (!already_chosen && v > local.val) {
                local.val = v;
                local.idx = idx;
            }
        }

        // Block-wide reduction
        Pair global_max = blockArgMax(local);
        
        if (tid == 0) {
            if (global_max.idx == -1) {
                // No more valid elements, truncate k
                for (int i = sel; i < k; ++i) {
                    topk_vals[i] = -INFINITY;
                    topk_idxs[i] = -1;
                }
            } else {
                topk_vals[sel] = global_max.val;
                topk_idxs[sel] = global_max.idx;
            }
        }
        __syncthreads();
        
        // Early exit if no valid candidate
        if (topk_idxs[sel] == -1) break;
    }

    // Softmax and sampling (only thread 0)
    if (tid == 0) {
        // Find actual k (number of valid elements)
        int actual_k = 0;
        for (int i = 0; i < k; ++i) {
            if (topk_idxs[i] != -1) actual_k++;
            else break;
        }
        
        if (actual_k == 0) {
            *output_token = -1;
            return;
        }

        // Apply temperature and find max for numerical stability
        float max_val = topk_vals[0] / temperature;
        for (int i = 1; i < actual_k; ++i) {
            float v = topk_vals[i] / temperature;
            if (v > max_val) max_val = v;
            topk_vals[i] = v;
        }
        topk_vals[0] = topk_vals[0] / temperature;

        // Compute softmax
        float sum = 0.0f;
        for (int i = 0; i < actual_k; ++i) {
            topk_vals[i] = expf(topk_vals[i] - max_val);
            sum += topk_vals[i];
        }

        // Initialize RNG
        curandState rng_state;
        curand_init(seed, subseq, 0, &rng_state);
        
        // Sample
        float u = curand_uniform(&rng_state) * sum;
        float cum_prob = 0.0f;
        int picked = topk_idxs[actual_k-1];
        
        for (int i = 0; i < actual_k; ++i) {
            cum_prob += topk_vals[i];
            if (u <= cum_prob) {
                picked = topk_idxs[i];
                break;
            }
        }
        
        *output_token = picked;
    }
}

__global__ void dump_top_logits(const __nv_bfloat16* logits, int vocab, int k){
    if (threadIdx.x==0 && blockIdx.x==0){
        int idxs[64]; float vals[64];
        for (int i=0;i<k;i++){ idxs[i]=-1; vals[i]=-1e30; }
        for (int i=0;i<vocab;i++){
            float v = __bfloat162float(logits[i]);
            int m=0; for (int j=1;j<k;j++) if (vals[j]<vals[m]) m=j;
            if (v>vals[m]){ vals[m]=v; idxs[m]=i; }
        }
        printf("top logits: ");
        for (int i=0;i<k;i++) printf("(%d:%.2f) ", idxs[i], vals[i]);
        printf("\n");
    }
}


// // More efficient top-k using bitonic sort
// __device__ void swap_if_greater(float& v1, int& i1, float& v2, int& i2) {
//     if (v1 < v2) {
//         float tv = v1; v1 = v2; v2 = tv;
//         int ti = i1; i1 = i2; i2 = ti;
//     }
// }

// __global__ void topk_temperature_sampling_kernel_bf16(
//     const __nv_bfloat16* __restrict__ logits,
//     int* __restrict__ output_token,
//     float temperature,
//     int k,
//     int vocab_size,
//     unsigned long long seed,
//     unsigned long long step  // MUST be different each call!
// ) {
//     if (blockIdx.x != 0) return;
    
//     int tid = threadIdx.x;
    
//     // Validation
//     if (k <= 0 || k > BLOCK_SIZE) {
//         if (tid == 0) *output_token = 0;
//         return;
//     }
//     if (temperature <= 0.0f) temperature = 1.0f;
    
//     __shared__ float svals[BLOCK_SIZE];
//     __shared__ int sidxs[BLOCK_SIZE];
//     __shared__ curandState rng;
    
//     // Each thread finds its local max
//     float local_max = -FLT_MAX;
//     int local_idx = tid;
    
//     for (int i = tid; i < vocab_size; i += blockDim.x) {
//         float val = __bfloat162float(logits[i]) / temperature;
//         if (val > local_max) {
//             local_max = val;
//             local_idx = i;
//         }
//     }
    
//     svals[tid] = local_max;
//     sidxs[tid] = local_idx;
//     __syncthreads();
    
//     // Simple selection sort for top-k (works well for small k)
//     if (tid < k) {
//         for (int i = tid; i < BLOCK_SIZE; ++i) {
//             if (i != tid && svals[i] > svals[tid]) {
//                 swap_if_greater(svals[i], sidxs[i], svals[tid], sidxs[tid]);
//             }
//         }
//     }
//     __syncthreads();
    
//     // Thread 0 does softmax and sampling
//     if (tid == 0) {
//         curand_init(seed, step, 0, &rng);
        
//         // Find max for numerical stability
//         float max_val = svals[0];
//         for (int i = 1; i < k; ++i) {
//             if (svals[i] > max_val) max_val = svals[i];
//         }
        
//         // Compute exp and sum
//         float sum = 0.0f;
//         for (int i = 0; i < k; ++i) {
//             svals[i] = expf(svals[i] - max_val);
//             sum += svals[i];
//         }
        
//         // Sample
//         float u = curand_uniform(&rng) * sum;
//         float cumsum = 0.0f;
//         int selected = sidxs[k-1];
        
//         for (int i = 0; i < k; ++i) {
//             cumsum += svals[i];
//             if (u <= cumsum) {
//                 selected = sidxs[i];
//                 break;
//             }
//         }
        
//         *output_token = selected;
//     }
// }


