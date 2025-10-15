#include <cuda.h>
#include <cuda_bf16.h>
#include <math.h>

// __global__ void qkNorm(__nv_bfloat16 *qk, __nv_bfloat16 *norm_weigths ,int head_dim, int sequence_len, int hidden_dim){

//     int idx = threadIdx.x + blockIdx.x * blockDim.x; // threads will be equal to head_dim*num_of_heads

//     //num_of_blocks  --> num of heads. i.e. for q it would be 40 and for k it would be 8
//     //num_of_threads --> head dimension

//     //each block will work on one head. and threads in each block will work on each head_dim.

//     float e = 1e-06;
//     __shared__ float buffer[128];

//     //each iteration of the loop will work on one token head of one head. i.e. one row of the block(128 elements)
//     for (int i = 0; i < sequence_len; i++){
//         float val = 0;

//         //load 128 elements, complete head row. all 40 block will load one complete row in parallel
//         val = __bfloat162float(qk[blockIdx.x * head_dim + threadIdx.x + i*hidden_dim]);
//         buffer[threadIdx.x] = val*val; 
//         __syncthreads();
        
//         //vector reduction to sum all elements of row.
//         for (int stride = head_dim/2; stride > 0; stride >>= 1) {
//             if (threadIdx.x < stride) {
//                 buffer[threadIdx.x] += buffer[threadIdx.x + stride];
//             }
//             __syncthreads();
//         }

//         //calculate rms
//         float rms = sqrtf((buffer[0] / head_dim) + e);

//         //calcualte normalization
//         qk[blockIdx.x * head_dim + threadIdx.x + i*hidden_dim] = __float2bfloat16((val / rms)*__bfloat162float(norm_weigths[threadIdx.x]));
//     }
    
// }

__global__ void qkNorm(__nv_bfloat16 *qk, __nv_bfloat16 *norm_weights, 
                       int head_dim, int sequence_len, int hidden_dim){

    const float eps = 1e-04f;
    __shared__ float buffer[128];

    // Each block handles one head
    // Each thread handles one position in head_dim
    for (int tok = 0; tok < sequence_len; tok++){
        
        // Load value for this token and head position
        int idx = tok * hidden_dim + blockIdx.x * head_dim + threadIdx.x;
        float val = __bfloat162float(qk[idx]);
        
        // Square and store in shared memory
        buffer[threadIdx.x] = val * val; 
        __syncthreads();
        
        // Parallel reduction to sum squares
        for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                buffer[threadIdx.x] += buffer[threadIdx.x + stride];
            }
            __syncthreads();
        }

        // Calculate RMS (all threads need this value)
        __shared__ float shared_rms;
        if (threadIdx.x == 0) {
            shared_rms = sqrtf((buffer[0] / head_dim) + eps);
        }
        __syncthreads();

        // Normalize and scale (all threads participate)
        float weight = __bfloat162float(norm_weights[threadIdx.x]);
        qk[idx] = __float2bfloat16((val / shared_rms) * weight);
    }
}