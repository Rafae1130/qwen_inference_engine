#include <cuda.h>
#include <cuda_bf16.h>
#include <math.h>

__global__ void rmsNorm(__nv_bfloat16 *embedding_matrix, __nv_bfloat16 *norm_weigths, __nv_bfloat16 *rms_out ,size_t hidden_dim, size_t sequence_len){

    int idx = threadIdx.x + blockIdx.x * blockDim.x; // threads will be equal to sequence length.

    float e = 1e-04;
    if (idx < sequence_len){
        float sum = 0;
        float temp1 =0;
        for(int i = 0; i < hidden_dim; i++){
            temp1 = __bfloat162float(embedding_matrix[idx*hidden_dim+ i]);
            sum += temp1 * temp1;
        }

        float rms = sqrtf((sum/ hidden_dim) + e);


        for(int i = 0; i < hidden_dim; i++){
            temp1 = __bfloat162float(embedding_matrix[idx*hidden_dim+ i]);
            rms_out[idx*hidden_dim+ i] = __float2bfloat16((temp1 /rms)*__bfloat162float(norm_weigths[i]));
        }
    }
}