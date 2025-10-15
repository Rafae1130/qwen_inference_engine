#include <cuda.h>
#include <cuda_bf16.h>

//optimization: launch a 2D block of threads to assign each row of each head to a single thread. 
//Currently one thread is computing one row of all heads.
__global__ void RoPE(float *cos_values, float *sin_values, __nv_bfloat16 *embedded_matrix, int seq_len, int head_dim, int embedding_dim, int num_heads){

    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int half_dim = head_dim / 2;
    if (idx < seq_len){
        for(int head_ind = 0; head_ind < num_heads; head_ind++){
            for (int i = 0; i < head_dim; i+=2){
                int trig_index = i/2;
                int base = idx*embedding_dim + head_ind * head_dim;
                float head_val1 = __bfloat162float(embedded_matrix[base + i]) * cos_values[idx*half_dim + trig_index] - __bfloat162float(embedded_matrix[base + i + 1]) * sin_values[idx*half_dim + trig_index]; //taking each head as a seperate matrix
                float head_val2 = __bfloat162float(embedded_matrix[base + i + 1]) * cos_values[idx*half_dim + trig_index] + __bfloat162float(embedded_matrix[base + i]) * sin_values[idx*half_dim + trig_index];
                embedded_matrix[base + i] = __float2bfloat16(head_val1);
                embedded_matrix[base + i + 1] = __float2bfloat16(head_val2);
            }
        }
    }
}




