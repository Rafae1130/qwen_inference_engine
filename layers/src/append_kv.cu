#include<cuda.h>
#include <cuda_bf16.h>


__global__ void append_kv(__nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ k_cache, __nv_bfloat16* __restrict__ v_cache, int current_seq_ind ,int sequence_len, int kv_dim){

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    //threads equal to sequence number will be launched. each thread will store all elements i.e. kv_dim of one token/sequence.
    if(idx < sequence_len){
        for(int i = 0; i < kv_dim; i++){
            k_cache[current_seq_ind*kv_dim + idx*kv_dim + i] = k[idx*kv_dim + i];
            v_cache[current_seq_ind*kv_dim + idx*kv_dim + i] = v[idx*kv_dim + i];
        }
    }
}

