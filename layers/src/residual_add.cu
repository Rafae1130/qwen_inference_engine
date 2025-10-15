#include <cuda.h>
#include <cuda_bf16.h>




__global__ void residual_add(__nv_bfloat16* __restrict__ mat_a, __nv_bfloat16* __restrict__ mat_b, size_t num_of_elements){

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    //number of threads equal to number of elements 

    if(idx<num_of_elements){
        float temp = 0;
        temp = __bfloat162float(mat_a[idx]) + __bfloat162float(mat_b[idx]);
        mat_a[idx] = __float2bfloat16(temp);
    }
}


