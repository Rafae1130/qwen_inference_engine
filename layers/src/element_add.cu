#include<cuda.h>
#include<cuda_bf16.h>

__global__ void element_mul(__nv_bfloat16 *a, __nv_bfloat16 *b, __nv_bfloat16 *c, int size){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float temp = 0;
    if(idx < size){
        temp =  __bfloat162float(a[idx]) * __bfloat162float(b[idx]);
        c[idx] = __float2bfloat16(temp);
    }
}