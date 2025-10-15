#include <cuda.h>
#include <math.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float sigmoid(float x){
    return 1/(1 + expf(-x));
}


__global__ void activation( __nv_bfloat16 *matrix, size_t size){


    //number of threads launched will be equal to the number of elements in the matrix.
    int idx = threadIdx.x + blockDim.x * blockIdx.x;


    if (idx < size){
        float x = __bfloat162float(matrix[idx]);   
        float y = x * sigmoid(x);             
        matrix[idx] = __float2bfloat16(y);         
    }

}



