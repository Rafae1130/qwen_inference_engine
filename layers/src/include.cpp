#include <math.h>


//the shape of this cos_value and sin_value will be seq_len x head_dim/2
void precompute_cos_sin(float *cos_values, float *sin_values, int seq_len, int head_dim){

    float base = 1000000;
    int half_head_dim = head_dim/2;
    for(int i = 0; i < half_head_dim; i++ ){
        float exponent = 2*((float) i / (float)head_dim);
        float theta = pow(base, -exponent);

        for(int pos = 0; pos < seq_len; pos++){
            cos_values[pos * half_head_dim + i] = cosf(pos*theta);
            sin_values[pos * half_head_dim + i] = sinf(pos*theta);
        }
    }
}