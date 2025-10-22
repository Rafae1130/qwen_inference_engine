#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include "iengine.cuh"
//the K_matrix and the V_matrix are the KV cache.
//there needs to be two seq_len, one for Q and another for KV, as Q can be multiple tokens in first 
//pass but afterwards it will be only one token. But KV caches will keep growing.
__global__ void selfattention(__nv_bfloat16*query_vec, __nv_bfloat16*K_matrix, __nv_bfloat16*V_matrix, __nv_bfloat16*output, size_t seq_len_q, size_t seq_len_kv, size_t head_dim, size_t hidden_dim, size_t kv_dim, int causal, size_t q_abs_base, int layer_id, page_table* kv_cache_seq1, int page_size){

    // int idx = threadIdx.x + blockDim.x*blockIdx.x;

    __shared__ float buffer[128];
    extern  __shared__ float score_row[];

    //QK
    for(int q_tok = 0; q_tok < seq_len_q; q_tok++){
        // page_table* page_temp = kv_cache_seq1;
        for(int k_tok = 0; k_tok < seq_len_kv; k_tok++){
            //q_tok * hidden_dim: q_tok will select each token. as each token is of dimension hidden_dim, hence to move to next token we multiply by hidden_dim.
            //blockIdx.x*head_dim: each token will have dimension 5120 which is divided into 40 heads, there will be 40 block each handling one head of Q. To move to next head we multiply by head_dim.
            //threadIdx.x: there will be 128 elements in each head, which will be computed in parallel as 128 threads will launched for each block.
            float q_val = __bfloat162float(query_vec[q_tok*hidden_dim + blockIdx.x*head_dim + threadIdx.x]);

            //each q_val will remain same for complete k_tok loop. i.e. one q head will be multiplied by all k heads for each block.
            //The K head is multiplied in transpose form but here transpose in memory is not taken just indexing is adjusted.
            //The shape of K is seq_lenx1024 which is then divided into 8 heads hence each head seq_lenx128. Instead of taking its transpose, we just multiply row of q head with row of k head. 
            //Dot product of first row of q with all rows( i.e equal to seq_len ) of k will give the first row of QVt matrix with dimension seq_len.
            //k_tok * kv_dim: k_tok selects the row i.e. token. And since dimension of K is 1024, we take a jump by kv_dim for next token. 
            //(blockIdx.x/5)*head_dim: within the row selected by k_tok * kv_dim, (blockIdx.x/5)*head_dim selects which head of K the currect block works on. Since there will be 5 k blocks per head block, we divide by 5. And multiply by head_dim to move to next head.
            //threadIdx.x: there will be 128 elements in each head, which will be computed in parallel as 128 threads will launched for each block.
            int head_id  = blockIdx.x / 5;      // same as before
            int token_id = k_tok;
            int num_layers = 40;
            // page_table *page_temp;
            // page_temp = kv_cache_seq1;
            // if(k_tok > page_size){
            //     token_id = 0;
            //     page_temp = page_temp->ptr_to_next_page;
            // }
            // float k_val = __bfloat162float(page_temp->k_cache[((token_id * num_layers + layer_id) * kv_dim) + head_id * head_dim + threadIdx.x]);

            // float k_val = __bfloat162float(K_matrix[((token_id * num_layers + layer_id) * kv_dim) + head_id * head_dim + threadIdx.x]);

            // walk to the right page
            page_table* page_temp = kv_cache_seq1;  // Reset for each token

            int page_idx   = k_tok / page_size;     // which page
            token_id   = k_tok % page_size;     // row inside that page
            for (int p = 0; p < page_idx && page_temp; ++p) {
                page_temp = page_temp->ptr_to_next_page;
            }
            if (!page_temp) return; // or continue / guard appropriately

            float k_val = __bfloat162float(
                page_temp->k_page_ptr[ ((token_id * num_layers + layer_id) * kv_dim)
                                    + head_id * head_dim
                                    + threadIdx.x ]
            );



            buffer[threadIdx.x] = q_val * k_val;

            __syncthreads();
            for (int stride = head_dim/2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    buffer[threadIdx.x] += buffer[threadIdx.x + stride];
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                float dot = buffer[0] / sqrtf((float)head_dim);
                //(blockIdx.x*seq_len + q_tok)*seq_len expands to blockIdx.x*seq_len*seq_len + q_tok*seq_len
                //blockIdx.x*seq_len*seq_len: each block will result in a matrix of seq_len x seq_len, hence for each block we jump by that much.
                //q_tok*seq_len: for each token we move to the next row.
                // scores[(blockIdx.x*seq_len + q_tok)*seq_len + k_tok] = dot;
                score_row[k_tok] = dot;
            }
            __syncthreads();
        }

        int q_abs = q_abs_base + q_tok;
        if (causal && threadIdx.x == 0) {
            for (int k_tok = 0; k_tok < seq_len_kv; k_tok++){
                if (k_tok > q_abs) score_row[k_tok] = -1e9f;
            }
        }



        //softmax(this is serial and only one thread is working here)
        if (threadIdx.x == 0) {
            float max_val = -1e9;
            for (int k_tok = 0; k_tok < seq_len_kv; k_tok++) {
                max_val = fmaxf(max_val, score_row[k_tok]);
            }
            float sum_val = 0.f;
            for (int k_tok = 0; k_tok < seq_len_kv; k_tok++) {
                score_row[k_tok] = expf(score_row[k_tok] - max_val);
                sum_val += score_row[k_tok];
            }
            for (int k_tok = 0; k_tok < seq_len_kv; k_tok++) {
                score_row[k_tok] /= sum_val;  // now buffer = softmax weights
            }
        }
        __syncthreads();

        
        //multiply with V
        if (threadIdx.x < head_dim) {
            float out_val = 0.f;
            for (int k_tok = 0; k_tok < seq_len_kv; k_tok++) {
                // float v_val = __bfloat162float(V_matrix[k_tok * kv_dim + (blockIdx.x/5) * head_dim + threadIdx.x ]);
                int head_id  = blockIdx.x / 5;      // same as before
                // int token_id = k_tok;
                int num_layers = 40;

                int page_idx       = k_tok / page_size;
                int offset_in_page = k_tok % page_size;

                // IMPORTANT: reset from head for each k_tok
                page_table* pt = kv_cache_seq1;
                for (int p = 0; p < page_idx && pt; ++p) pt = pt->ptr_to_next_page;
                if (!pt) return;  // guard

                float v_val = __bfloat162float(
                    pt->v_page_ptr[ ((offset_in_page * num_layers + layer_id) * kv_dim)
                                    + head_id * head_dim
                                    + threadIdx.x ]
                );



                // float v_val = __bfloat162float(page_temp->v_page_ptr[ ((token_id * num_layers + layer_id) * kv_dim)+ head_id * head_dim + threadIdx.x ]);                
                out_val += score_row[k_tok] * v_val;
            }

                    //ouput matrix
            output[q_tok * hidden_dim + blockIdx.x * head_dim + threadIdx.x] = __float2bfloat16(out_val);
            __syncthreads();

        }

    }

}







