#include<cuda.h>

#include "layers_include.cuh"
#include "tensor_parser.hh"
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cuda_bf16.h>
#include <iomanip>
#include <utils.hh>
#include <helpers.cuh>
#include "iengine.cuh"




// #define PRINT_TIME

inline void startCudaTimer(cudaEvent_t &start, cudaEvent_t &stop) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
}

inline float stopCudaTimer(cudaEvent_t &start, cudaEvent_t &stop, const char* label, int layer) {
    float ms = 0.0f;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (label) {
        std::cout <<"Layer: "<< layer <<label << ": " << ms << " ms" << std::endl;
    } else {
        std::cout << "Elapsed time: " << ms << " ms" << std::endl;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}





int llm(batch_metadata *new_seq , std::unordered_map<std::string, std::vector<tensor>> tensors, std::ifstream &weights, page_table *kv_cache_seq1, int page_size){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    ModelBuffers *buffer;
    buffer = new_seq->buffer;
    if (new_seq->state == prefill){


        for(size_t i = 0; i < buffer->number_of_layers; i++){

            std::cout << "loop iteration: " << i << std::endl;

            //normalization block
            load_weight(tensors["input_layernorm.weight"][i], weights, buffer->norm_weights_h, buffer->norm_weights_d, buffer->hidden_dim);
            launch_rms(buffer->embeddings_out, buffer->norm_weights_d, buffer->rms_out, buffer->hidden_dim, buffer->sequence_len);

            //projections - Q
            proj(tensors["self_attn.q_proj.weight"][i], weights,
                buffer->q_proj_weights_h, buffer->q_proj_weights_d, buffer->q_proj_size,
                buffer->rms_out, buffer->Q, buffer->sequence_len, 5120, 5120);    // Load q_proj weights (H->D) and compute Q = rms_out · W_q  [m=seqlen, n=hidden=5120, k=hidden=5120]
            //projections - K
            proj(tensors["self_attn.k_proj.weight"][i], weights,
                buffer->kv_proj_weights_h, buffer->kv_proj_weights_d, buffer->kv_proj_size,
                buffer->rms_out, buffer->K, buffer->sequence_len, 5120, 1024);    // Load k_proj weights (H->D) and compute K = rms_out · W_k  [m=seqlen, n=hidden=5120, k=kv=1024]
            //projections - V
            proj(tensors["self_attn.v_proj.weight"][i], weights,
                buffer->kv_proj_weights_h, buffer->kv_proj_weights_d, buffer->kv_proj_size,
                buffer->rms_out, buffer->V, buffer->sequence_len, 5120, 1024);    // Load v_proj weights (H->D) and compute V = rms_out · W_v  [m=seqlen, n=hidden=5120, k=kv=1024]

            // q_norm
            load_weight(tensors["self_attn.q_norm.weight"][i], weights, buffer->qk_norm_weights_h, buffer->qk_norm_weights_d, buffer->head_dim);
            launch_qknorm(buffer->Q, buffer->qk_norm_weights_d, buffer->head_dim, buffer->sequence_len, buffer->hidden_dim, buffer->num_of_qheads);

            // k_norm
            load_weight(tensors["self_attn.k_norm.weight"][i], weights, buffer->qk_norm_weights_h, buffer->qk_norm_weights_d, buffer->head_dim);
            launch_qknorm(buffer->K, buffer->qk_norm_weights_d, buffer->head_dim, buffer->sequence_len, buffer->hidden_dim_kv, buffer->num_of_kvheads);

            // RoPE
            launch_rope(buffer->cos_values_d, buffer->sin_values_d, buffer->Q, buffer->sequence_len, buffer->head_dim, buffer->hidden_dim, buffer->num_of_qheads); // Apply RoPE to Q for all query heads
            launch_rope(buffer->cos_values_d, buffer->sin_values_d, buffer->K, buffer->sequence_len, buffer->head_dim, buffer->hidden_dim_kv, buffer->num_of_kvheads); // Apply RoPE to K for all KV heads

            kv_copy_layer_to_cache_prefill(buffer, /*i=*/(int)i, kv_cache_seq1, /*page_size=*/page_size);

            // #ifdef PRINT_TIME
            // stopCudaTimer(start, stop, "KV paged copy", i);
            // #endif


            //self attention
            #ifdef PRINT_TIME
            startCudaTimer(start, stop);
            #endif
            launch_attn(buffer->Q, buffer->k_cache, buffer->v_cache, buffer->atten_out, buffer->sequence_len, buffer->sequence_len, buffer->head_dim, buffer->hidden_dim, buffer->hidden_dim_kv, /*causal=*/1, 0, i, kv_cache_seq1, page_size); // self-attention (prefill uses full causal window)
            #ifdef PRINT_TIME
            stopCudaTimer(start, stop, "Time in attention", i);
            #endif
            
            //outut projection self attention
            #ifdef PRINT_TIME
            startCudaTimer(start, stop);
            #endif
            proj(tensors["self_attn.o_proj.weight"][i], weights, buffer->o_proj_weights_h, buffer->o_proj_weights_d, buffer->o_proj_size, buffer->atten_out, buffer->out_proj, buffer->sequence_len, /*n=*/5120, /*k=*/5120); // output projection (O proj)
            #ifdef PRINT_TIME

            stopCudaTimer(start, stop, "Time in o_projction", i);
            #endif

            //residual add
            #ifdef PRINT_TIME
            startCudaTimer(start, stop);
            #endif
            launch_resadd(buffer->embeddings_out, buffer->out_proj, buffer->sequence_len * buffer->hidden_dim); // residual add: x += out_proj
            #ifdef PRINT_TIME

            stopCudaTimer(start, stop, "Time in residual add", i);
            #endif

            //normalization
            #ifdef PRINT_TIME
            startCudaTimer(start, stop);
            #endif
            load_weight(tensors["post_attention_layernorm.weight"][i], weights, buffer->norm_weights_h, buffer->norm_weights_d, buffer->hidden_dim); // load post-attn RMSNorm weights to device
            launch_rms(buffer->embeddings_out, buffer->norm_weights_d, buffer->rms_out, buffer->hidden_dim, buffer->sequence_len); // post-attention RMSNorm
            
            #ifdef PRINT_TIME

            stopCudaTimer(start, stop, "Time in rms", i);
            #endif

            //MLP
            #ifdef PRINT_TIME
            startCudaTimer(start, stop);
            #endif
            proj(tensors["mlp.up_proj.weight"][i],   weights, buffer->mlp_up_proj_weights_h, buffer->mlp_up_proj_weights_d, buffer->mlp_up_proj_size, buffer->rms_out,      buffer->MLP_UP,        buffer->sequence_len, /*n=*/5120,           /*k=*/buffer->up_dim); // MLP up-proj
            #ifdef PRINT_TIME

            stopCudaTimer(start, stop, "Time in MLP up proj", i);
            #endif


            #ifdef PRINT_TIME
            startCudaTimer(start, stop);
            #endif
            proj(tensors["mlp.gate_proj.weight"][i], weights, buffer->mlp_up_proj_weights_h, buffer->mlp_up_proj_weights_d, buffer->mlp_up_proj_size, buffer->rms_out,      buffer->MLP_GATE,      buffer->sequence_len, /*n=*/5120,           /*k=*/buffer->up_dim); // MLP gate-proj
            #ifdef PRINT_TIME

            stopCudaTimer(start, stop, "Time in MLP  gate proj", i);
            #endif

            #ifdef PRINT_TIME
            startCudaTimer(start, stop);
            #endif
            launch_act(buffer->MLP_GATE, buffer->sequence_len * buffer->up_dim); // activation (e.g., SiLU) on gate
            launch_elem(buffer->MLP_UP, buffer->MLP_GATE, buffer->MLP_GATE_OUT, buffer->sequence_len * buffer->up_dim); // elementwise multiply: up * act(gate)
            #ifdef PRINT_TIME

            stopCudaTimer(start, stop, "Time in activation and multiply", i);
            #endif

            #ifdef PRINT_TIME
            startCudaTimer(start, stop);
            #endif
            proj(tensors["mlp.down_proj.weight"][i], weights, buffer->mlp_up_proj_weights_h, buffer->mlp_up_proj_weights_d, buffer->mlp_up_proj_size, buffer->MLP_GATE_OUT, buffer->MLP_DOWN,      buffer->sequence_len, /*n=*/buffer->up_dim,  /*k=*/buffer->hidden_dim); // MLP down-proj
            #ifdef PRINT_TIME

            stopCudaTimer(start, stop, "Time in down projection", i);
            #endif


            #ifdef PRINT_TIME
            startCudaTimer(start, stop);
            #endif
            launch_resadd(buffer->embeddings_out, buffer->MLP_DOWN, buffer->sequence_len * buffer->hidden_dim); // residual add: x += mlp_down
            #ifdef PRINT_TIME

            stopCudaTimer(start, stop, "Time in final residual add", i);
            #endif
        }


        // final RMSNorm before logits
        load_weight(tensors["norm.weight"][0], weights, buffer->norm_weights_h, buffer->norm_weights_d, buffer->hidden_dim);  // load γ
        launch_rms (buffer->embeddings_out, buffer->norm_weights_d, buffer->rms_out, buffer->hidden_dim, buffer->sequence_len); // y = rmsnorm(x)

        // load logits (W_vocab)
        load_weight(tensors["logits"][0], weights, buffer->logits_weights_h, buffer->logits_weights_d, buffer->logtis_shape);          

        // pick last token activation
        copy_last_vocab_vec(buffer->rms_out, buffer->last_x, buffer->hidden_dim, buffer->sequence_len);                                            

        // logits = last_x · W_vocab
        launch_matmul(buffer->last_x, buffer->logits_weights_d, buffer->prefill_output_d, /*m=*/1, /*n=*/buffer->hidden_dim, /*k=*/buffer->vocab_size); 


        //output sampling
        int step = 0;                                                                                                      // decode step counter
        int out_logit = sample_topk_bf16(buffer->prefill_output_d, buffer->vocab_size, /*temperature=*/1.0f, /*topk=*/50,
                                        /*seed=*/1234ULL, /*step=*/step);                                                 // sample next token id
        // std::cout << "out_logit: " << out_logit << std::endl;  


        return out_logit;
    }


    else if (new_seq->state == decode){
        int out_logit = new_seq->generated_token;
        // int seq = buffer->sequence_len;
        int *d_token_ids_decode;
        cudaMalloc((void **)&d_token_ids_decode, sizeof(int));
        
        //decode stage
        while(out_logit != 151645){
            int step = new_seq->step;
            int threads = 1;
            int blocks = 1;
            cudaMemcpy(d_token_ids_decode, &out_logit, sizeof(int), cudaMemcpyHostToDevice);
            cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "cudaMemcpy before embedding decode: %s\n", cudaGetErrorString(e)); return 0; } cudaDeviceSynchronize();

            size_t sequence_len_q = 1;
            buffer->sequence_len = buffer->sequence_len + 1;

            embedding_matrix_func<<<blocks,threads>>>(buffer->embeddings_out, buffer->embeddings_d, d_token_ids_decode, buffer->hidden_dim, sequence_len_q);
            e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "Kernel embedding decode: %s\n", cudaGetErrorString(e)); return 0; } cudaDeviceSynchronize();


            for(size_t  i =0; i < buffer->number_of_layers; i++){
                // std::cout << "loop iteration: " << i << std::endl;
                // rms norm (input LN for decode token)
                load_weight(tensors["input_layernorm.weight"][i], weights, buffer->norm_weights_h, buffer->norm_weights_d, buffer->hidden_dim);
                launch_rms(buffer->embeddings_out, buffer->norm_weights_d, buffer->rms_out, buffer->hidden_dim, sequence_len_q);
                e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "Kernel rmsNorm decode: %s\n", cudaGetErrorString(e)); return 0; } cudaDeviceSynchronize();

                // Q projection
                proj(tensors["self_attn.q_proj.weight"][i], weights, buffer->q_proj_weights_h, buffer->q_proj_weights_d, buffer->q_proj_size,
                    /*x=*/buffer->rms_out, /*y=*/buffer->Q, /*m=*/sequence_len_q, /*n=*/5120, /*k=*/5120);
                e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "Kernel matmul: %s\n", cudaGetErrorString(e)); return 0; } cudaDeviceSynchronize();

                // K projection
                proj(tensors["self_attn.k_proj.weight"][i], weights, buffer->kv_proj_weights_h, buffer->kv_proj_weights_d, buffer->kv_proj_size,
                    /*x=*/buffer->rms_out, /*y=*/buffer->K, /*m=*/sequence_len_q, /*n=*/5120, /*k=*/1024);

                // V projection
                proj(tensors["self_attn.v_proj.weight"][i], weights, buffer->kv_proj_weights_h, buffer->kv_proj_weights_d, buffer->kv_proj_size,
                    /*x=*/buffer->rms_out, /*y=*/buffer->V, /*m=*/sequence_len_q, /*n=*/5120, /*k=*/1024);

                // q_norm
                load_weight(tensors["self_attn.q_norm.weight"][i], weights, buffer->qk_norm_weights_h, buffer->qk_norm_weights_d, buffer->head_dim);
                launch_qknorm(buffer->Q, buffer->qk_norm_weights_d, buffer->head_dim, sequence_len_q, buffer->hidden_dim, buffer->num_of_qheads);

                // k_norm
                load_weight(tensors["self_attn.k_norm.weight"][i], weights, buffer->qk_norm_weights_h, buffer->qk_norm_weights_d, buffer->head_dim);
                launch_qknorm(buffer->K, buffer->qk_norm_weights_d, buffer->head_dim, sequence_len_q, buffer->hidden_dim_kv, buffer->num_of_kvheads);

                // RoPE (single position = buffer->sequence_len - 1)
                launch_rope_single(buffer->cos_values_d, buffer->sin_values_d, buffer->Q, buffer->sequence_len - 1, buffer->head_dim, buffer->hidden_dim,    buffer->num_of_qheads);
                e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "Kernel rope: %s\n", cudaGetErrorString(e)); return 0; } cudaDeviceSynchronize();
                launch_rope_single(buffer->cos_values_d, buffer->sin_values_d, buffer->K, buffer->sequence_len - 1, buffer->head_dim, buffer->hidden_dim_kv, buffer->num_of_kvheads);
                e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "Kernel rope: %s\n", cudaGetErrorString(e)); return 0; } cudaDeviceSynchronize();
            


                // //kv cache
                #ifdef PRINT_TIME
                startCudaTimer(start, stop);
                #endif

                kv_copy_layer_to_cache_decode(buffer, /*i=*/(int)i, kv_cache_seq1, /*page_size=*/page_size);

                #ifdef PRINT_TIME
                float ms = stopCudaTimer(start, stop, "Layer copy (paged)", i);
                #endif

                cudaDeviceSynchronize();
                    
                #ifdef PRINT_TIME
                    startCudaTimer(start, stop);
                #endif
                // self‑attention 
                int q_abs = buffer->sequence_len - 1;
                // launch_attn(buffer->Q, &buffer->k_cache[i*(buffer->context_size*buffer->hidden_dim_kv)], &buffer->v_cache[i*(buffer->context_size*buffer->hidden_dim_kv)], buffer->atten_out, /*mq=*/sequence_len_q, /*mkv=*/buffer->sequence_len, /*head_dim=*/buffer->head_dim, /*hidden=*/buffer->hidden_dim, /*hidden_kv=*/buffer->hidden_dim_kv, /*causal=*/0, q_abs, i); // SA(Q, Kcache, Vcache) → atten_out
                launch_attn(buffer->Q, buffer->k_cache, buffer->v_cache, buffer->atten_out, /*mq=*/sequence_len_q, /*mkv=*/buffer->sequence_len, /*head_dim=*/buffer->head_dim, /*hidden=*/buffer->hidden_dim, /*hidden_kv=*/buffer->hidden_dim_kv, /*causal=*/0, q_abs, i, kv_cache_seq1, page_size); // SA(Q, Kcache, Vcache) → atten_out

                #ifdef PRINT_TIME
                    cudaDeviceSynchronize();
                    ms = stopCudaTimer(start, stop, "Time in attention", i);
                #endif

                // output projection 
                proj(tensors["self_attn.o_proj.weight"][i], weights, buffer->o_proj_weights_h, buffer->o_proj_weights_d, buffer->o_proj_size, /*x=*/buffer->atten_out, /*y=*/buffer->out_proj, /*m=*/sequence_len_q, /*n=*/5120, /*k=*/5120); // O-proj

                // residual add
                launch_resadd(buffer->embeddings_out, buffer->out_proj, /*n=*/sequence_len_q * buffer->hidden_dim); // add SA output

                // post-attention RMSNorm
                load_weight(tensors["post_attention_layernorm.weight"][i], weights, buffer->norm_weights_h, buffer->norm_weights_d, buffer->hidden_dim); // load LN2 weights
                launch_rms(buffer->embeddings_out, buffer->norm_weights_d, buffer->rms_out, buffer->hidden_dim, sequence_len_q); // LN2(x) → rms_out

                // MLP up-proj
                proj(tensors["mlp.up_proj.weight"][i],   weights, buffer->mlp_up_proj_weights_h, buffer->mlp_up_proj_weights_d, buffer->mlp_up_proj_size, /*x=*/buffer->rms_out, /*y=*/buffer->MLP_UP,   /*m=*/sequence_len_q, /*n=*/5120, /*k=*/buffer->up_dim); // up-proj

                // MLP gate-proj
                proj(tensors["mlp.gate_proj.weight"][i], weights, buffer->mlp_up_proj_weights_h, buffer->mlp_up_proj_weights_d, buffer->mlp_up_proj_size, /*x=*/buffer->rms_out, /*y=*/buffer->MLP_GATE, /*m=*/sequence_len_q, /*n=*/5120, /*k=*/buffer->up_dim); // gate-proj

                // activation on gate
                launch_act(buffer->MLP_GATE, /*n=*/sequence_len_q * buffer->up_dim); // act(G) in-place

                // gated product
                launch_elem(buffer->MLP_UP, buffer->MLP_GATE, buffer->MLP_GATE_OUT, /*n=*/sequence_len_q * buffer->up_dim); // U * act(G)

                // MLP down-proj
                proj(tensors["mlp.down_proj.weight"][i], weights, buffer->mlp_up_proj_weights_h, buffer->mlp_up_proj_weights_d, buffer->mlp_up_proj_size, /*x=*/buffer->MLP_GATE_OUT, /*y=*/buffer->MLP_DOWN, /*m=*/sequence_len_q, /*n=*/buffer->up_dim, /*k=*/buffer->hidden_dim); // down-proj

                // residual add
                launch_resadd(buffer->embeddings_out, buffer->MLP_DOWN, /*n=*/sequence_len_q * buffer->hidden_dim); // add MLP output

            }

            // final token RMSNorm
            load_weight(tensors["norm.weight"][0], weights, buffer->norm_weights_h, buffer->norm_weights_d, buffer->hidden_dim);                   // γ
            launch_rms(buffer->embeddings_out, buffer->norm_weights_d, buffer->rms_out, buffer->hidden_dim, /*seqlen=*/sequence_len_q);            // y = rmsnorm(x)

            // load logits matrix and copy first token state
            load_weight(tensors["logits"][0],     weights, buffer->logits_weights_h, buffer->logits_weights_d, buffer->logtis_shape);             // W_vocab
            copy_first_token(buffer->rms_out, buffer->last_x, buffer->hidden_dim);                                                                 // last_x ← rms_out[0]

            // logits 
            launch_matmul(buffer->last_x, buffer->logits_weights_d, buffer->prefill_output_d, /*m=*/1, /*n=*/buffer->hidden_dim, /*k=*/buffer->vocab_size);

            e = cudaGetLastError();
            if (e != cudaSuccess) { fprintf(stderr, "Kernel matrix_mul: decode %s\n", cudaGetErrorString(e)); return 0; }
            cudaDeviceSynchronize();


            //output sampling
            int* d_output_token;
            float temperature = 0.7;
            int topk = 50;
            cudaMalloc(&d_output_token, sizeof(int));
            cudaDeviceSynchronize();

            // std::cout<<"going in logits_decode kernel"<< std::endl;
            
            topk_temperature_softmax_sampling_kernel_bf16<<<1, 256>>>(buffer->prefill_output_d, d_output_token,temperature,topk,buffer->vocab_size,1234 + step, 0 );

            e = cudaGetLastError();
            if (e != cudaSuccess) { fprintf(stderr, "Kernel topk_temperature_softmax_sampling_kernel_bf16: decode %s\n", cudaGetErrorString(e)); return 0; }
            cudaDeviceSynchronize();
            
            int out_logit = 0;
            
            cudaMemcpy(&out_logit, d_output_token, sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            // std::cout << "out_logit: "<< out_logit<< std::endl;
            cudaFree(d_output_token);
            
            cudaFree(d_token_ids_decode);
            return out_logit;   
        }   


    }
    cudaDeviceSynchronize();
 
    // if (weights.is_open()) weights.close();
    // destroy_model_buffers(buffer);
    std::cout <<"Reached end"<<std::endl;



    return 0;
}







