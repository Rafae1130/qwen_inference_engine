#include <cuda_bf16.h>
#include <cuda.h>

#include <cuda_runtime.h>
#include <cstdio>

#include "iengine.cuh"

#include <iostream>
#include <fstream>
#include "utils.hh"



batch_metadata* create_new_sequence(
    int sequence_id,
    int* h_token_ids,
    int sequence_len,
    std::unordered_map<std::string, std::vector<tensor>> tensors, std::ifstream &weights
) {
    // Allocate model buffers for this sequence
    ModelBuffers* buffer = new ModelBuffers();
    initialize_model_buffers(*buffer, h_token_ids, tensors, weights, sequence_len);

    // Allocate metadata
    batch_metadata* seq = (batch_metadata*) malloc(sizeof(batch_metadata));
    seq->state = prefill;
    seq->k_ptr = buffer->k_cache;
    seq->v_ptr = buffer->v_cache;
    seq->sequence_id = sequence_id;
    seq->buffer = buffer;
    seq->sequence_len = sequence_len;
    seq->step = 0;
    seq->generated_token = 0;

    return seq;
}



static void check(cudaError_t e) { if (e != cudaSuccess) { 
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); std::abort(); } }

size_t printMem(int dev) {
    check(cudaSetDevice(dev));
    size_t free_b=0, total_b=0;
    check(cudaMemGetInfo(&free_b, &total_b));
    // printf("GPU %d: used %.2f MiB / %.2f MiB (free %.2f MiB)\n", dev, (total_b - free_b) / (1024.0*1024.0), total_b / (1024.0*1024.0), free_b / (1024.0*1024.0));

    return free_b;
}



int main(){

    auto tensors = build_indexed_tensors();

    //opening weights file to read
    std::ifstream weights("/mnt/data/rafae/qwen_weights/weights.bin", std::ios::binary);
    if (!weights) { std::cout << "Failed to open weights.bin\n"; return 1; }


    int n=0;
    size_t free_mem = 0;    
    free_mem = printMem(n);




    size_t kvcache_size = 4* CONTEXT_SIZE * HIDDEN_DIM_KV * NUM_OF_LAYERS * 2 * sizeof(__nv_bfloat16); //for fixed batch of 4 for now


    //new prompt arrives
    if (free_mem < kvcache_size){
        // printf("Not enough VRam available\r\n");
    }
    else
    {
        // //1
        // int h_token_ids_1 [] = {151643, 785, 4767, 315, 279, 3639, 4180, 374};
        // int sequence_len = sizeof(h_token_ids_1) / sizeof(h_token_ids_1[0]);
        // ModelBuffers buffer_1;
        // initialize_model_buffers(buffer_1, h_token_ids_1, tensors, weights, sequence_len);

        // batch_metadata *new_seq_1 = (batch_metadata *) malloc(sizeof(batch_metadata));
        // new_seq_1->state = prefill;
        // new_seq_1->k_ptr = buffer_1.k_cache;
        // new_seq_1->v_ptr = buffer_1.v_cache;
        // new_seq_1->sequence_id = 0;
        // new_seq_1->buffer = &buffer_1;
        // new_seq_1->sequence_len = sequence_len;
        // int out_token_1 = 0; 

        // //2
        // int h_token_ids_2 [] = {151643,785, 50802, 1525, 3818, 315, 65085, 73773, 323, 2310, 20475, 61056, 12236, 2411, 825, 835, 315, 432, 264, 57819, 22361, 11, 2238, 3460, 369, 29519, 3037};
        // int sequence_len2 = sizeof(h_token_ids_2) / sizeof(h_token_ids_2[0]);
        // ModelBuffers buffer_2;
        // initialize_model_buffers(buffer_2, h_token_ids_2, tensors, weights, sequence_len2);

        // batch_metadata *new_seq_2 = (batch_metadata *) malloc(sizeof(batch_metadata));
        // new_seq_2->state = prefill;
        // new_seq_2->k_ptr = buffer_2.k_cache;
        // new_seq_2->v_ptr = buffer_2.v_cache;
        // new_seq_2->sequence_id = 0;
        // new_seq_2->buffer = &buffer_2;
        // new_seq_2->sequence_len = sequence_len2;
        // int out_token_2 = 0; 

        // Example input 1
        int h_token_ids_1[] = {151643, 785, 4767, 315, 279, 3639, 4180, 374};
        int seq_len_1 = sizeof(h_token_ids_1) / sizeof(h_token_ids_1[0]);
        batch_metadata* new_seq_1 = create_new_sequence(0, h_token_ids_1, seq_len_1, tensors, weights);

        // Example input 2
        int h_token_ids_2[] = {151643,785,50802,1525,3818,315,65085,73773,323,2310,20475,61056,
                        12236,2411,825,835,315,432,264,57819,22361,11,2238,3460,369,29519,3037};
        int seq_len_2 = sizeof(h_token_ids_2) / sizeof(h_token_ids_2[0]);
        batch_metadata* new_seq_2 = create_new_sequence(1, h_token_ids_2, seq_len_2, tensors, weights);



        while(1){
            int out_token_1 = llm(new_seq_1, tensors, weights);
            std::cout << "out-token-1 " << out_token_1 << std::endl;
            new_seq_1->step = new_seq_1->step + 1;
            new_seq_1->generated_token = out_token_1;
            new_seq_1->state = decode;


            int out_token_2 = llm(new_seq_2, tensors, weights);
            std::cout << "out-token-2: " << out_token_2 << std::endl;
            new_seq_2->step = new_seq_2->step + 1;
            new_seq_2->generated_token = out_token_2;
            new_seq_2->state = decode;



        }

        // out_token_1 = llm(new_seq_1, tensors, weights);
        // std::cout << "out-token: (prefill)" << out_token_1 << std::endl;




        if (weights.is_open()) weights.close();
        
        destroy_model_buffers(*new_seq_1->buffer);
        destroy_model_buffers(*new_seq_2->buffer);
        free(new_seq_1);
        free(new_seq_2);

    }



    return 0;
}