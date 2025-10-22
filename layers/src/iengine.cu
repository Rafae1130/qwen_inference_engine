#include <cuda_bf16.h>
#include <cuda.h>

#include <cuda_runtime.h>
#include <cstdio>

#include "iengine.cuh"

#include <iostream>
#include <fstream>
#include "utils.hh"

#include <chrono>


#include <cstring>

#include <cstdlib>


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

#define CUDA_CHECK(expr) do {                                \
    cudaError_t _err = (expr);                               \
    if (_err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s:%d: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(_err));\
        std::abort();                                        \
    }                                                        \
} while (0)

// Create a singly linked list with `pages_required` nodes (unallocated device ptrs)
page_table* create_page_list(int pages_required) {
    page_table* head = nullptr;
    page_table** cur = &head;
    for (int i = 0; i < pages_required; ++i) {
        *cur = (page_table*)std::malloc(sizeof(page_table));
        if (!*cur) { perror("malloc"); std::abort(); }
        (*cur)->k_page_ptr = nullptr;
        (*cur)->v_page_ptr = nullptr;
        (*cur)->page_allocated = 0;
        (*cur)->ptr_to_next_page = nullptr;
        cur = &((*cur)->ptr_to_next_page);
    }
    return head;
}

// Allocate K/V device buffers for a single node (in elements, not bytes)
void allocate_page_buffers(page_table* node, size_t elems_per_page) {
    if (!node) return;
    size_t bytes = elems_per_page * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc((void**)&node->k_page_ptr, bytes));
    CUDA_CHECK(cudaMalloc((void**)&node->v_page_ptr, bytes));
    node->page_allocated = 1;
}

// Free an entire list (device + host)
void free_page_list(page_table* head) {
    while (head) {
        page_table* next = head->ptr_to_next_page;
        if (head->k_page_ptr) CUDA_CHECK(cudaFree(head->k_page_ptr));
        if (head->v_page_ptr) CUDA_CHECK(cudaFree(head->v_page_ptr));
        std::free(head);
        head = next;
    }
}









int main(){

    auto tensors = build_indexed_tensors();

    //opening weights file to read
    std::ifstream weights("/mnt/data/rafaedata/weights.bin", std::ios::binary);
    if (!weights) { std::cout << "Failed to open weights.bin\n"; return 1; }


    int n=0;
    size_t free_mem = 0;    
    free_mem = printMem(n);




    size_t kvcache_size = CONTEXT_SIZE * HIDDEN_DIM_KV * NUM_OF_LAYERS * 2 * sizeof(__nv_bfloat16); //for fixed batch of 4 for now


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

        std::cout << "here" << std::endl;
        int page_size = 4;    //page_size should be multiple of 40. i.e. sequence_len_per_page required x 40, so that all layers are in one page.
        // page_table *kv_cache_seq1 = (page_table*)malloc(sizeof(page_table));
        // kv_cache_seq1->k_page_ptr = (__nv_bfloat16*)malloc(page_size * new_seq_1->buffer->number_of_layers*new_seq_1->buffer->number_of_layers*sizeof(__nv_bfloat16)); //4096 x 40 x 1024 x 2
        // kv_cache_seq1->v_page_ptr = (__nv_bfloat16*)malloc(page_size * new_seq_1->buffer->number_of_layers*new_seq_1->buffer->number_of_layers*sizeof(__nv_bfloat16)); //4096 x 40 x 1024 x 2

        int pages_required = ((seq_len_1 + page_size -1) / page_size) + 1; 



        page_table* kv_cache_seq1 = create_page_list(pages_required);

        std::cout << "here 2" << std::endl;
        int elements_per_page =  page_size * new_seq_1->buffer->number_of_layers * (size_t)new_seq_1->buffer->hidden_dim_kv;;
        // Eager allocate device buffers for all pages (or do it lazily as pages fill)
        for (page_table* p = kv_cache_seq1; p; p = p->ptr_to_next_page) {
            allocate_page_buffers(p, elements_per_page);
        }

        std::cout << "here 3" << std::endl;
        // cudaMalloc((void **)&kv_cache_seq1->k_page_ptr, page_size * new_seq_1->buffer->number_of_layers*new_seq_1->buffer->number_of_layers*sizeof(__nv_bfloat16));
        // cudaMalloc((void **)&kv_cache_seq1->v_page_ptr, page_size * new_seq_1->buffer->number_of_layers*new_seq_1->buffer->number_of_layers*sizeof(__nv_bfloat16));
        
        // kv_cache_seq1->page_allocated++;

        // new_seq_1->buffer->k_cache = kv_cache_seq1->k_page_ptr;
        // new_seq_1->buffer->v_cache = kv_cache_seq1->v_page_ptr;








        // // Example input 2
        // int h_token_ids_2[] = {151643,785,50802,1525,3818,315,65085,73773,323,2310,20475,61056,
        //                 12236,2411,825,835,315,432,264,57819,22361,11,2238,3460,369,29519,3037};
        // int seq_len_2 = sizeof(h_token_ids_2) / sizeof(h_token_ids_2[0]);
        // batch_metadata* new_seq_2 = create_new_sequence(1, h_token_ids_2, seq_len_2, tensors, weights);


        // __nv_bfloat16* cpu_kcache = (__nv_bfloat16*)malloc(kvcache_size/2);
        // __nv_bfloat16* cpu_vcache = (__nv_bfloat16*)malloc(kvcache_size/2);

        // std::cout << "k cache size: "<< kvcache_size << std::endl;

        cudaStream_t io_stream;
        cudaStreamCreate(&io_stream);

        while(1){
            // cudaEvent_t start, stop;
            // cudaEventCreate(&start);
            // cudaEventCreate(&stop);
            // cudaEventRecord(start);

            // if(new_seq_1->state == decode){
            //     // copy kv cache from disk to vram. 
            //     // cudaMemcpyAsync(new_seq_1->buffer->k_cache, cpu_kcache, kvcache_size/2, cudaMemcpyHostToDevice, io_stream);
            //     // cudaMemcpyAsync(new_seq_1->buffer->v_cache, cpu_vcache, kvcache_size/2, cudaMemcpyHostToDevice, io_stream);
            //     cudaEventCreate(&start);
            //     cudaEventCreate(&stop);
            //     cudaEventRecord(start);
            //     cudaMemcpy(new_seq_1->buffer->k_cache, cpu_kcache, kvcache_size/2, cudaMemcpyHostToDevice);
            //     cudaMemcpy(new_seq_1->buffer->v_cache, cpu_vcache, kvcache_size/2, cudaMemcpyHostToDevice);
            //     cudaDeviceSynchronize();

            //     cudaEventRecord(stop);
            //     float ms = 0;
            //     cudaEventSynchronize(stop);
            //     cudaEventElapsedTime(&ms, start, stop);

            //     std::cout<< "Time in one copy: "<< ms << " ms" << std::endl;
            //     cudaEventDestroy(start);
            //     cudaEventDestroy(stop);

            // }

            // cudaEventCreate(&start);
            // cudaEventCreate(&stop);
            // cudaEventRecord(start);

            std::cout << "going in llm" << std::endl;
            int out_token_1 = llm(new_seq_1, tensors, weights, kv_cache_seq1, page_size);
            std::cout << "out-token-1: " << out_token_1 << std::endl;
            new_seq_1->step = new_seq_1->step + 1;
            new_seq_1->generated_token = out_token_1;
            new_seq_1->state = decode;
            getchar();

            // cudaMemcpyAsync(cpu_kcache, new_seq_1->buffer->k_cache, kvcache_size/2, cudaMemcpyDeviceToHost, io_stream);
            // cudaMemcpyAsync(cpu_vcache, new_seq_1->buffer->v_cache, kvcache_size/2, cudaMemcpyDeviceToHost, io_stream);
            
            // cudaMemcpy(cpu_kcache, new_seq_1->buffer->k_cache, kvcache_size/2, cudaMemcpyDeviceToHost);
            // cudaMemcpy(cpu_vcache, new_seq_1->buffer->v_cache, kvcache_size/2, cudaMemcpyDeviceToHost);
            // cudaDeviceSynchronize();
            

            // cudaEventRecord(stop);
            // float ms = 0;
            // cudaEventSynchronize(stop);
            // cudaEventElapsedTime(&ms, start, stop);

            // std::cout<< "Time in one loop: "<< ms << " ms" << std::endl;
            // cudaEventDestroy(start);
            // cudaEventDestroy(stop);
            
            
            // cudaDeviceSynchronize();
            //copy the kv cache to disk




            // int out_token_2 = llm(new_seq_2, tensors, weights);
            // std::cout << "out-token-2: " << out_token_2 << std::endl;
            // new_seq_2->step = new_seq_2->step + 1;
            // new_seq_2->generated_token = out_token_2;
            // new_seq_2->state = decode;



        }

        // out_token_1 = llm(new_seq_1, tensors, weights);
        // std::cout << "out-token: (prefill)" << out_token_1 << std::endl;




        if (weights.is_open()) weights.close();
        free_page_list(kv_cache_seq1);
        destroy_model_buffers(*new_seq_1->buffer);
        // destroy_model_buffers(*new_seq_2->buffer);

        free(new_seq_1);
        // free(new_seq_2);
        // free(cpu_kcache);
        // free(cpu_vcache);

    }



    return 0;
}