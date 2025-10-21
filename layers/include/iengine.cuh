#pragma once
#include "utils.hh"
#include <fstream>

#include <cuda_bf16.h>          // __nv_bfloat16
#include <vector>               // std::vector
#include <string>               // std::string
#include <unordered_map>        // std::unordered_map
#include <fstream>              // std::ifstream

#include "utils.hh"             // make sure this defines/declares `tensor` (see note below)

// ---- Forward declarations to break include cycles ----
struct ModelBuffers;            // you only store a pointer in batch_metadata
struct tensor;     



#define CONTEXT_SIZE 32786UL
#define HIDDEN_DIM_KV 1024UL
#define NUM_OF_LAYERS 40UL

typedef enum {prefill, decode} State;



typedef struct {
    int sequence_id;
    __nv_bfloat16 *k_ptr;
    __nv_bfloat16 *v_ptr; 
    State state;
    int sequence_len;
    int generated_token;
    int step;
    ModelBuffers *buffer;

}batch_metadata;

struct page_table_struct;


typedef struct page_table_struct{

    __nv_bfloat16* k_page_ptr;
    __nv_bfloat16* v_page_ptr;
    int page_allocated = 0;
    struct page_table_struct* ptr_to_next_page;
}page_table;


int llm(batch_metadata *new_seq, std::unordered_map<std::string, std::vector<tensor>> tensors, std::ifstream &weights, page_table* kv_cache_seq1, int page_size);

void free_page_list(page_table* head);
void allocate_page_buffers(page_table* node, size_t elems_per_page);
page_table* create_page_list(int pages_required);

