#pragma once
#include "utils.hh"
#include <fstream>
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


int llm(batch_metadata *new_seq, std::unordered_map<std::string, std::vector<tensor>> tensors, std::ifstream &weights);