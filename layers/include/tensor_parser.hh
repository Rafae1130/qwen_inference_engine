// #pragma once

// #include"../lib/json.hpp"
// #include<string>
// #include <vector>
// #include<fstream>


// using json = nlohmann::json;

// struct tensor {
//     std::string tensor_name;
//     std::vector<size_t> shape;
//     std::vector<size_t> data_offsets;
// };



// void from_json(const json& j, tensor& p);
// std::ostream& operator<<(std::ostream& os, const tensor& t);
// std::vector<tensor> parsed_tensors();

#pragma once

#include "json.hh"
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <ostream>
#include <stddef.h>

using json = nlohmann::json;

struct tensor {
    std::string tensor_name;           // full name like "model.layers.0.self_attn.k_proj.weight"
    std::vector<size_t> shape;
    std::vector<size_t> data_offsets;
    int layer_index = -1;              // field for layer index
    std::string short_name;            // e.g. "self_attn.k_proj.weight"
};

void from_json(const json& j, tensor& p);
std::ostream& operator<<(std::ostream& os, const tensor& t);

// Returns flat list of all tensors
std::vector<tensor> parsed_tensors();

// returns dictionary for fast lookup: map[short_name][layer_index]
std::unordered_map<std::string, std::vector<tensor>> build_indexed_tensors();

void precompute_cos_sin(float *cos_values, float *sin_values, int seq_len, int head_dim);
