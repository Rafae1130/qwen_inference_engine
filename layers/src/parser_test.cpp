#include "tensor_parser.hh"
#include <iostream>

int test() {
    auto tensors = build_indexed_tensors();

    // Access layer 20 self_attn.k_proj
    tensor t = tensors["self_attn.k_proj.weight"][20];
    std::cout << t << std::endl;

    // Access embedding
     t = tensors["self_attn.k_proj.weight"][21];
    std::cout << t << std::endl;
    
    
    tensor embed = tensors["logits"][0]; 
    std::cout << embed << std::endl;
    return 0;
}
