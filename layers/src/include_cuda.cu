#include <cuda.h>
#include <cuda_bf16.h>
#include<fstream>
#include "tensor_parser.hh"
#include <iostream>

#include "layers_include.cuh"


inline void log_cache_host_range(
    std::ofstream& out,
    const __nv_bfloat16* d_cache_layer_base, // cache + layer_off
    int hidden_dim_kv,
    int layer_idx,
    const char* tag,   // "K" or "V"
    int start_pos,     // inclusive
    int end_pos,       // exclusive
    int max_cols )
{
    if (!out || start_pos >= end_pos) return;

    out << std::fixed << std::setprecision(6);
    out << "=== Layer " << layer_idx << ' ' << tag
        << "_cache dump: positions [" << start_pos << ", " << (end_pos - 1) << "] ===\n";

    std::vector<__nv_bfloat16> h_row(hidden_dim_kv);

    for (int pos = start_pos; pos < end_pos; ++pos) {
        size_t off = size_t(pos) * size_t(hidden_dim_kv);
        cudaError_t err = cudaMemcpy(h_row.data(),
                                     d_cache_layer_base + off,
                                     size_t(hidden_dim_kv) * sizeof(__nv_bfloat16),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            out << "[ERROR] memcpy failed at pos=" << pos << " : "
                << cudaGetErrorString(err) << "\n";
            return;
        }

        out << "[pos " << pos << "] ";
        int cols = (max_cols < hidden_dim_kv ? max_cols : hidden_dim_kv);
        for (int d = 0; d < cols; ++d) {
            out << __bfloat162float(h_row[d]) << (d + 1 < cols ? ' ' : '\n');
        }
    }
    out << '\n';
    out.flush();
}







void weights_read(tensor t, std::ifstream&  weights_file, __nv_bfloat16* weights_buffer){

    size_t weights_start_pos;
    size_t weights_end_pos;
    size_t weights_span;

    weights_file.clear(); 
    weights_start_pos = t.data_offsets[0];
    weights_end_pos = t.data_offsets[1];
    weights_span = weights_end_pos - weights_start_pos;
    // std::cout <<"start pos" << weights_start_pos << "end pos" << weights_end_pos<<std::endl;
    weights_file.seekg(static_cast<std::streamoff>(weights_start_pos), std::ios::beg);
    weights_file.read(reinterpret_cast<char*>(weights_buffer), static_cast<std::streamsize>(weights_span));
    if (!weights_file) {
        std::cout << "Failed to read weight at offset " << weights_start_pos << " for layer tensor.\n";
    }
}


size_t calculate_tensor_size(const tensor& t) {
    size_t size = 1;
    for (size_t dim : t.shape) {
        size *= dim;
    }
    return size;
}

// Initialize memory for a tensor with host and device pointers
void init_weights_memory(__nv_bfloat16** host_ptr, 
                        __nv_bfloat16** device_ptr_in,
                        __nv_bfloat16**  device_ptr_out,
                        const tensor& t) {
    size_t size = calculate_tensor_size(t);
    size_t size_in_bytes = size * sizeof(__nv_bfloat16);
    
    cudaMallocHost((void**)host_ptr, size_in_bytes);
    cudaMalloc((void**)device_ptr_in, size_in_bytes);
    cudaMalloc((void**)device_ptr_out, size_in_bytes);
}


void init_weights_memory(__nv_bfloat16** host_ptr, 
                        __nv_bfloat16** device_ptr,
                        const tensor& t) {
    size_t size = calculate_tensor_size(t);
    size_t size_in_bytes = size * sizeof(__nv_bfloat16);
    
    cudaMallocHost((void**)host_ptr, size_in_bytes);
    cudaMalloc((void**)device_ptr, size_in_bytes);
}

// Overload for device-only allocation
void init_weights_memory(__nv_bfloat16** device_ptr, 
                        const tensor& t) {
    size_t size = calculate_tensor_size(t);
    size_t size_in_bytes = size * sizeof(__nv_bfloat16);
    
    cudaMalloc((void**)device_ptr, size_in_bytes);
}

int find_max(const __nv_bfloat16 *array, int size) {
    if (size <= 0) {
        printf("Error: Empty array.\n");
        return 0.0f; // or handle error differently
    }

    float max_val = __bfloat162float(array[0]);
    int max_ind = 0;
    for (int i = 0; i < size; ++i) {
        if (__bfloat162float(array[i]) > max_val) {
            max_val = __bfloat162float(array[i]);
            max_ind = i;
            
        }
        // std::cout << "max_value:" << max_val << "index: "<< i << std::endl;
    }
    return max_ind;
}

