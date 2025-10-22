#include <cuda.h>
#include <cuda_bf16.h>
#include<fstream>
#include "tensor_parser.hh"
#include <iostream>

#include "layers_include.cuh"

#include <string>

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



inline void startCudaTimer(cudaEvent_t &start, cudaEvent_t &stop) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
}

inline float stopCudaTimer(cudaEvent_t &start, cudaEvent_t &stop, const char* label, std::string layer) {
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



void weights_read(tensor t, std::ifstream&  weights_file, __nv_bfloat16* weights_buffer){


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    startCudaTimer(start, stop);
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

    stopCudaTimer(start, stop, "Time in weights copy", t.tensor_name);
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


void kv_copy_layer_to_cache_prefill(
    ModelBuffers* buffer,
    int           i,               // layer index
    page_table*   kv_cache_seq1,   // head of page list
    int           page_size        // tokens per page
){
    const int tokens_per_page = page_size;                         
    const size_t vec_elems     = (size_t)buffer->hidden_dim_kv;
    const size_t pitch_bytes   = vec_elems * sizeof(__nv_bfloat16);
    const size_t width_bytes   = pitch_bytes;                 // copy a full token vector per row

    int sequence_len = buffer->sequence_len;



    if (sequence_len <= tokens_per_page) {
        const size_t kv_dim      = buffer->hidden_dim_kv;
        const size_t dpitch = (size_t)buffer->number_of_layers * kv_dim * sizeof(__nv_bfloat16);;
        const size_t spitch = kv_dim * sizeof(__nv_bfloat16);;
        const int height    = sequence_len;                   // rows = tokens

        // Copy K
        cudaMemcpy2D(&buffer->k_cache[i * buffer->hidden_dim_kv],  dpitch, buffer->K, spitch, width_bytes, height, cudaMemcpyDeviceToDevice);

        // Copy V
        cudaMemcpy2D(&buffer->v_cache[i * buffer->hidden_dim_kv], dpitch, buffer->V, spitch, width_bytes, height, cudaMemcpyDeviceToDevice);
    } else {
        // Paged path
        // How many pages do we need?
        const int pages_required = (sequence_len + tokens_per_page - 1) / tokens_per_page;

        page_table* temp = kv_cache_seq1; // head page
        if (!temp) { /* handle null / allocate first page */ }

        int copied_tokens = 0;
        for (int k = 0; k < pages_required; k++) {
            if (!temp) {
                break;
            }

            // How many tokens to write into this page
            int remaining      = sequence_len - copied_tokens;
            int this_page_rows = (remaining < tokens_per_page) ? remaining : tokens_per_page;

            const size_t kv_dim      = buffer->hidden_dim_kv;                // 1024
            const size_t src_pitch   = kv_dim * sizeof(__nv_bfloat16);       // row = kv vector
            const size_t dst_pitch   = (size_t)buffer->number_of_layers * kv_dim * sizeof(__nv_bfloat16); // jump over all layers per token
            const size_t width_bytes = src_pitch;

            
            // K
            cudaMemcpy2D(temp->k_page_ptr + (size_t)i * kv_dim, dst_pitch, buffer->K + (size_t)copied_tokens * kv_dim, src_pitch, width_bytes, this_page_rows,  cudaMemcpyDeviceToDevice);

            // V
            cudaMemcpy2D(temp->v_page_ptr + (size_t)i * kv_dim, dst_pitch, buffer->V + (size_t)copied_tokens * kv_dim, src_pitch, width_bytes, this_page_rows, cudaMemcpyDeviceToDevice);

            // Bookkeeping
            temp->page_allocated = this_page_rows;  // optional
            copied_tokens += this_page_rows;
            temp = temp->ptr_to_next_page;
        }
    }

}




void kv_copy_layer_to_cache_decode(
    ModelBuffers* buffer,
    int           i,               // layer index
    page_table*   kv_cache_seq1,   // head of page list
    int           page_size        // tokens per page
){

    size_t pos = buffer->sequence_len - 1;
    int page_idx = pos / page_size;
    int offset_in_page = pos % page_size;

    // walk to the correct page
    page_table* page = kv_cache_seq1;
    for (int p = 0; p < page_idx && page; ++p)
        page = page->ptr_to_next_page;
    if (!page) {
        fprintf(stderr, "Error: Page %d not allocated (pos=%zu, page_size=%d)\n", page_idx, pos, page_size);
        page_table* new_page = kv_cache_seq1;
        for (int p = 0; p < page_idx -1; ++p)
            new_page = new_page->ptr_to_next_page;
        new_page->ptr_to_next_page = create_page_list(1);
        page = new_page->ptr_to_next_page;
        int elements_per_page =  page_size * buffer->number_of_layers * (size_t)buffer->hidden_dim_kv;;
        allocate_page_buffers(page, elements_per_page);
        if(page){
            std::cout << "new page allocated" << std::endl;
        }

    }

    // compute offset inside that page
    size_t layer_off = (size_t)offset_in_page * buffer->number_of_layers * buffer->hidden_dim_kv  // jump to token
                    + (size_t)i * buffer->hidden_dim_kv;   

    std::cout <<"before cache copy in decode" << std::endl;
    // write into the correct page’s memory
    cudaMemcpy(&page->k_page_ptr[layer_off], buffer->K,
            buffer->hidden_dim_kv * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice);

    cudaMemcpy(&page->v_page_ptr[layer_off], buffer->V,
            buffer->hidden_dim_kv * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice);



}
