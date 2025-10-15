#include <cuda.h>
#include <cuda_bf16.h>


__global__ void embedding_matrix_func(__nv_bfloat16 *embeddings_out, __nv_bfloat16 *embeddings_matrix, int *token_ids, size_t embedding_dim, size_t sequence_len){
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //number of threads will be equal to number of tokens. each thread will get one row of the embedding matrix.
    if(idx < sequence_len){
        int token_id = token_ids[idx];
        for(int i = 0; i < embedding_dim; i++){
            
           embeddings_out[idx*embedding_dim + i] = embeddings_matrix[token_id*embedding_dim +i];

        }
    }
}



// void embedding_lookup_cpu(
//     __nv_bfloat16 *out,       // output [sequence_len x embedding_dim]
//     const __nv_bfloat16 *embedding_matrix, // [vocab_size x embedding_dim]
//     const int *token_ids,
//     int sequence_len,
//     int embedding_dim
// ) {
//     for (int i = 0; i < sequence_len; i++) {
//         int token = token_ids[i];
//         for (int j = 0; j < embedding_dim; j++) {
//             out[i * embedding_dim + j] =
//                 embedding_matrix[token * embedding_dim + j];
//         }
//     }
// }

// int test(){


//     int h_token_ids[] = {785, 50802, 1525, 3818, 315, 65085, 73773, 323, 2310, 20475, 61056, 12236, 2411, 825, 835, 315, 432, 264, 57819, 22361, 11, 2238, 3460, 369, 29519, 3037, 11, 715, 1030, 1012, 259, 11191, 311, 279, 7002, 13, 1084, 43794, 4936, 458, 22399, 3579, 11, 715, 803, 1091, 264, 81573, 6884, 25, 279, 3579, 315, 264, 883, 315, 911, 35398, 35299, 11, 448, 264, 8811, 3691, 296, 25112, 1777, 715, 323, 54783, 398, 43777, 4419, 13, 47108, 1865, 369, 279, 31149, 13, 1084, 572, 902, 990, 4460, 279, 11893, 13, 715, 7418, 518, 279, 1850, 315, 3039, 432, 572, 55352, 3238, 11, 323, 518, 3042, 279, 9072, 1482, 572, 3931, 1007, 715, 2337, 52021, 4115, 13, 1084, 572, 949, 315, 279, 8584, 6541, 304, 17975, 369, 65812, 10348, 13, 576, 715, 10063, 572, 8094, 24908, 705, 11, 323, 47108, 11, 879, 572, 26127, 85603, 323, 1030, 264, 762, 292, 960, 95871, 3403, 806, 715, 1290, 38348, 11, 3937, 13970, 11, 40119, 3807, 3039, 389, 279, 1616, 13, 1913, 1817, 20327, 11, 14002, 279, 11893, 7514, 64, 723, 11, 715, 279, 22361, 448, 279, 22399, 3579, 342, 27011, 504, 279, 7002, 13, 1084, 572, 825, 315, 1846, 9185, 892, 525, 773, 715, 6027, 2221, 429, 279, 6414, 1795, 498, 911, 979, 498, 3271, 13, 36854, 425, 59218, 3012, 3424, 47407, 1718, 14985, 11, 279, 17256, 23969, 715, 432, 10613, 13, 27368, 279, 10063, 264, 91932, 7743, 572, 5290, 700, 264, 1140, 315, 12396, 892, 1030, 2494, 311, 653, 448, 715, 279, 5670, 315, 23694, 12, 2475, 13, 576, 7743, 3697, 504, 458, 1508, 4825, 9317, 60364, 1075, 264, 81527, 832, 17846, 892, 14122, 715, 949, 315, 279, 7329, 315, 279, 1290, 24413, 7002, 13, 47108, 6519, 264, 3398, 323, 279, 7743, 64230, 14400, 11, 3498, 715, 279, 4244, 1033, 2058, 32037, 480, 13, 576, 14141, 320, 1782, 1013, 642, 2191, 11, 432, 572, 2598, 8, 1410, 387, 5103, 2061, 11, 714, 715, 1052, 572, 902, 1616, 315, 50026, 432, 1007, 6587, 13, 1260, 7726, 916, 311, 279, 3241, 25, 264, 2613, 812, 11, 90123, 7071, 11, 279, 715, 752, 351, 1440, 433, 315, 806, 2487, 16234, 45628, 553, 279, 6303, 916, 5583, 892, 1033, 279, 13794, 315, 279, 4614, 13, 5301, 6869, 715, 572, 1602, 6624, 11, 806, 3579, 17712, 274, 2325, 482, 11, 806, 6787, 11165, 6758, 553, 49247, 26785, 323, 48670, 59130, 41642, 323, 715, 279, 9255, 315, 279, 12406, 429, 1030, 1101, 9482, 13, 4710, 41151, 11, 1496, 1526, 279, 9311, 3241, 38452, 11, 279, 1879, 6966, 9255, 13, 6285, 304, 279, 8592, 2632, 1578, 67, 550, 315, 9956, 715, 1033, 420, 50768, 15797, 323, 21145, 5567, 1119, 18318, 1127, 11, 323, 3498, 279, 7015, 572, 47925, 323, 279, 12884, 264, 24939, 6303, 11, 1052, 198, 9324, 311, 387, 902, 12463, 304, 4113, 11, 3650, 279, 38468, 429, 1033, 61927, 291, 16852, 13, 576, 3691, 76, 25112, 610, 815, 4172, 715, 3579, 342, 27011, 1495, 504, 1449, 64340, 9131, 13, 2619, 572, 825, 389, 279, 3753, 63626, 7069, 14002, 13, 715, 36854, 425, 59218, 3012, 3424, 47407, 1718, 14985, 11, 279, 17256, 1053, 11, 1393, 279, 6319, 6414, 6966, 5538, 1119, 47108, 594, 1828, 13, 6285, 518, 8592, 3294, 715, 2441, 22361, 11, 21145, 518, 825, 9131, 11, 1320, 5677, 4946, 3641, 304, 279, 9956, 11, 6919, 2652, 18202, 323, 43963, 287, 279, 3175, 3409, 715, 1964, 16522, 7612, 13, 758, 279, 3041, 6010, 264, 35025, 78569, 2061, 1495, 1948, 279, 76295, 11, 90451, 369, 458, 9690, 1075, 264, 6303, 65, 62118, 11, 323, 715, 55967, 291, 3123, 1549, 448, 264, 2847, 4405, 10971, 13, 1084, 572, 279, 4282, 32522, 11, 65084, 33707, 1119, 1251, 594, 11030, 13, 576, 87042, 1521, 537, 715, 4925, 11, 4764, 13, 8278, 279, 35187, 10082, 80620, 13, 4710, 42374, 47108, 594, 1182, 279, 7743, 504, 279, 1013, 642, 2191, 572, 2058, 16584, 9695, 3123, 911, 23694, 12, 2475, 323, 279, 916, 1262, 12441, 478, 315, 279, 715, 85758, 14513, 70898, 9680, 13, 576, 1013, 642, 2191, 3949, 323, 33599, 24303, 13, 5765, 5112, 429, 47108, 1865, 11, 3403, 279, 715, 2188, 315, 264, 1602, 3347, 34855, 11, 1035, 387, 12771, 705, 553, 432, 11, 43543, 11, 773, 1293, 438, 566, 14616, 2878, 279, 2070, 315, 11129, 892, 715, 279, 9317, 60364, 53657, 11, 566, 1410, 387, 3884, 438, 1632, 438, 6617, 13, 2619, 572, 315, 3308, 902, 1616, 315, 14063, 3425, 498, 1033, 715, 1660, 15384, 518, 894, 2661, 4445, 13, 2585, 3545, 11, 476, 389, 1128, 1849, 11, 279, 35187, 10082, 58229, 304, 389, 894, 3842, 9067, 715, 572, 7942, 1778, 13, 1084, 572, 1496, 94783, 429, 807, 15384, 16083, 678, 279, 882, 13, 1988, 518, 894, 4379, 807, 1410, 19633, 304, 697, 715, 9067, 15356, 807, 4829, 311, 13, 1446, 1030, 311, 3887, 1177, 1521, 3887, 11, 504, 14132, 429, 6116, 30555, 1177, 304, 279, 24335, 429, 715, 1449, 5112, 498, 1865, 572, 916, 54503, 11, 323, 11, 3650, 304, 26298, 11, 1449, 7203, 69242, 1506, 13, 715, 47108, 8604, 806, 1182, 6519, 311, 279, 1013, 642, 2191, 13, 1084, 572, 29449, 11, 3498, 11, 438, 566, 1632, 6876, 11, 1496, 264, 1182, 646, 387, 30620, 13, 715, 362, 43887, 265, 3123, 279, 19640, 315, 29098, 11, 806, 1992, 315, 975, 11, 21271, 291, 12767, 323, 4158, 3403, 279, 43417, 88, 18414, 13, 1096, 11, 566, 715, 3381, 448, 264, 3378, 315, 39046, 1582, 5525, 1177, 419, 572, 7148, 11, 10178, 3283, 315, 362, 864, 4561, 3776, 11, 5086, 279, 4843, 1429, 94451, 315, 715, 279, 39921, 315, 506, 346, 9166, 13, 1260, 6679, 311, 36563, 700, 1045, 19990, 4938, 429, 1265, 3291, 1435, 3425, 7148, 1030, 2677, 715, 1012, 5008, 1075, 419, 13, 38970, 1052, 2677, 1493, 95141, 315, 5749, 1280, 64989, 33357, 14967, 11, 862, 11067, 557, 3018, 705, 448, 715, 293, 4943, 2787, 315, 44788, 11, 862, 11030, 71732, 448, 53943, 323, 862, 76295, 448, 44353, 768, 657, 11001, 11, 862, 14264, 13551, 14285, 715, 29711, 3173, 304, 678, 17961, 30, 1597, 279, 88838, 6594, 1380, 279, 61927, 15797, 2021, 404, 832, 304, 279, 3720, 323, 279, 686, 363, 12, 1923, 65, 607, 15718, 832, 715, 916, 279, 92640, 315, 73435, 26, 323, 279, 7482, 1380, 279, 32506, 1030, 22949, 264, 8131, 10900, 323, 1052, 1030, 91210, 705, 274, 539, 307, 715, 47928, 315, 22360, 43835, 819, 1075, 16158, 2832, 19757, 30, 1988, 432, 572, 902, 990, 11, 566, 1410, 537, 6099, 25, 4302, 14616, 315, 806, 198, 19990, 3650, 264, 4013, 315, 9906, 2852, 275, 1965, 11981, 30865, 2348, 902, 4004, 323, 10008, 34777, 6703, 1238, 13, 31906, 576, 19640, 315, 29098, 1177, 386, 2327, 81, 361, 11, 304, 5398, 22792, 1177, 572, 67734, 398, 2155, 504, 894, 1008, 1633, 304, 13929, 13, 1084, 572, 458, 715, 22399, 4510, 2396, 25880, 5944, 315, 54151, 287, 4158, 14175, 11, 68897, 705, 11, 51478, 1283, 51478, 11, 220, 18, 15, 15, 36256, 1119, 279, 3720, 13, 715, 5542, 1380, 47108, 14638, 432, 572, 1101, 3204, 311, 1349, 11, 12771, 700, 389, 1181, 4158, 3579, 304, 25777, 6524, 287, 11, 279, 2326, 715, 77001, 315, 279, 8554, 25, 31906, 57983, 3424, 21804, 5576, 31906, 62606, 1479, 1898, 3424, 16797, 32, 30519, 31906, 49839, 868, 8440, 3424, 3928, 787, 8977, 198};
//     int *d_token_ids;

//     // cudaMalloc((void **)&d_token_ids,sizeof(h_token_ids));
//     // cudaMemcpy(d_token_ids, h_token_ids, sizeof(h_token_ids), cudaMemcpyHostToDevice);
    
    
//     // int sequence_len = sizeof(h_token_ids)/sizeof(h_token_ids[0]);

//     int sequence_len = sizeof(h_token_ids)/sizeof(h_token_ids[0]);

//     cudaMalloc((void **)&d_token_ids, sequence_len * sizeof(int));
//     cudaMemcpy(d_token_ids, h_token_ids, sequence_len * sizeof(int), cudaMemcpyHostToDevice);

//     std::cout << sequence_len << std::endl;

//     std::vector<tensor> all_tensors;
//     all_tensors = parsed_tensors();
//     std::cout << all_tensors[0] << std::endl;

//     int embedding_matrix_size = all_tensors[0].shape[0] * all_tensors[0].shape[1];
//     int embedding_matrix_dim = all_tensors[0].shape[1];

//     __nv_bfloat16 *h_embedding_matrix = (__nv_bfloat16*)malloc(embedding_matrix_size*sizeof(__nv_bfloat16));
//     __nv_bfloat16 *d_embedding_matrix;
//     cudaMalloc((void **)&d_embedding_matrix,embedding_matrix_size*sizeof(__nv_bfloat16));


//     __nv_bfloat16 *h_embeddings_out = (__nv_bfloat16*)malloc(sequence_len*embedding_matrix_dim*sizeof(__nv_bfloat16));
//     __nv_bfloat16 *d_embeddings_out;
//     cudaMalloc((void **)&d_embeddings_out,sequence_len * embedding_matrix_dim * sizeof(__nv_bfloat16));


//     std::ifstream weights("../model_files/weights.bin", std::ios::binary);
//     weights.read(reinterpret_cast<char*>(h_embedding_matrix), embedding_matrix_size*sizeof(__nv_bfloat16));


//     // for (int i = 0; i < 5000; i++) {
//     //     float val = __nv_bfloat162float(h_embedding_matrix[i]);
//     //     std::cout << val << " ";
//     // }
 
//     cudaMemcpy(d_embedding_matrix, h_embedding_matrix, embedding_matrix_size*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice );
    

//     int threads = 512;
//     int blocks = (sequence_len + threads -1) / threads;

//     embedding_matrix_func<<<blocks, threads>>>(d_embeddings_out, d_embedding_matrix, d_token_ids, embedding_matrix_dim, sequence_len);

//     cudaDeviceSynchronize();
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         std::cout << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
//     }
//     cudaMemcpy(h_embeddings_out, d_embeddings_out, sequence_len * embedding_matrix_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost );

//     cudaDeviceSynchronize();

//     // for (int i = 0; i < (sequence_len * embedding_matrix_dim); i++) {
//     //     float val = __nv_bfloat162float(h_embeddings_out[i]);
//     //     std::cout << val << " ";
//     // }

//     // std::cout << std::endl;


        
//     // Convert half -> float for comparison
//     std::vector<float> gpu_out(sequence_len * embedding_matrix_dim);
//     for (int i = 0; i < sequence_len * embedding_matrix_dim; i++) {
//         gpu_out[i] = __bfloat162float(h_embeddings_out[i]);
//     }

//     // Run CPU reference
//     std::vector<__nv_bfloat16> cpu_out(sequence_len * embedding_matrix_dim);
//     embedding_lookup_cpu(cpu_out.data(),
//                         h_embedding_matrix,  // convert __nv_bfloat16 → float beforehand
//                         h_token_ids,
//                         sequence_len,
//                         embedding_matrix_dim);

//     // Compare
//     for (int i = 0; i < sequence_len * embedding_matrix_dim; i++) {
//         if (fabs(__bfloat162float(cpu_out[i]) - __bfloat162float(gpu_out[i])) > 1e-3) {
//             std::cout << "Mismatch at " << i
//                     << ": CPU=" << __bfloat162float(cpu_out[i])
//                     << " GPU=" << __bfloat162float(gpu_out[i]) << std::endl;

//         std::cout << "Match at " << i << ": CPU=" << __bfloat162float(cpu_out[i]) << " GPU=" << __bfloat162float(gpu_out[i]) << std::endl;
//     }




//     free(h_embedding_matrix);
//     free(h_embeddings_out);
//     cudaFree(d_embedding_matrix);
//     cudaFree(d_embeddings_out);
//     cudaFree(d_token_ids);


//     return 0;
// }
// }
