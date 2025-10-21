
#include <cuda.h>
#include <cuda_bf16.h>
#include "utils.hh"
#include "mma.h"

using namespace nvcuda::wmma;

// __global__ void matrix_mul(__nv_bfloat16 *A, __nv_bfloat16 *B, __nv_bfloat16 *C, int M, int N, int K) // (M X N) x (N x K)
// {

//     int col = threadIdx.x + blockIdx.x*blockDim.x;
//     int row = threadIdx.y + blockIdx.y*blockDim.y;


//     if (row < M && col < K){
//         float sum = 0;
//         for(int i = 0; i < N; i++){
//             sum += __bfloat162float(A[row*N + i]) * __bfloat162float(B[col*N + i]);
//         }
//         C[col+K*row] = __float2bfloat16(sum);
//     }
// }



// Kernel: A(M×K) × B(N×K)^T = C(M×N)
// A is row-major: A[i][k] at index i*K + k
// B is row-major: B[j][k] at index j*K + k (we want B^T[k][j])
// Output C is row-major: C[i][j] at index i*N + j
// __global__ void matrix_mul(
//     __nv_bfloat16* A, 
//     __nv_bfloat16* B, 
//     __nv_bfloat16* C, 
//     int M, 
//     int N, 
//     int K)
// {
//     // Calculate warp ID within grid
//     int warpsPerBlock = blockDim.x / WARP_SIZE;
//     int warpId = blockIdx.x * warpsPerBlock + (threadIdx.x / WARP_SIZE);
    
//     // Calculate number of tiles in M and N dimensions
//     int tilesM = (M + TILE_SIZE - 1) / TILE_SIZE;
//     int tilesN = (N + TILE_SIZE - 1) / TILE_SIZE;
//     int totalTiles = tilesM * tilesN;
    
//     if (warpId >= totalTiles) return;
    
//     // Determine which output tile this warp handles
//     int tileRow = warpId / tilesN;
//     int tileCol = warpId % tilesN;
    
//     // Actual starting indices in the output matrix
//     int rowStart = tileRow * TILE_SIZE;
//     int colStart = tileCol * TILE_SIZE;
    
//     // Check if this tile is within bounds
//     if (rowStart >= M || colStart >= N) return;
    
//     // Shared memory for loading tiles
//     __shared__ __nv_bfloat16 sharedA[4 * TILE_SIZE * TILE_SIZE]; // 4 warps worth
//     __shared__ __nv_bfloat16 sharedB[4 * TILE_SIZE * TILE_SIZE];
    
//     // Each warp gets its own portion of shared memory
//     int warpInBlock = threadIdx.x / WARP_SIZE;
//     __nv_bfloat16* mySharedA = sharedA + warpInBlock * TILE_SIZE * TILE_SIZE;
//     __nv_bfloat16* mySharedB = sharedB + warpInBlock * TILE_SIZE * TILE_SIZE;
    
//     // Create WMMA fragments
//     fragment<matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, __nv_bfloat16, row_major> aFrag;
//     fragment<matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, __nv_bfloat16, col_major> bFrag;
//     fragment<accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> cFragAcc;
    
//     // Initialize accumulator to 0
//     fill_fragment(cFragAcc, 0.0f);
    
//     int laneId = threadIdx.x % WARP_SIZE;
    
//     // Loop over K dimension in steps of TILE_SIZE
//     int numKTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
//     for (int kt = 0; kt < numKTiles; kt++) {
//         int kStart = kt * TILE_SIZE;
        
//         // Load A tile into shared memory: A[rowStart:rowStart+16, kStart:kStart+16]
//         // A is row-major, so A[i][k] is at A[i*K + k]
//         for (int i = laneId; i < TILE_SIZE * TILE_SIZE; i += WARP_SIZE) {
//             int localRow = i / TILE_SIZE;
//             int localCol = i % TILE_SIZE;
//             int globalRow = rowStart + localRow;
//             int globalCol = kStart + localCol;
            
//             if (globalRow < M && globalCol < K) {
//                 mySharedA[localRow * TILE_SIZE + localCol] = A[globalRow * K + globalCol];
//             } else {
//                 mySharedA[localRow * TILE_SIZE + localCol] = __float2bfloat16(0.0f);
//             }
//         }
        
//         // Load B tile into shared memory for transpose: we want B^T[kStart:kStart+16, colStart:colStart+16]
//         // B is row-major (N×K), B[j][k] is at B[j*K + k]
//         // We want B^T[k][j] which is B[j][k]
//         // For col-major storage of B^T: B^T[k][j] goes to column k, row j in col-major
//         // In col-major: element at row i, col j is at j*TILE_SIZE + i
//         for (int i = laneId; i < TILE_SIZE * TILE_SIZE; i += WARP_SIZE) {
//             int localRow = i / TILE_SIZE; // This will be the K dimension (0-15)
//             int localCol = i % TILE_SIZE; // This will be the N dimension (0-15)
            
            
//             int globalK = colStart + localRow;
//             int globalN = kStart + localCol;

//             if (globalK < K && globalN < N) {
//                 mySharedB[localRow * TILE_SIZE + localCol] = B[globalK * N + globalN];
//             } else {
//                 mySharedB[localRow * TILE_SIZE + localCol] = __float2bfloat16(0.0f);
//             }
//         }
        
//         __syncwarp();
        
//         // Load from shared memory
//         load_matrix_sync(aFrag, mySharedA, TILE_SIZE);
//         load_matrix_sync(bFrag, mySharedB, TILE_SIZE);
        
//         // Multiply-accumulate
//         mma_sync(cFragAcc, aFrag, bFrag, cFragAcc);
        
//         __syncwarp();
//     }
    
//     // Store result using shared memory
//     // WMMA only supports storing float accumulators, so we store as float then convert
//     __shared__ float sharedCFloat[4 * TILE_SIZE * TILE_SIZE];
//     __shared__ __nv_bfloat16 sharedC[4 * TILE_SIZE * TILE_SIZE];
//     float* mySharedCFloat = sharedCFloat + warpInBlock * TILE_SIZE * TILE_SIZE;
//     __nv_bfloat16* mySharedC = sharedC + warpInBlock * TILE_SIZE * TILE_SIZE;
    
//     // Store float accumulator to shared memory
//     store_matrix_sync(mySharedCFloat, cFragAcc, TILE_SIZE, mem_row_major);
    
//     __syncwarp();
    
//     // Convert float to bfloat16 in shared memory
//     for (int i = laneId; i < TILE_SIZE * TILE_SIZE; i += WARP_SIZE) {
//         mySharedC[i] = __float2bfloat16(mySharedCFloat[i]);
//     }
    
//     __syncwarp();
    
//     // Write to global memory
//     for (int i = laneId; i < TILE_SIZE * TILE_SIZE; i += WARP_SIZE) {
//         int localRow = i / TILE_SIZE;
//         int localCol = i % TILE_SIZE;
//         int globalRow = rowStart + localRow;
//         int globalCol = colStart + localCol;
        
//         if (globalRow < M && globalCol < N) {
//             C[globalRow * N + globalCol] = mySharedC[i];
//         }
//     }
// }

__global__ void matrix_mul(
     __nv_bfloat16* __restrict__ A,
     __nv_bfloat16* __restrict__ B, // stored transposed: B[k*N + n]
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    // warp and tile bookkeeping
    int warpsPerBlock = blockDim.x / WARP_SIZE;
    int warpId = blockIdx.x * warpsPerBlock + (threadIdx.x / WARP_SIZE);

    int tilesM = (M + TILE_SIZE - 1) / TILE_SIZE; // along rows (M)
    int tilesK = (K + TILE_SIZE - 1) / TILE_SIZE; // along cols (K)
    int totalTiles = tilesM * tilesK;
    if (warpId >= totalTiles) return;

    int tileRow = warpId / tilesK; // which tile of M
    int tileCol = warpId % tilesK; // which tile of K

    int rowStart = tileRow * TILE_SIZE;
    int colStart = tileCol * TILE_SIZE;

    if (rowStart >= M || colStart >= K) return;

    // Shared memory sized to hold 4 warps worth of tiles (adjust if threadsPerBlock differs)
    extern __shared__ __nv_bfloat16 shm[]; // dynamic shared memory
    // We'll partition the shared memory manually: first half for A, second half for B
    // required size per warp: TILE_SIZE*TILE_SIZE for A and TILE_SIZE*TILE_SIZE for B
    // We'll compute offsets:
    int warpInBlock = threadIdx.x / WARP_SIZE;
    int perWarpElems = TILE_SIZE * TILE_SIZE;
    __nv_bfloat16* sharedA = shm + (warpInBlock * 2 + 0) * perWarpElems; // per-warp A tile
    __nv_bfloat16* sharedB = shm + (warpInBlock * 2 + 1) * perWarpElems; // per-warp B tile (in column-major for WMMA)

    // WMMA fragments
    fragment<matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, __nv_bfloat16, row_major> aFrag;
    fragment<matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, __nv_bfloat16, row_major> bFrag;
    fragment<accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> cFrag;
    fill_fragment(cFrag, 0.0f);

    int laneId = threadIdx.x % WARP_SIZE;
    int numNTiles = (N + TILE_SIZE - 1) / TILE_SIZE; // iterate over N (inner dimension)

    // Loop over N in TILE_SIZE chunks
    for (int nt = 0; nt < numNTiles; ++nt) {
        int nStart = nt * TILE_SIZE; // start index along N for this tile

        // --- Load A tile into sharedA (row-major)
        // Each warp-thread cooperatively loads the TILE_SIZE*TILE_SIZE elements
        for (int idx = laneId; idx < perWarpElems; idx += WARP_SIZE) {
            int localR = idx / TILE_SIZE; // 0..15 (rows inside tile)
            int localC = idx % TILE_SIZE; // 0..15 (cols inside tile -> along N)
            int globalR = rowStart + localR; // M dimension
            int globalN = nStart + localC;   // N dimension

            if (globalR < M && globalN < N) {
                sharedA[localR * TILE_SIZE + localC] = A[globalR * N + globalN];
            } else {
                sharedA[localR * TILE_SIZE + localC] = __float2bfloat16(0.0f);
            }
        }

        // --- Load B tile into sharedB
        // B is stored transposed as B[k][n] at B[k*N + n].
        // We want to form the tile representing B^T[n][k] (i.e. shape TILE x TILE),
        // but for WMMA bFrag declared as col_major, we must store sharedB in column-major:
        // element at (row, col) in the TILE must be placed at index col*TILE + row.
        for (int idx = laneId; idx < perWarpElems; idx += WARP_SIZE) {
            int localR = idx / TILE_SIZE; // along K dimension (0..15)
            int localC = idx % TILE_SIZE; // along N dimension (0..15)
            int globalK = colStart + localR; // K dimension index (this is the output col)
            int globalN = nStart + localC;   // N dimension index (inner)

            // B is stored as B[k * N + n]
            if (globalK < K && globalN < N) {
                // store in column-major layout for bFrag:
                // sharedB[col * TILE + row] = value
                sharedB[localC * TILE_SIZE + localR] = B[globalK * N + globalN];
            } else {
                sharedB[localC * TILE_SIZE + localR] = __float2bfloat16(0.0f);
            }
        }

        // Ensure all warp lanes have completed shared loads
        __syncwarp();

        // Load fragments from shared memory into WMMA fragments
        // Leading dimension/stride = TILE_SIZE for both calls
        load_matrix_sync(aFrag, sharedA, TILE_SIZE); // row-major tile
        load_matrix_sync(bFrag, sharedB, TILE_SIZE); // col-major tile

        // Multiply-accumulate
        mma_sync(cFrag, aFrag, bFrag, cFrag);

        __syncwarp();
    } // end loop over N tiles

    // Store cFrag to shared float buffer (WMMA requires float accumulator)
    // allocate per-warp float buffer in shared memory (reuse same dynamic shm region but after the bf16 tiles)
    float* sharedCfloat = reinterpret_cast<float*>(shm + (warpsPerBlock * 2) * perWarpElems) + warpInBlock * perWarpElems;
    // Note: We reserved enough dynamic shared memory in launcher (see below)

    // store fragment to shared float memory (row-major)
    store_matrix_sync(sharedCfloat, cFrag, TILE_SIZE, mem_row_major);
    __syncwarp();

    // Convert float -> bfloat16 in shared space (use the bf16 slot area: reuse sharedA area for conversion buffer)
    // We'll write converted values into a small per-warp bf16 buffer (reuse sharedA area of this warp)
    for (int idx = laneId; idx < perWarpElems; idx += WARP_SIZE) {
        // reuse sharedA as output holder for converted C
        sharedA[idx] = __float2bfloat16(sharedCfloat[idx]);
    }
    __syncwarp();

    // Write results to global memory C (row-major C[row * K + col])
    for (int idx = laneId; idx < perWarpElems; idx += WARP_SIZE) {
        int localR = idx / TILE_SIZE;
        int localC = idx % TILE_SIZE;
        int globalR = rowStart + localR;
        int globalC = colStart + localC;
        if (globalR < M && globalC < K) {
            C[globalR * K + globalC] = sharedA[idx];
        }
    }
}