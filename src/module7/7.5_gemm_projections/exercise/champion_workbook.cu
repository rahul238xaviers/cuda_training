// ==============================================================================
// Module 7.5: GEMM & Linear Projection Suite — Champion Workbook
// ==============================================================================
// In this workbook, you will master the 7 production-grade GEMM & projection
// kernels derived directly from the LLaMA training engine:
// 1. High-Performance Register Micro-Tiled GEMM (BM=64, BN=64, BK=16, TM=4, TN=4)
// 2. Fused Dual-Projection SwiGLU FFN GEMM (gemm_ffn.metal):
//    C = SwiGLU(X * W_gate, X * W_up) without writing intermediate tensors to DRAM
// 3. Fused SwiGLU Down-Projection GEMM (fused_swiglu_gemm.metal):
//    C = SwiGLU(Gate, Up) * W_down with activation computed during shared memory load
// 4. BF16 GEMM with FP32 Accumulators (gemm_bf16.metal)
// 5. Fused QKV Projection GEMM: Splitting single combined projection into Q, K, V
//
// All exercises must print "Passed: X / Y tests." and return 0 on success.
// ==============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " code=" << err << " \"" << cudaGetErrorString(err)  \
                      << "\"" << std::endl;                                   \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

__device__ __forceinline__ float silu_act(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float swiglu_act(float gate, float up) {
    return silu_act(gate) * up;
}

// ==============================================================================
// Exercise 1: High-Performance Register Micro-Tiled GEMM
// Tile dimensions: BM = 64, BN = 64, BK = 16.
// Thread micro-tile: TM = 4, TN = 4.
// Threads per block: (64/4) x (64/4) = 16 x 16 = 256 threads.
// Each thread computes a 4x4 patch (16 elements) using 16 register accumulators.
// Shared memory:
//   __shared__ float s_A[64][17]; // 16 + 1 padding to eliminate bank conflicts
//   __shared__ float s_B[16][65]; // 64 + 1 padding
// ==============================================================================
const int BM = 64;
const int BN = 64;
const int BK = 16;
const int TM = 4;
const int TN = 4;

__global__ void gemm_register_tiled_64x64_kernel(const float* A, const float* B, float* C,
                                                 int M, int N, int K) {
    __shared__ float s_A[BM][BK + 1];
    __shared__ float s_B[BK][BN + 1];

    // TODO:
    // 1. Thread indices: tid_y = threadIdx.y (0 to 15), tid_x = threadIdx.x (0 to 15)
    //    Flat tid = tid_y * 16 + tid_x (0 to 255)
    // 2. Local register accumulators: float accum[TM][TN] = {0.0f};
    // 3. Loop over K in chunks of BK = 16:
    //    - Load s_A: 64 * 16 = 1024 floats. Each of the 256 threads loads 4 floats.
    //    - Load s_B: 16 * 64 = 1024 floats. Each of the 256 threads loads 4 floats.
    //    - __syncthreads()
    //    - For k from 0 to BK - 1:
    //        float reg_a[TM];
    //        for (int m = 0; m < TM; ++m) reg_a[m] = s_A[tid_y * TM + m][k];
    //        float reg_b[TN];
    //        for (int n = 0; n < TN; ++n) reg_b[n] = s_B[k][tid_x * TN + n];
    //        for (int m = 0; m < TM; ++m)
    //            for (int n = 0; n < TN; ++n)
    //                accum[m][n] += reg_a[m] * reg_b[n];
    //    - __syncthreads()
    // 4. Write out the 4x4 accumulators to C with boundary checks.
}

// ==============================================================================
// Exercise 2: Fused Dual-Projection SwiGLU FFN GEMM (gemm_ffn.metal)
// In LLaMA MLP:
//   Gate = X [M x K] * W_gate [K x N]
//   Up   = X [M x K] * W_up   [K x N]
//   Output [M x N] = SwiGLU(Gate, Up) = (Gate / (1 + exp(-Gate))) * Up
// Compute Gate and Up simultaneously in registers from shared memory, and write
// only the activated output to C.
// Block: (16, 16), Grid: ((N + 15) / 16, (M + 15) / 16)
// ==============================================================================
__global__ void gemm_ffn_swiglu_kernel(const float* X, const float* W_gate, const float* W_up,
                                       float* C, int M, int N, int K) {
    // TODO:
    // 1. row = blockIdx.y * blockDim.y + threadIdx.y;
    //    col = blockIdx.x * blockDim.x + threadIdx.x;
    // 2. Initialize float acc_gate = 0.0f, acc_up = 0.0f;
    // 3. Accumulate dot products over k = 0 to K-1:
    //    float x_val = X[row * K + k];
    //    acc_gate += x_val * W_gate[k * N + col];
    //    acc_up   += x_val * W_up[k * N + col];
    // 4. If row < M && col < N:
    //    C[row * N + col] = swiglu_act(acc_gate, acc_up);
}

// ==============================================================================
// Exercise 3: Fused SwiGLU Down-Projection GEMM (fused_swiglu_gemm.metal)
// Computes C [M x N] = SwiGLU(Gate [M x K], Up [M x K]) * W_down [K x N]
// Notice Gate and Up are [M x K], W_down is [K x N].
// In tile loading:
//   tile_A[r, k] is computed on-the-fly as: swiglu_act(Gate[r, k], Up[r, k])
// Then multiplied with W_down.
// Block: (16, 16), Grid: ((N + 15) / 16, (M + 15) / 16)
// ==============================================================================
__global__ void fused_swiglu_gemm_kernel(const float* Gate, const float* Up,
                                         const float* W_down, float* C,
                                         int M, int N, int K) {
    __shared__ float s_A[16][16];
    __shared__ float s_B[16][16];

    // TODO:
    // 1. Map row = blockIdx.y * 16 + threadIdx.y, col = blockIdx.x * 16 + threadIdx.x
    // 2. Loop over K in chunks of 16
    //    - Load s_A: if row < M && t * 16 + threadIdx.x < K:
    //        int k_idx = t * 16 + threadIdx.x;
    //        s_A[threadIdx.y][threadIdx.x] = swiglu_act(Gate[row * K + k_idx], Up[row * K + k_idx]);
    //      else: 0.0f
    //    - Load s_B: if t * 16 + threadIdx.y < K && col < N:
    //        s_B[threadIdx.y][threadIdx.x] = W_down[(t * 16 + threadIdx.y) * N + col];
    //      else: 0.0f
    //    - __syncthreads()
    //    - Multiply tile and accumulate
    //    - __syncthreads()
    // 3. Write out C[row * N + col]
}

// ==============================================================================
// Exercise 4: BF16 GEMM with FP32 Accumulators (gemm_bf16.metal)
// Inputs: A [M x K], B [K x N] in __nv_bfloat16.
// Output: C [M x N] in __nv_bfloat16.
// Math: All multiplications and summations must be performed in float (FP32).
// Output is cast back to __nv_bfloat16 via __float2bfloat16.
// ==============================================================================
__global__ void gemm_bf16_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B,
                                 __nv_bfloat16* C, int M, int N, int K) {
    // TODO:
    // 1. row = blockIdx.y * blockDim.y + threadIdx.y
    //    col = blockIdx.x * blockDim.x + threadIdx.x
    // 2. If row < M && col < N:
    //    float sum = 0.0f;
    //    for k = 0 to K-1:
    //        float a_val = __bfloat162float(A[row * K + k]);
    //        float b_val = __bfloat162float(B[k * N + col]);
    //        sum += a_val * b_val;
    //    C[row * N + col] = __float2bfloat16(sum);
}

// ==============================================================================
// Exercise 5: Fused QKV Projection GEMM
// In Multi-Head Attention, input X [M x D_in] is multiplied by a single packed
// projection weight W_qkv [D_in x (D_q + D_k + D_v)].
// This kernel projects and directly demuxes output rows into Q [M x D_q],
// K [M x D_k], and V [M x D_v].
// D_total = D_q + D_k + D_v.
// If col < D_q: write to Q[row * D_q + col]
// If col >= D_q && col < D_q + D_k: write to K[row * D_k + (col - D_q)]
// If col >= D_q + D_k: write to V[row * D_v + (col - D_q - D_k)]
// ==============================================================================
__global__ void fused_qkv_proj_kernel(const float* X, const float* W_qkv,
                                      float* Q, float* K, float* V,
                                      int M, int D_in, int D_q, int D_k, int D_v) {
    // TODO:
    // 1. row = blockIdx.y * blockDim.y + threadIdx.y (0 to M-1)
    //    col = blockIdx.x * blockDim.x + threadIdx.x (0 to D_total - 1)
    // 2. int D_total = D_q + D_k + D_v;
    //    if row < M && col < D_total:
    //        float sum = 0.0f;
    //        for d = 0 to D_in - 1:
    //            sum += X[row * D_in + d] * W_qkv[d * D_total + col];
    //        if col < D_q:
    //            Q[row * D_q + col] = sum;
    //        else if col < D_q + D_k:
    //            K[row * D_k + (col - D_q)] = sum;
    //        else
    //            V[row * D_v + (col - D_q - D_k)] = sum;
}

// ==============================================================================
// Verification Test Harness
// ==============================================================================
int main() {
    int passed = 0;
    const int total_tests = 5;

    // --------------------------------------------------------------------------
    // Test 1: Register Micro-Tiled GEMM (64x64)
    // --------------------------------------------------------------------------
    {
        int M = 64, N = 64, K = 64;
        std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.0f), h_ref(M * N, 0.0f);
        for (int i = 0; i < M * K; ++i) h_A[i] = 0.1f * ((i % 13) + 1);
        for (int i = 0; i < K * N; ++i) h_B[i] = 0.1f * ((i % 7) + 1);

        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) acc += h_A[m * K + k] * h_B[k * N + n];
                h_ref[m * N + n] = acc;
            }
        }

        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((N + BM - 1) / BM, (M + BN - 1) / BN);
        gemm_register_tiled_64x64_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < M * N; ++i) {
            if (std::fabs(h_C[i] - h_ref[i]) > 1e-2f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 1: 64x64 Register Micro-Tiled GEMM] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 1: 64x64 Register Micro-Tiled GEMM] FAILED" << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // --------------------------------------------------------------------------
    // Test 2: Fused Dual-Projection SwiGLU FFN GEMM
    // --------------------------------------------------------------------------
    {
        int M = 32, N = 32, K = 32;
        std::vector<float> h_X(M * K), h_W_gate(K * N), h_W_up(K * N),
                           h_C(M * N, 0.0f), h_ref(M * N, 0.0f);
        for (int i = 0; i < M * K; ++i) h_X[i] = 0.05f * ((i % 5) - 2);
        for (int i = 0; i < K * N; ++i) {
            h_W_gate[i] = 0.08f * ((i % 7) - 3);
            h_W_up[i]   = 0.06f * ((i % 9) - 4);
        }

        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float g = 0.0f, u = 0.0f;
                for (int k = 0; k < K; ++k) {
                    g += h_X[m * K + k] * h_W_gate[k * N + n];
                    u += h_X[m * K + k] * h_W_up[k * N + n];
                }
                float silu = g / (1.0f + std::exp(-g));
                h_ref[m * N + n] = silu * u;
            }
        }

        float *d_X, *d_W_gate, *d_W_up, *d_C;
        CHECK_CUDA(cudaMalloc(&d_X, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_W_gate, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_W_up, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_W_gate, h_W_gate.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_W_up, h_W_up.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        gemm_ffn_swiglu_kernel<<<grid, block>>>(d_X, d_W_gate, d_W_up, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < M * N; ++i) {
            if (std::fabs(h_C[i] - h_ref[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 2: Fused Dual-Proj SwiGLU FFN GEMM] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 2: Fused Dual-Proj SwiGLU FFN GEMM] FAILED" << std::endl;
        }

        cudaFree(d_X); cudaFree(d_W_gate); cudaFree(d_W_up); cudaFree(d_C);
    }

    // --------------------------------------------------------------------------
    // Test 3: Fused SwiGLU Down-Projection GEMM
    // --------------------------------------------------------------------------
    {
        int M = 32, N = 32, K = 32;
        std::vector<float> h_Gate(M * K), h_Up(M * K), h_W_down(K * N),
                           h_C(M * N, 0.0f), h_ref(M * N, 0.0f);
        for (int i = 0; i < M * K; ++i) {
            h_Gate[i] = 0.1f * ((i % 7) - 3);
            h_Up[i]   = 0.12f * ((i % 5) - 2);
        }
        for (int i = 0; i < K * N; ++i) h_W_down[i] = 0.08f * ((i % 9) - 4);

        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    float g = h_Gate[m * K + k];
                    float u = h_Up[m * K + k];
                    float sw = (g / (1.0f + std::exp(-g))) * u;
                    acc += sw * h_W_down[k * N + n];
                }
                h_ref[m * N + n] = acc;
            }
        }

        float *d_Gate, *d_Up, *d_W_down, *d_C;
        CHECK_CUDA(cudaMalloc(&d_Gate, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_Up, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_W_down, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_Gate, h_Gate.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_Up, h_Up.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_W_down, h_W_down.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        fused_swiglu_gemm_kernel<<<grid, block>>>(d_Gate, d_Up, d_W_down, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < M * N; ++i) {
            if (std::fabs(h_C[i] - h_ref[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 3: Fused SwiGLU Down-Proj GEMM] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 3: Fused SwiGLU Down-Proj GEMM] FAILED" << std::endl;
        }

        cudaFree(d_Gate); cudaFree(d_Up); cudaFree(d_W_down); cudaFree(d_C);
    }

    // --------------------------------------------------------------------------
    // Test 4: BF16 GEMM with FP32 Accumulators
    // --------------------------------------------------------------------------
    {
        int M = 32, N = 32, K = 32;
        std::vector<__nv_bfloat16> h_A(M * K), h_B(K * N), h_C(M * N);
        std::vector<float> h_ref(M * N, 0.0f);

        for (int i = 0; i < M * K; ++i) {
            float val = 0.2f * ((i % 7) + 1);
            h_A[i] = __float2bfloat16(val);
        }
        for (int i = 0; i < K * N; ++i) {
            float val = 0.15f * ((i % 5) + 1);
            h_B[i] = __float2bfloat16(val);
        }

        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    acc += __bfloat162float(h_A[m * K + k]) * __bfloat162float(h_B[k * N + n]);
                }
                h_ref[m * N + n] = acc;
            }
        }

        __nv_bfloat16 *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(__nv_bfloat16)));

        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(__nv_bfloat16)));

        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        gemm_bf16_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < M * N; ++i) {
            float out_f = __bfloat162float(h_C[i]);
            if (std::fabs(out_f - h_ref[i]) > 0.05f) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::cout << "[Test 4: BF16 GEMM with FP32 Accumulators] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 4: BF16 GEMM with FP32 Accumulators] FAILED" << std::endl;
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // --------------------------------------------------------------------------
    // Test 5: Fused QKV Projection GEMM
    // --------------------------------------------------------------------------
    {
        int M = 16, D_in = 32, D_q = 16, D_k = 8, D_v = 8;
        int D_total = D_q + D_k + D_v;
        std::vector<float> h_X(M * D_in), h_W(D_in * D_total),
                           h_Q(M * D_q, 0.0f), h_K(M * D_k, 0.0f), h_V(M * D_v, 0.0f),
                           h_Q_ref(M * D_q, 0.0f), h_K_ref(M * D_k, 0.0f), h_V_ref(M * D_v, 0.0f);

        for (int i = 0; i < M * D_in; ++i) h_X[i] = 0.1f * ((i % 9) + 1);
        for (int i = 0; i < D_in * D_total; ++i) h_W[i] = 0.05f * ((i % 11) + 1);

        for (int m = 0; m < M; ++m) {
            for (int col = 0; col < D_total; ++col) {
                float acc = 0.0f;
                for (int d = 0; d < D_in; ++d) {
                    acc += h_X[m * D_in + d] * h_W[d * D_total + col];
                }
                if (col < D_q) h_Q_ref[m * D_q + col] = acc;
                else if (col < D_q + D_k) h_K_ref[m * D_k + (col - D_q)] = acc;
                else h_V_ref[m * D_v + (col - D_q - D_k)] = acc;
            }
        }

        float *d_X, *d_W, *d_Q, *d_K, *d_V;
        CHECK_CUDA(cudaMalloc(&d_X, M * D_in * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_W, D_in * D_total * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_Q, M * D_q * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_K, M * D_k * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_V, M * D_v * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), M * D_in * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), D_in * D_total * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_Q, 0, M * D_q * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_K, 0, M * D_k * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_V, 0, M * D_v * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((D_total + 15) / 16, (M + 15) / 16);
        fused_qkv_proj_kernel<<<grid, block>>>(d_X, d_W, d_Q, d_K, d_V, M, D_in, D_q, D_k, D_v);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_Q.data(), d_Q, M * D_q * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_K.data(), d_K, M * D_k * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_V.data(), d_V, M * D_v * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (size_t i = 0; i < h_Q.size(); ++i) {
            if (std::fabs(h_Q[i] - h_Q_ref[i]) > 1e-3f) ok = false;
        }
        for (size_t i = 0; i < h_K.size(); ++i) {
            if (std::fabs(h_K[i] - h_K_ref[i]) > 1e-3f) ok = false;
        }
        for (size_t i = 0; i < h_V.size(); ++i) {
            if (std::fabs(h_V[i] - h_V_ref[i]) > 1e-3f) ok = false;
        }

        if (ok) {
            std::cout << "[Test 5: Fused QKV Projection GEMM] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 5: Fused QKV Projection GEMM] FAILED" << std::endl;
        }

        cudaFree(d_X); cudaFree(d_W); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    }

    std::cout << "Passed: " << passed << " / " << total_tests << " tests." << std::endl;
    return (passed == total_tests) ? 0 : 1;
}
