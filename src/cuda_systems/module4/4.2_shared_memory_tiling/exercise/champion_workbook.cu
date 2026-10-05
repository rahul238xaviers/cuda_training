#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

void reportStatus(const std::string &name, bool passed) {
    std::cout << "  " << std::left << std::setw(65) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

// -----------------------------------------------------------------------------
// PROBLEM 1: 2D Register Micro-Tiling (TM=4, TN=4, 16 Accumulators per Thread)
// -----------------------------------------------------------------------------
__global__ void kernel_gemm_register_tiling_2d(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K
) {
    // Block computes BM=32, BN=32. Each thread computes TM=4, TN=4.
    // Thread block has (32/4) x (32/4) = 8x8 = 64 threads!
    const int BM = 32, BN = 32, BK = 8;
    const int TM = 4, TN = 4;
    __shared__ float s_A[BM][BK + 1]; // +1 padding
    __shared__ float s_B[BK][BN + 1];

    // Thread-private register accumulators
    float c_reg[TM][TN] = {0.0f};

    // TODO: 1. Loop over K in steps of BK
    // TODO: 2. Collaboratively load s_A and s_B
    // TODO: 3. __syncthreads();
    // TODO: 4. Inner loop: load TM elements from s_A into registers, TN elements from s_B into registers
    // TODO: 5. Perform TM x TN outer-product FMA updates in registers
    // TODO: 6. __syncthreads();
    // TODO: 7. Store 16 elements per thread to C
}

// -----------------------------------------------------------------------------
// PROBLEM 2: Double-Buffered Shared Memory Pipeline for GEMM
// -----------------------------------------------------------------------------
__global__ void kernel_gemm_double_buffered(
    const float *A,
    const float *B,
    float *C,
    int N
) {
    const int TILE = 16;
    __shared__ float s_A[2][TILE][TILE];
    __shared__ float s_B[2][TILE][TILE];

    // TODO: Prefetch tile 0 into buffer 0
    // Loop: compute on buffer `write_stage ^ 1`, prefetch into buffer `write_stage`
}

// -----------------------------------------------------------------------------
// PROBLEM 3: Fused GEMM + SwiGLU Activation in Shared/Register Memory
// Gate projection + Up projection computed simultaneously, activation fused in registers:
// out = (gate * up) / (1.0f + exp(-gate))
// -----------------------------------------------------------------------------
__global__ void kernel_fused_gemm_swiglu(
    const float *A,
    const float *W_gate,
    const float *W_up,
    float *out,
    int M, int N, int K
) {
    // Compute both projections and fuse SiLU(gate) * up before writing out!
    // TODO: Implement fused matrix multiply and SwiGLU activation
}

// -----------------------------------------------------------------------------
// PROBLEM 4: Grouped Query Attention (GQA) Head Projection GEMM
// GQA: Multiple query heads share the same Key/Value head:
// kv_head = q_head / gqa_group_size
// -----------------------------------------------------------------------------
__global__ void kernel_gemm_gqa_tiled(
    const float *Q,
    const float *K_shared,
    float *scores,
    int num_q_heads,
    int gqa_ratio,
    int seq_len,
    int head_dim
) {
    // TODO: Tiled matrix multiply of Query head with broadcasted Key head
}

// -----------------------------------------------------------------------------
// PROBLEM 5: Dynamic Shared Memory Feed-Forward Network Projection (FFN GEMM)
// -----------------------------------------------------------------------------
extern __shared__ float s_dyn_gemm[];

__global__ void kernel_ffn_gemm_dynamic(
    const float *in,
    const float *weight,
    float *out,
    int M, int N, int K,
    int BM, int BN, int BK
) {
    // Partition s_dyn_gemm dynamically into s_A (BM * BK) and s_B (BK * BN)
    // TODO: Implement dynamic tiled FFN GEMM
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: 4.2 Shared Memory Matrix Tiling (Champion Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: 2D Register Micro-Tiling
    {
        int M = 32, N = 32, K = 32;
        std::vector<float> h_A(M * K, 1.0f), h_B(K * N, 2.0f), h_C(M * N, 0.0f);
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));
        cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(8, 8); // 64 threads computing 32x32
        dim3 grid(1, 1);
        kernel_gemm_register_tiling_2d<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < M * N; ++i) {
            if (std::fabs(h_C[i] - (K * 2.0f)) > 1e-3f) { ok = false; break; }
        }
        reportStatus("Problem 1: 2D Register Micro-Tiling (TM=4, TN=4, 16 Outputs/Thread)", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // Test 2: Double-Buffered GEMM
    {
        int N = 32;
        std::vector<float> h_A(N * N, 2.0f), h_B(N * N, 3.0f), h_C(N * N, 0.0f);
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, N * N * sizeof(float));
        cudaMalloc(&d_B, N * N * sizeof(float));
        cudaMalloc(&d_C, N * N * sizeof(float));
        cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid(N / 16, N / 16);
        kernel_gemm_double_buffered<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < N * N; ++i) {
            if (std::fabs(h_C[i] - (N * 6.0f)) > 1e-3f) { ok = false; break; }
        }
        reportStatus("Problem 2: Double-Buffered Shared Memory Pipeline for GEMM", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // Test 3: Fused GEMM + SwiGLU
    {
        int M = 16, N = 16, K = 16;
        std::vector<float> h_A(M * K, 1.0f);
        std::vector<float> h_Wgate(K * N, 0.5f);
        std::vector<float> h_Wup(K * N, 2.0f);
        std::vector<float> h_out(M * N, 0.0f);

        float *d_A, *d_Wgate, *d_Wup, *d_out;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_Wgate, K * N * sizeof(float));
        cudaMalloc(&d_Wup, K * N * sizeof(float));
        cudaMalloc(&d_out, M * N * sizeof(float));

        cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Wgate, h_Wgate.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Wup, h_Wup.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

        kernel_fused_gemm_swiglu<<<1, 256>>>(d_A, d_Wgate, d_Wup, d_out, M, N, K);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

        // Ground truth: gate = 16 * 0.5 = 8.0f; up = 16 * 2.0 = 32.0f
        // SwiGLU = 8.0f / (1.0f + exp(-8.0f)) * 32.0f
        float gate = (float)K * 0.5f;
        float up = (float)K * 2.0f;
        float exp_val = (gate / (1.0f + std::exp(-gate))) * up;

        bool ok = true;
        for (int i = 0; i < M * N; ++i) {
            if (std::fabs(h_out[i] - exp_val) > 1e-3f) { ok = false; break; }
        }
        reportStatus("Problem 3: Fused GEMM + SwiGLU Activation in Shared/Registers", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_A); cudaFree(d_Wgate); cudaFree(d_Wup); cudaFree(d_out);
    }

    // Test 4: GQA GEMM
    {
        int num_q_heads = 4, gqa_ratio = 2, seq_len = 16, head_dim = 16;
        int num_kv_heads = num_q_heads / gqa_ratio;
        std::vector<float> h_Q(num_q_heads * seq_len * head_dim, 1.0f);
        std::vector<float> h_K(num_kv_heads * seq_len * head_dim, 2.0f);
        std::vector<float> h_scores(num_q_heads * seq_len * seq_len, 0.0f);

        float *d_Q, *d_K, *d_scores;
        cudaMalloc(&d_Q, h_Q.size() * sizeof(float));
        cudaMalloc(&d_K, h_K.size() * sizeof(float));
        cudaMalloc(&d_scores, h_scores.size() * sizeof(float));

        cudaMemcpy(d_Q, h_Q.data(), h_Q.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, h_K.data(), h_K.size() * sizeof(float), cudaMemcpyHostToDevice);

        kernel_gemm_gqa_tiled<<<num_q_heads, 256>>>(d_Q, d_K, d_scores, num_q_heads, gqa_ratio, seq_len, head_dim);
        cudaDeviceSynchronize();

        cudaMemcpy(h_scores.data(), d_scores, h_scores.size() * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        float expected = (float)head_dim * (1.0f * 2.0f);
        for (size_t i = 0; i < h_scores.size(); ++i) {
            if (std::fabs(h_scores[i] - expected) > 1e-3f) { ok = false; break; }
        }
        reportStatus("Problem 4: Grouped Query Attention (GQA) Head Projection GEMM", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_scores);
    }

    // Test 5: Dynamic Shared Memory FFN GEMM
    {
        int M = 16, N = 16, K = 16;
        int BM = 16, BN = 16, BK = 16;
        std::vector<float> h_in(M * K, 2.0f), h_w(K * N, 2.0f), h_out(M * N, 0.0f);
        float *d_in, *d_w, *d_out;
        cudaMalloc(&d_in, M * K * sizeof(float));
        cudaMalloc(&d_w, K * N * sizeof(float));
        cudaMalloc(&d_out, M * N * sizeof(float));
        cudaMemcpy(d_in, h_in.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, h_w.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

        size_t smem_bytes = (BM * BK + BK * BN) * sizeof(float);
        kernel_ffn_gemm_dynamic<<<1, 256, smem_bytes>>>(d_in, d_w, d_out, M, N, K, BM, BN, BK);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < M * N; ++i) {
            if (std::fabs(h_out[i] - (K * 4.0f)) > 1e-3f) { ok = false; break; }
        }
        reportStatus("Problem 5: Dynamic Shared Memory Feed-Forward Network GEMM", ok);
        if (ok) passed++;
        total++;

        cudaFree(d_in); cudaFree(d_w); cudaFree(d_out);
    }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "--- SCORECARD ---" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  Passed: " << passed << " / " << total << " tests." << std::endl;
    if (passed == total) {
        std::cout << "\033[1;32m  [STATUS] ALL " << total << " TESTS PASSED! \033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m  [STATUS] INCOMPLETE (" << (total - passed) << " tests failed) \033[0m" << std::endl;
    }
    std::cout << "=================================================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
