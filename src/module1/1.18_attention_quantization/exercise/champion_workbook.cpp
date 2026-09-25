#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <cstdint>

// =========================================================================
// CHAMPION WORKBOOK: Attention & Quantization
//
// Module: 1.18 - ML Tensor Memory Primitives
// Level:  Champion / High-Performance Systems Engineer
//
// These exercises simulate core systems-level algorithms powering 
// FlashAttention, FP8 Hopper/Blackwell matrix cores, and W8A16 inference.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.18_champion
//   ../../../output/1.18_champion
// =========================================================================

void reportStatus(const std::string& name, bool passed, double throughputGBs = -1.0) {
    std::cout << "  " << std::left << std::setw(60) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m";
        if (throughputGBs > 0.0) {
            std::cout << " (" << std::fixed << std::setprecision(2) << throughputGBs << " GB/s)";
        }
        std::cout << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: Attention & Quantization (Champion Masterclass) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Online Softmax Accumulation (FlashAttention-Style Tiled Reduction)
    //
    // Context: Standard Softmax requires materializing the entire S x S score matrix in SRAM.
    //          FlashAttention computes Softmax incrementally over tiles (blocks) of keys/values.
    //          When moving from chunk (j-1) to chunk (j):
    //            1. Find chunk local max: m_curr = max(S_chunk)
    //            2. Update running max:   m_new = max(m_prev, m_curr)
    //            3. Rescaling factor:     alpha = exp(m_prev - m_new)
    //            4. Update running sum:   d_new = d_prev * alpha + sum(exp(S_chunk - m_new))
    //            5. Rescale output vector:
    //               acc_out = acc_out * alpha + sum_k(exp(S_k - m_new) * V_k)
    //          Finally: out = acc_out / d_final.
    //
    // Task: Process a row of S=64 logits and values (DIM=16) in 4 chunks of CHUNK_SIZE=16.
    //       Implement the online softmax accumulator and verify against standard 3-pass softmax.
    // -------------------------------------------------------------------------
    {
        const int S = 64;
        const int DIM = 16;
        const int CHUNK_SIZE = 16;
        const int NUM_CHUNKS = S / CHUNK_SIZE;

        std::vector<float> logits(S);
        std::vector<float> values(S * DIM);
        for (int i = 0; i < S; ++i) {
            logits[i] = static_cast<float>((i % 13) * 0.8f - 4.0f);
            for (int d = 0; d < DIM; ++d) {
                values[i * DIM + d] = static_cast<float>((i + d) % 7 + 1.0f);
            }
        }

        std::vector<float> online_out(DIM, 0.0f);
        float m_prev = -1e30f;
        float d_prev = 0.0f;

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Iterate over each of the 4 chunks, update running m, d, and online_out.
        // After all chunks, divide online_out by final d_prev.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();

        // Ground-truth: standard full-sequence 3-pass softmax + matmul
        float full_max = *std::max_element(logits.begin(), logits.end());
        float full_denom = 0.0f;
        std::vector<float> exp_scores(S);
        for (int i = 0; i < S; ++i) {
            exp_scores[i] = std::exp(logits[i] - full_max);
            full_denom += exp_scores[i];
        }
        for (int i = 0; i < S; ++i) exp_scores[i] /= full_denom;

        std::vector<float> expected_out(DIM, 0.0f);
        for (int i = 0; i < S; ++i) {
            for (int d = 0; d < DIM; ++d) {
                expected_out[d] += exp_scores[i] * values[i * DIM + d];
            }
        }

        bool p1_passed = true;
        for (int d = 0; d < DIM; ++d) {
            if (std::abs(online_out[d] - expected_out[d]) > 1e-4f) {
                p1_passed = false;
                break;
            }
        }

        reportStatus("Problem 1: FlashAttention Online Softmax Accumulation", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: FP8 (E4M3) Symmetric Quantization Emulation
    //
    // Context: NVIDIA Hopper & Blackwell Tensor Cores support FP8 E4M3
    //          (1 sign bit, 4 exponent bits [bias=7], 3 mantissa bits).
    //          Max representable normal value = 448.0f.
    //          Given a float tensor and a maximum scaling factor:
    //            scale = max(|x|) / 448.0f
    //            scaled_val = x / scale
    //            quantized = clamp to [-448.0, 448.0], round to nearest 3 mantissa bits.
    //
    // Task: Implement FP8 E4M3 simulated quantization on vector `activations` of size N=32.
    //       For each float:
    //         1. Scale by `1.0f / scale`
    //         2. Clamp to [-448.0f, 448.0f]
    //         3. Truncate/round to 3 bits of fraction:
    //            Extract float exponent using frexpf, round fraction to 8 levels (steps of 1/8).
    //         4. Multiply back by `scale` to obtain `fp8_reconstructed`.
    // -------------------------------------------------------------------------
    {
        const int N = 32;
        std::vector<float> activations = {
            12.5f, -8.25f, 440.0f, -500.0f, 0.125f, -0.0625f, 120.0f, -240.0f,
            1.0f, 2.0f, 3.5f, 7.25f, 15.125f, -31.0f, 63.0f, -127.0f,
            0.5f, -0.25f, 16.0f, -32.0f, 64.0f, -128.0f, 256.0f, -448.0f,
            0.0f, -1.0f, 4.0f, -8.0f, 16.0f, -32.0f, 64.0f, -128.0f
        };

        const float MAX_FP8 = 448.0f;
        float max_abs = 0.0f;
        for (float v : activations) max_abs = std::max(max_abs, std::abs(v));
        float scale = max_abs / MAX_FP8;

        std::vector<float> fp8_reconstructed(N, 0.0f);

        // TODO: Quantize each element to simulated FP8 E4M3 and reconstruct.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = true;
        for (int i = 0; i < N && p2_passed; ++i) {
            float orig = activations[i];
            float recon = fp8_reconstructed[i];
            // Check that value is bounded by clamping
            if (std::abs(recon) > max_abs + 1e-3f) p2_passed = false;
            // FP8 has 3 mantissa bits (~12.5% max quantization step relative error)
            float rel_err = std::abs(recon - orig) / (std::abs(orig) + 1e-4f);
            if (std::abs(orig) <= MAX_FP8 * scale && rel_err > 0.15f) p2_passed = false;
        }

        reportStatus("Problem 2: FP8 (E4M3) Symmetric Quantization Emulation", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Fused INT8 Dequantize-GEMV Memory Traversal (W8A16)
    //
    // Context: In low-latency LLM generation (batch size 1), memory bandwidth is the
    //          primary bottleneck. Weights are stored in INT8 with a per-channel scale,
    //          and dequantized on the fly during vector dot products:
    //            y[i] = scale[i] * sum_k (W_int8[i, k] * x[k])
    //
    // Task: Given weight matrix W [M=512, K=1024] (524,288 int8_t elements),
    //       scale vector `scales` [M=512], and activation vector `x` [K=1024]:
    //       Compute output vector `y` [M=512].
    //       Benchmark the memory throughput!
    // -------------------------------------------------------------------------
    {
        const int M = 512;
        const int K = 1024;
        std::vector<int8_t> W(M * K);
        for (size_t i = 0; i < W.size(); ++i) {
            W[i] = static_cast<int8_t>((i % 251) - 125);
        }
        std::vector<float> scales(M);
        for (int i = 0; i < M; ++i) scales[i] = 0.005f * (i % 10 + 1);

        std::vector<float> x(K);
        for (int i = 0; i < K; ++i) x[i] = static_cast<float>((i % 17) - 8) * 0.1f;

        std::vector<float> y(M, 0.0f);

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Compute fused dequantized matrix-vector multiplication y = scales * (W @ x).
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double bytes_moved = (size_t)M * K * sizeof(int8_t) + K * sizeof(float) + M * sizeof(float);
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        bool p3_passed = true;
        for (int i = 0; i < M && p3_passed; ++i) {
            int32_t dot = 0;
            for (int k = 0; k < K; ++k) {
                dot += (int32_t)W[i * K + k] * (int32_t)(x[k] * 10.0f + (x[k] >= 0 ? 0.5f : -0.5f));
            }
            // Continuous ground truth check
            float float_dot = 0.0f;
            for (int k = 0; k < K; ++k) {
                float_dot += static_cast<float>(W[i * K + k]) * x[k];
            }
            float expected = scales[i] * float_dot;
            if (std::abs(y[i] - expected) > 1e-2f) {
                p3_passed = false;
                break;
            }
        }

        reportStatus("Problem 3: Fused INT8 Dequantize-GEMV Memory Traversal", p3_passed, p3_passed ? throughput : -1.0);
        if (p3_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 4: Grouped-Query Attention (GQA) Head Broadcast Stride Resolution
    //
    // Context: In Llama-3, there are 32 Query heads, but only 8 Key/Value heads.
    //          Each Key/Value head is shared across GROUP_SIZE = 32 / 8 = 4 query heads.
    //          When performing attention for query head `h_q`, the corresponding
    //          key head is `h_kv = h_q / GROUP_SIZE`.
    //
    // Task: Given packed Key buffer `K_cache` of shape [BATCH=2, KV_HEADS=8, SEQ_LEN=32, HEAD_DIM=16]:
    //       For any query head `q_head` in [0, Q_HEADS=32), resolve the exact memory pointer
    //       to the Key vector at batch `target_b = 1` and sequence step `target_s = 14`.
    //       Verify that all 4 heads within the same group map to the EXACT same memory address!
    // -------------------------------------------------------------------------
    {
        const int BATCH = 2, Q_HEADS = 32, KV_HEADS = 8, SEQ_LEN = 32, HEAD_DIM = 16;
        const int GROUP_SIZE = Q_HEADS / KV_HEADS; // 4

        std::vector<float> K_cache(BATCH * KV_HEADS * SEQ_LEN * HEAD_DIM);
        for (size_t i = 0; i < K_cache.size(); ++i) {
            K_cache[i] = static_cast<float>(i + 1);
        }

        const int target_b = 1;
        const int target_s = 14;

        bool p4_passed = true;

        for (int q_head = 0; q_head < Q_HEADS; ++q_head) {
            const float* resolved_k_ptr = nullptr;

            // TODO: Compute h_kv and point resolved_k_ptr to the start of the 16-element Key vector
            // at (target_b, h_kv, target_s) inside K_cache.
            // --- YOUR CODE STARTS HERE ---

            // --- YOUR CODE ENDS HERE ---

            int expected_h_kv = q_head / GROUP_SIZE;
            size_t expected_offset = (size_t)target_b * (KV_HEADS * SEQ_LEN * HEAD_DIM) +
                                     (size_t)expected_h_kv * (SEQ_LEN * HEAD_DIM) +
                                     (size_t)target_s * HEAD_DIM;
            if (resolved_k_ptr != &K_cache[expected_offset]) {
                p4_passed = false;
                break;
            }
        }

        reportStatus("Problem 4: Grouped-Query Attention (GQA) Head Broadcast", p4_passed);
        if (p4_passed) passed++;
        total++;
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
