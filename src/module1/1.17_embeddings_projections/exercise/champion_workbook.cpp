#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <random>
#include <cstring>
#include <cstdint>

// =========================================================================
// CHAMPION WORKBOOK: Embeddings, Projections & Multi-Head Attention Layouts
// 
// Module: 1.17 - ML Tensor Memory Primitives
// Level:  Champion / High-Performance Systems Engineer
//
// These exercises simulate real low-level tensor operations found in 
// modern LLM inference engines (e.g., vLLM, TensorRT-LLM, FlashAttention)
// and custom CUDA kernel memory prep.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.17_champion
//   ../../../output/1.17_champion
// =========================================================================

void reportStatus(const std::string& name, bool passed, double throughputGBs = -1.0) {
    std::cout << "  " << std::left << std::setw(58) << name;
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
    std::cout << "--- WORKBOOK: Embeddings & Projections (Champion Masterclass) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: 4D Batched Multi-Head Layout Permutation ([B, S, H, D] -> [B, H, S, D])
    //
    // Context: Standard Linear projection produces activations shaped [B, S, H * D].
    //          Before computing Scaled Dot-Product Attention (Q @ K^T), attention heads
    //          must be contiguous along sequence length: [B, H, S, D].
    //          In CUDA, this requires coalesced reads and conflict-free strided writes.
    //
    // Task: Implement the in-memory layout permutation from `src` to `dst`.
    //       Given shapes: B=4, S=128, H=16, D=64 (Total = 524,288 floats = 2 MB).
    //       You must calculate the source and destination offsets for every element
    //       and copy the data correctly.
    // -------------------------------------------------------------------------
    {
        const int B = 4, S = 128, H = 16, D = 64;
        const size_t total_elements = B * S * H * D;
        std::vector<float> src(total_elements);
        std::vector<float> dst(total_elements, -1.0f);

        // Fill source with recognizable pattern
        for (size_t i = 0; i < total_elements; ++i) {
            src[i] = static_cast<float>(i * 0.001f);
        }

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Transform elements from src [B, S, H, D] into dst [B, H, S, D].
        // Compute strides for both layouts and permute all elements.
        // --- YOUR CODE STARTS HERE ---
        
        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double bytes_moved = 2.0 * total_elements * sizeof(float); // read + write
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        // Ground-truth verification across all elements
        bool p1_passed = true;
        for (int b = 0; b < B && p1_passed; ++b) {
            for (int h = 0; h < H && p1_passed; ++h) {
                for (int s = 0; s < S && p1_passed; ++s) {
                    for (int d = 0; d < D; ++d) {
                        size_t src_idx = (size_t)b * (S * H * D) + (size_t)s * (H * D) + (size_t)h * D + d;
                        size_t dst_idx = (size_t)b * (H * S * D) + (size_t)h * (S * D) + (size_t)s * D + d;
                        if (std::abs(dst[dst_idx] - src[src_idx]) > 1e-5f) {
                            p1_passed = false;
                            break;
                        }
                    }
                }
            }
        }

        reportStatus("Problem 1: 4D Multi-Head Tensor Permutation [B,S,H,D]->[B,H,S,D]", p1_passed, p1_passed ? throughput : -1.0);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Fused QKV Projection Slicing & Strided Extraction
    //
    // Context: In production transformer forward passes, a single fused GEMM
    //          computes Query, Key, and Value projections simultaneously into
    //          a packed buffer of shape [B, S, 3, H, D].
    //
    // Task: Given packed buffer `qkv_packed` of shape [B=2, S=64, 3, H=8, D=64],
    //       extract the Key tensor into a contiguous buffer `k_contiguous` 
    //       of shape [B, H, S, D].
    //       Projection indices: Q = 0, K = 1, V = 2.
    // -------------------------------------------------------------------------
    {
        const int B = 2, S = 64, NUM_PROJ = 3, H = 8, D = 64;
        const size_t packed_size = B * S * NUM_PROJ * H * D;
        const size_t k_size = B * H * S * D;

        std::vector<float> qkv_packed(packed_size);
        std::vector<float> k_contiguous(k_size, -999.0f);

        for (size_t i = 0; i < packed_size; ++i) {
            qkv_packed[i] = static_cast<float>(i + 1);
        }

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Extract ONLY the Key projection (proj = 1) from qkv_packed [B, S, 3, H, D]
        // into k_contiguous [B, H, S, D].
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double bytes_moved = (k_size + k_size) * sizeof(float);
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        bool p2_passed = true;
        for (int b = 0; b < B && p2_passed; ++b) {
            for (int h = 0; h < H && p2_passed; ++h) {
                for (int s = 0; s < S && p2_passed; ++s) {
                    for (int d = 0; d < D; ++d) {
                        size_t packed_idx = (size_t)b * (S * NUM_PROJ * H * D) +
                                            (size_t)s * (NUM_PROJ * H * D) +
                                            1 * (H * D) + // Key projection
                                            (size_t)h * D + d;
                        size_t k_idx = (size_t)b * (H * S * D) + (size_t)h * (S * D) + (size_t)s * D + d;
                        if (std::abs(k_contiguous[k_idx] - qkv_packed[packed_idx]) > 1e-5f) {
                            p2_passed = false;
                            break;
                        }
                    }
                }
            }
        }

        reportStatus("Problem 2: Fused QKV Projection Slicing & Strided Extraction", p2_passed, p2_passed ? throughput : -1.0);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Vectorized Ragged Batched Embedding Gather with Out-of-Bounds Clamping
    //
    // Context: Embedding tables [VocabSize, HiddenDim] are gathered per token.
    //          In real production serving, inputs are ragged batches of tokens.
    //          Tokens might contain padding (PAD_ID = 0) or corrupted IDs >= VocabSize.
    //
    // Task: Given an embedding table [VocabSize=1000, HiddenDim=128],
    //       gather embedding rows for a batch of token IDs.
    //       - If token_id == 0 (PAD): output vector must be all zeros.
    //       - If token_id >= VocabSize: clamp token_id to (VocabSize - 1) before lookup.
    //       - Copy full 128-float embedding vectors into `out_embeddings`.
    // -------------------------------------------------------------------------
    {
        const int VOCAB_SIZE = 1000;
        const int HIDDEN_DIM = 128;
        const int BATCH_TOKENS = 64;

        std::vector<float> embed_table(VOCAB_SIZE * HIDDEN_DIM);
        for (size_t i = 0; i < embed_table.size(); ++i) {
            embed_table[i] = static_cast<float>(i % 37 + 1.0f);
        }

        // Test token batch including valid IDs, PAD (0), and out-of-bounds IDs
        std::vector<int> tokens(BATCH_TOKENS);
        for (int i = 0; i < BATCH_TOKENS; ++i) {
            if (i % 8 == 0) tokens[i] = 0;            // PAD
            else if (i % 11 == 0) tokens[i] = 1500;   // Out-of-bounds -> clamp to 999
            else tokens[i] = (i * 31) % VOCAB_SIZE;  // Valid
        }

        std::vector<float> out_embeddings(BATCH_TOKENS * HIDDEN_DIM, -1.0f);

        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Perform the ragged embedding gather with padding and clamping.
        // Output shape: [BATCH_TOKENS, HIDDEN_DIM].
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double bytes_moved = BATCH_TOKENS * HIDDEN_DIM * sizeof(float);
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        bool p3_passed = true;
        for (int i = 0; i < BATCH_TOKENS && p3_passed; ++i) {
            int tok = tokens[i];
            float* out_vec = &out_embeddings[i * HIDDEN_DIM];

            if (tok == 0) {
                for (int d = 0; d < HIDDEN_DIM; ++d) {
                    if (out_vec[d] != 0.0f) { p3_passed = false; break; }
                }
            } else {
                int effective_tok = std::min(tok, VOCAB_SIZE - 1);
                const float* expected_vec = &embed_table[effective_tok * HIDDEN_DIM];
                for (int d = 0; d < HIDDEN_DIM; ++d) {
                    if (std::abs(out_vec[d] - expected_vec[d]) > 1e-5f) {
                        p3_passed = false;
                        break;
                    }
                }
            }
        }

        reportStatus("Problem 3: Ragged Batched Embedding Gather with Clamping", p3_passed, p3_passed ? throughput : -1.0);
        if (p3_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 4: Paged KV-Cache Block Table Indexing (vLLM / PagedAttention Primitives)
    //
    // Context: Modern LLM serving systems cannot store continuous KV-caches due to
    //          memory fragmentation. They allocate physical "blocks" of tokens
    //          (e.g., BLOCK_SIZE = 16 tokens) from a shared memory pool.
    //          A "block table" maps logical token sequence positions to physical block IDs.
    //
    // Geometry:
    //   Physical KV Cache Pool: [NUM_BLOCKS, 2 (K/V), NUM_HEADS, BLOCK_SIZE, HEAD_DIM]
    //   BLOCK_SIZE = 16, NUM_HEADS = 8, HEAD_DIM = 64
    //
    // Task: Given a sequence's `block_table` and a logical token query at `logical_seq_idx`:
    //       1. Compute the block index in the table: `block_table_idx = logical_seq_idx / BLOCK_SIZE`
    //       2. Retrieve `physical_block_id = block_table[block_table_idx]`
    //       3. Compute token offset within block: `block_offset = logical_seq_idx % BLOCK_SIZE`
    //       4. Retrieve the pointer to the Key vector of head `target_head`:
    //          `const float* k_ptr`
    //       5. Copy `HEAD_DIM` floats into `out_k_vector`.
    // -------------------------------------------------------------------------
    {
        const int NUM_BLOCKS = 64;
        const int NUM_HEADS = 8;
        const int BLOCK_SIZE = 16;
        const int HEAD_DIM = 64;

        // Total pool size: 64 * 2 * 8 * 16 * 64 = 1,048,576 floats (4 MB)
        std::vector<float> kv_pool(NUM_BLOCKS * 2 * NUM_HEADS * BLOCK_SIZE * HEAD_DIM);
        for (size_t i = 0; i < kv_pool.size(); ++i) {
            kv_pool[i] = static_cast<float>(i * 0.1f);
        }

        // Logical sequence has 5 blocks allocated across non-contiguous physical blocks:
        std::vector<int> block_table = {42, 7, 19, 3, 58};

        int query_logical_seq_idx = 45; // Falls in block 2 (idx 45 / 16 = 2 -> physical block 19)
        int target_head = 5;

        std::vector<float> out_k_vector(HEAD_DIM, 0.0f);

        // TODO: Resolve the physical pointer and copy the Key vector for target_head into out_k_vector.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        // Expected calculation
        int exp_table_idx = query_logical_seq_idx / BLOCK_SIZE; // 2
        int exp_phys_block = block_table[exp_table_idx];         // 19
        int exp_block_offset = query_logical_seq_idx % BLOCK_SIZE; // 13

        // Pool layout: [block_id][is_value (0=K, 1=V)][head_id][token_in_block][dim]
        size_t expected_offset = (size_t)exp_phys_block * (2 * NUM_HEADS * BLOCK_SIZE * HEAD_DIM) +
                                 0 * (NUM_HEADS * BLOCK_SIZE * HEAD_DIM) + // 0 = Key
                                 (size_t)target_head * (BLOCK_SIZE * HEAD_DIM) +
                                 (size_t)exp_block_offset * HEAD_DIM;

        bool p4_passed = true;
        for (int d = 0; d < HEAD_DIM; ++d) {
            if (std::abs(out_k_vector[d] - kv_pool[expected_offset + d]) > 1e-5f) {
                p4_passed = false;
                break;
            }
        }

        reportStatus("Problem 4: PagedAttention Block Table Physical KV Resolution", p4_passed);
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
