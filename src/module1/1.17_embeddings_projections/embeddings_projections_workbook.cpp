#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

// Global status reporter
void reportStatus(const std::string& name, bool passed) {
    std::cout << "  " << std::left << std::setw(65) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

// Structs used across exercises
struct Point2D { int r; int c; };
struct Point3D { int d; int h; int w; };
struct Tensor4D { int b; int c; int h; int w; };

struct alignas(16) AlignedStruct {
    float x;
    float y;
    float z;
    float w;
};

struct UnalignedStruct {
    char a;
    double b;
    int c;
};

struct CustomMLBatch {
    int batch_id;
    float* data;
};


int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: Embeddings & Projections (Problems 81-85) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P81: Embedding Lookup
  {
    float embeddings[4 * 8] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
                               1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f,
                               2.0f, 2.1f, 2.2f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f,
                               3.0f, 3.1f, 3.2f, 3.3f, 3.4f, 3.5f, 3.6f, 3.7f};
    int token_id = 2;
    int embedding_dim = 8;
    float *lookup_ptr = nullptr;
    // TODO: Get pointer to the embedding vector of token_id

    bool ok = (lookup_ptr == &embeddings[16] && lookup_ptr[1] == 2.1f);
    reportStatus("Problem 81: Embedding table lookup offset", ok);
    if (ok)
      passed++;
    total++;
  }

  // P82: Batched Embedding Lookup
  {
    float embeddings[4 * 4] = {0.0f, 0.1f, 0.2f, 0.3f, 1.0f, 1.1f, 1.2f, 1.3f,
                               2.0f, 2.1f, 2.2f, 2.3f, 3.0f, 3.1f, 3.2f, 3.3f};
    int batch_tokens[2] = {3, 1};
    float *batch_ptrs[2] = {nullptr, nullptr};
    // TODO: Store pointer for each token in batch_tokens into batch_ptrs

    bool ok =
        (batch_ptrs[0] == &embeddings[12] && batch_ptrs[1] == &embeddings[4]);
    reportStatus("Problem 82: Batched embedding row resolution", ok);
    if (ok)
      passed++;
    total++;
  }

  // P83: Key-Value Cache Offset
  {
    // KV cache tensor shape: (Batch B, Sequence_Length S, Num_Heads H, Head_Dim
    // D) Dimensions: B=2, S=32, H=4, D=64 Flat index for (b=1, s=15, h=2, d=0)
    int b = 1, s = 15, h = 2, d = 0;
    int B = 2, S = 32, H = 4, D = 64;
    int offset = -1;
    // TODO: Compute offset in KV cache tensor

    bool ok = (offset == b * (S * H * D) + s * (H * D) + h * D + d);
    reportStatus("Problem 83: KV Cache tensor flat offset calculation", ok);
    if (ok)
      passed++;
    total++;
  }

  // P84: Attention Q/K/V Head Pointer Offset
  {
    // Combined QKV projection output shape: (Batch B, SeqLen S, 3, NumHeads H,
    // HeadDim D) Get pointer to Key vector (projection index = 1) for batch
    // b=0, seq s=5, head h=3
    int B = 2, S = 10, H = 8, D = 64;
    float qkv_tensor[2 * 10 * 3 * 8 * 64];
    float *k_head_ptr = nullptr;
    // TODO: Point to the start of Key vector for batch 0, sequence step 5, head
    // 3

    bool ok = (k_head_ptr == &qkv_tensor[0 * (S * 3 * H * D) + 5 * (3 * H * D) +
                                         1 * (H * D) + 3 * D]);
    reportStatus("Problem 84: QKV tensor sub-projection head address", ok);
    if (ok)
      passed++;
    total++;
  }

  // P85: Transpose Head Offset (B, S, H, D -> B, H, S, D)
  {
    // Transpose elements to make heads contiguous for attention dot product
    int b = 1, h = 2, s = 3, d = 4;
    int B = 2, H = 4, S = 8, D = 16;
    int dst_offset = -1;
    // TODO: Compute destination index inside transposed tensor with shape (B,
    // H, S, D)

    bool ok = (dst_offset == b * (H * S * D) + h * (S * D) + s * D + d);
    reportStatus("Problem 85: Transposed Attention Head layout index", ok);
    if (ok)
      passed++;
    total++;
  }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "--- SCORECARD: Embeddings & Projections ---" << std::endl;
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
