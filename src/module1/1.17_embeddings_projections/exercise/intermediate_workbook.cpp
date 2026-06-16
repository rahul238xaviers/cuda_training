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
    std::cout << "--- WORKBOOK: Embeddings & Projections (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

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
