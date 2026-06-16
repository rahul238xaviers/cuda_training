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
    std::cout << "--- WORKBOOK: Attention & Quantization (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P86: Attention Score Query-Key Similarity Index
  {
    // Score matrix shape: (Batch B, Heads H, SeqLen_Q Sq, SeqLen_K Sk)
    // Dimensions: B=2, H=8, Sq=16, Sk=16
    // Flat index for b=1, h=4, q_idx=3, k_idx=7
    int B = 2, H = 8, Sq = 16, Sk = 16;
    int b = 1, h = 4, q_idx = 3, k_idx = 7;
    int flat_idx = -1;
    // TODO: Compute flat offset for attention score of Q-K pair

    bool ok =
        (flat_idx == b * (H * Sq * Sk) + h * (Sq * Sk) + q_idx * Sk + k_idx);
    reportStatus("Problem 86: Attention score index mapping", ok);
    if (ok)
      passed++;
    total++;
  }

  // P87: Sliding Window Attention Bound Check
  {
    int q_idx = 15;
    int k_idx = 8;
    int window_size = 4;
    bool in_window = true;
    // TODO: Set in_window = false if k_idx is outside the sliding window [q_idx
    // - window_size, q_idx]

    bool ok = (!in_window);
    reportStatus("Problem 87: Sliding window boundary validation", ok);
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
