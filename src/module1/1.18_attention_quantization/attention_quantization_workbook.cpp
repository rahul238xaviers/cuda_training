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
    std::cout << "--- WORKBOOK: Attention & Quantization (Problems 86-90) ---" << std::endl;
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

  // P88: Quantized weights (INT8 to FP32 reconstruct)
  {
    int8_t weight = -50;
    float scale = 0.05f;
    float bias = 0.1f;
    float dequantized = 0.0f;
    // TODO: Reconstruction calculation: dequantized = weight * scale + bias

    bool ok = (std::abs(dequantized - (-2.4f)) < 1e-5);
    reportStatus("Problem 88: Weight dequantization arithmetic", ok);
    if (ok)
      passed++;
    total++;
  }

  // P89: In-place ReLU via pointers
  {
    float x[5] = {-2.0f, 1.5f, -0.5f, 3.0f, 0.0f};
    // TODO: Apply ReLU (x = max(0, x)) in-place using pointers

    bool ok = (x[0] == 0.0f && x[1] == 1.5f && x[2] == 0.0f && x[3] == 3.0f);
    reportStatus("Problem 89: In-place ReLU pointer step operation", ok);
    if (ok)
      passed++;
    total++;
  }

  // P90: Softmax Max-Subtraction Helper
  {
    float logits[3] = {1.0f, 4.0f, 2.0f};
    float max_logit = logits[0];
    // TODO: Find max_logit using pointer stepping
    // Then subtract max_logit from all logits in-place using pointer stepping
    // (to prevent overflow)

    bool ok = (logits[0] == -3.0f && logits[1] == 0.0f && logits[2] == -2.0f);
    reportStatus("Problem 90: Softmax max-subtraction offset adjustment", ok);
    if (ok)
      passed++;
    total++;
  }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "--- SCORECARD: Attention & Quantization ---" << std::endl;
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
