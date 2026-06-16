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
    std::cout << "--- WORKBOOK: Attention & Quantization (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

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
