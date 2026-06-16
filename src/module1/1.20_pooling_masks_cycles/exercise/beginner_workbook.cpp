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
    std::cout << "--- WORKBOOK: Pooling, Masks & Cycles (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P96: Max Pooling Output Stride
  {
    // Pool index tracker. Input cell (in_h, in_w) pooled with window 2x2.
    // Output coordinate (out_h = in_h / 2, out_w = in_w / 2).
    // Map input coordinate (5, 6) to output coordinate.
    int in_h = 5, in_w = 6;
    int out_h = -1, out_w = -1;
    // TODO: Compute output coordinates

    bool ok = (out_h == 2 && out_w == 3);
    reportStatus("Problem 96: Max pooling output downsampling scale", ok);
    if (ok)
      passed++;
    total++;
  }

  // P97: Dropout Mask Element-wise Step
  {
    float x[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    bool mask[4] = {true, false, true, false};
    float scale = 2.0f; // Inverted dropout scale (1.0 / (1.0 - dropout_rate))
    // TODO: Apply dropout in-place: x[i] = mask[i] ? (x[i] * scale) : 0.0f

    bool ok = (x[0] == 20.0f && x[1] == 0.0f && x[2] == 60.0f && x[3] == 0.0f);
    reportStatus("Problem 97: Element-wise dropout mask application", ok);
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
