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
    std::cout << "--- WORKBOOK: Array Reductions & Swaps (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P13: Find max via pointers
  {
    float arr[5] = {1.1f, 5.5f, 3.3f, 9.9f, 2.2f};
    float max_val = arr[0];
    // TODO: Find the max element using pointer stepping

    bool ok = (std::abs(max_val - 9.9f) < 1e-5);
    reportStatus("Problem 13: Find maximum element via pointers", ok);
    if (ok)
      passed++;
    total++;
  }

  // P14: Vector addition (1D offset pointer math)
  {
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float z[4] = {0};
    // TODO: Compute z[i] = x[i] + y[i] using pointer arithmetic

    bool ok = (z[0] == 11.0f && z[3] == 44.0f);
    reportStatus("Problem 14: Vector addition using pointer offsets", ok);
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
