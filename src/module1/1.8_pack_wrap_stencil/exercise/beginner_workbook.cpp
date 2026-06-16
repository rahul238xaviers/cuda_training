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
    std::cout << "--- WORKBOOK: Pack, Wrap, Stencil & Crops (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P36: Upper Triangular Matrix Flattening
  {
    int r = 2, c = 3, N = 5;
    int flat_idx = -1;
    // TODO: Map (r, c) in the upper triangular part of NxN matrix to a packed
    // 1D index Order: (0,0), (0,1), (0,2)... (0,N-1), (1,1), (1,2)... (1,N-1),
    // (2,2), (2,3)...

    bool ok = (flat_idx == 10);
    reportStatus("Problem 36: Upper triangular packed mapping", ok);
    if (ok)
      passed++;
    total++;
  }

  // P37: Circular Buffer Wrapping
  {
    int idx = 17, buffer_size = 8;
    int wrapped_idx = -1;
    // TODO: Perform circular buffer pointer wrapping math on index

    bool ok = (wrapped_idx == 1);
    reportStatus("Problem 37: Circular buffer index wrapping", ok);
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
