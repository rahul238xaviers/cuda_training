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
    std::cout << "--- WORKBOOK: Pack, Wrap, Stencil & Crops (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P38: Block-Cyclic Partitioning Local Coordinate
  {
    int r = 18, c = 22;
    int block_R = 8, block_C = 8;
    int local_r = -1, local_c = -1;
    // TODO: Find local block row/col index (r % block_R, c % block_C)

    bool ok = (local_r == 2 && local_c == 6);
    reportStatus("Problem 38: Block-cyclic local mapping", ok);
    if (ok)
      passed++;
    total++;
  }

  // P39: 2D Stencil Left Neighbor Offset
  {
    int r = 3, c = 4, W = 8;
    int left_neighbor_flat = -1;
    // TODO: Compute flat index of the cell directly left of (r, c) in row-major
    // width W

    bool ok = (left_neighbor_flat == 27);
    reportStatus("Problem 39: 2D Stencil neighbor lookup", ok);
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
