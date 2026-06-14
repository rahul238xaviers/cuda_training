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
    std::cout << "--- WORKBOOK: Pack, Wrap, Stencil & Crops (Problems 36-40) ---" << std::endl;
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

  // P40: 3D Crop Grid Volumetric Mapping
  {
    int d = 1, h = 2, w = 3;
    int d0 = 2, h0 = 3, w0 = 4;
    int H_parent = 10, W_parent = 10;
    int flat_parent_idx = -1;
    // TODO: Compute flat index inside parent grid starting at global offsets
    // (d0, h0, w0)

    bool ok = (flat_parent_idx == 357);
    reportStatus("Problem 40: 3D Crop parent mapping", ok);
    if (ok)
      passed++;
    total++;
  }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "--- SCORECARD: Pack, Wrap, Stencil & Crops ---" << std::endl;
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
