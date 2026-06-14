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
    std::cout << "--- WORKBOOK: 3D/4D Permute & Flatten (Problems 26-30) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P26: 3D Row-Major Coordinate Extraction
  {
    int flat_idx = 308, H = 8, W = 16;
    int d = -1, h = -1, w = -1;
    // TODO: Extract d, h, w from flat_idx

    bool ok = (d == 2 && h == 3 && w == 4);
    reportStatus("Problem 26: 3D Row-Major coordinate decoding", ok);
    if (ok)
      passed++;
    total++;
  }

  // P27: 3D Column-Major Indexing
  {
    int d = 2, h = 3, w = 4;
    int D = 4, H = 8;
    int flat_idx = -1;
    // TODO: Compute flat index in 3D Column-Major order (D * H * w + D * h + d)

    bool ok = (flat_idx == 142);
    reportStatus("Problem 27: 3D Column-Major index mapping", ok);
    if (ok)
      passed++;
    total++;
  }

  // P28: Flat Transpose Index (2D)
  {
    int src_idx = 14; // Index inside 3x5 matrix
    int R_src = 3, C_src = 5;
    int dst_idx = -1; // Transposed index inside 5x3 matrix
    // TODO: Calculate dst_idx of the transposed matrix coordinate

    bool ok =
        (dst_idx ==
         14); // element (2, 4) in 3x5 goes to (4, 2) in 5x3 -> 4 * 3 + 2 = 14
    reportStatus("Problem 28: 2D Flat transpose index calculation", ok);
    if (ok)
      passed++;
    total++;
  }

  // P29: Permute 3D Index (D, H, W -> W, H, D)
  {
    int src_idx = 45; // 3D index in 3x4x5 grid (D=3, H=4, W=5)
    int D = 3, H = 4, W = 5;
    int dst_idx = -1; // Index in 5x4x3 grid (permuted)
    // TODO: Decode src_idx to (d, h, w), then compute flat index in WxHxD grid

    bool ok =
        (dst_idx ==
         45); // (2, 1, 0) -> (0, 1, 2) in 5x4x3 grid: 0*(4*3) + 1*3 + 2 = 5
    // Wait, let's trace: 45 in 3x4x5 -> 45 / (4*5) = 2 (d), remainder 5. 5 / 5
    // = 1 (h), remainder 0 (w). Coords are (2, 1, 0). Permuted coords: d' = w =
    // 0, h' = h = 1, w' = d = 2. Index in 5x4x3 grid: d'* (4*3) + h'*3 + w' =
    // 0*(12) + 1*3 + 2 = 5. Let's set the target check: Decode: d = 45 / 20 =
    // 2; remainder = 5. h = 5 / 5 = 1; w = 0. New coords (w, h, d) inside
    // 5x4x3: (0, 1, 2) -> 0 * 12 + 1 * 3 + 2 = 5. So dst_idx should be 5. Let's
    // fix the test check to assert 5.
    ok = (dst_idx == 5);
    reportStatus("Problem 29: 3D Permute index calculation", ok);
    if (ok)
      passed++;
    total++;
  }

  // P30: 4D Tensor Batch-Channel-Height-Width Flattening
  {
    int b = 1, c = 2, h = 3, w = 4;
    int C = 3, H = 8, W = 8;
    int flat_idx = -1;
    // TODO: Flatten (b, c, h, w) into 1D linear offset in BCHW order

    bool ok = (flat_idx == 348);
    reportStatus("Problem 30: 4D BCHW Tensor flattening", ok);
    if (ok)
      passed++;
    total++;
  }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "--- SCORECARD: 3D/4D Permute & Flatten ---" << std::endl;
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
