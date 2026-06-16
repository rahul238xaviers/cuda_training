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
    std::cout << "--- WORKBOOK: Subgrids, Padding & Diagonals (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P31: 4D Tensor Coordinate Decoding
  {
    int flat_idx = 348;
    int C = 3, H = 8, W = 8;
    int b = -1, c = -1, h = -1, w = -1;
    // TODO: Decode flat_idx back to (b, c, h, w)

    bool ok = (b == 1 && c == 2 && h == 3 && w == 4);
    reportStatus("Problem 31: 4D BCHW Tensor coordinate decoding", ok);
    if (ok)
      passed++;
    total++;
  }

  // P32: Subgrid Slicing Offset
  {
    int r = 2, c = 3;
    int r_offset = 1, c_offset = 2;
    int parent_W = 10;
    int flat_idx = -1;
    // TODO: Compute flat index of local crop coord (r, c) inside the parent
    // grid

    bool ok = (flat_idx == 35);
    reportStatus("Problem 32: Subgrid slicing parent index calculation", ok);
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
