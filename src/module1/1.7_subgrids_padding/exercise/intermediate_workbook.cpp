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
    std::cout << "--- WORKBOOK: Subgrids, Padding & Diagonals (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P33: Boundary Padding / Halo Cells
  {
    int r = 4, c = 5, W = 10;
    int flat_padded_idx = -1;
    // TODO: If we pad a grid with a 1-cell border, compute the flat index of
    // (r, c) in the padded grid (width = W + 2)

    bool ok = (flat_padded_idx == 66);
    reportStatus("Problem 33: Halo cell padded index calculation", ok);
    if (ok)
      passed++;
    total++;
  }

  // P34: Strided Matrix (Leading Dimension)
  {
    int r = 2, c = 3, LD = 16;
    int flat_idx = -1;
    // TODO: Compute flat index for row r, column c, with leading dimension
    // (stride) LD

    bool ok = (flat_idx == 35);
    reportStatus("Problem 34: Leading-dimension strided index mapping", ok);
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
