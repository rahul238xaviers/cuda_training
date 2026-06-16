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
    std::cout << "--- WORKBOOK: Row/Col-Major Indexing (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P21: 2D Row-Major Indexing
  {
    int r = 3, c = 4, W = 8;
    int flat_idx = -1;
    // TODO: Compute flat index for row r and col c in row-major layout of width
    // W

    bool ok = (flat_idx == 28);
    reportStatus("Problem 21: 2D Row-Major index mapping", ok);
    if (ok)
      passed++;
    total++;
  }

  // P22: 2D Row-Major Coordinate Extraction
  {
    int flat_idx = 28, W = 8;
    int r = -1, c = -1;
    // TODO: Extract r and c from flat_idx in row-major layout of width W

    bool ok = (r == 3 && c == 4);
    reportStatus("Problem 22: 2D Row-Major coordinate decoding", ok);
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
