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
    std::cout << "--- WORKBOOK: Multi-Dim & Alignment (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P46: 2D Dynamic Array Allocation (Array of pointers)
  {
    int **arr = nullptr;
    int H = 3, W = 4;
    // TODO: Allocate memory for a 3x4 2D array using double pointer syntax

    bool ok = (arr != nullptr && arr[0] != nullptr && arr[2] != nullptr);
    if (arr) {
      for (int i = 0; i < H; ++i)
        delete[] arr[i];
      delete[] arr;
    }
    reportStatus("Problem 46: 2D pointer-of-pointers allocation", ok);
    if (ok)
      passed++;
    total++;
  }

  // P47: Flat 2D Array Allocation
  {
    int *arr = nullptr;
    int H = 3, W = 4;
    // TODO: Allocate memory for flat 2D array of size H*W

    bool ok = (arr != nullptr);
    delete[] arr;
    reportStatus("Problem 47: Flat 2D array allocation", ok);
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
