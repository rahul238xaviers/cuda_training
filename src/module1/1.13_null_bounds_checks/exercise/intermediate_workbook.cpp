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
    std::cout << "--- WORKBOOK: Null, Bounds & Alignment Check (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P63: Safe pointer increment
  {
    int arr[3] = {1, 2, 3};
    int *ptr = &arr[2];
    int *end = arr + 3;
    bool increment_succeeded = false;
    // TODO: Attempt to increment ptr. Only do so if ptr + 1 < end. Set
    // increment_succeeded accordingly.

    bool ok = (!increment_succeeded && ptr == &arr[2]);
    reportStatus("Problem 63: Bound-checked pointer increment", ok);
    if (ok)
      passed++;
    total++;
  }

  // P64: Alignment Check (8-byte)
  {
    double val = 3.14;
    double *ptr = &val;
    bool is_aligned_8 = false;
    // TODO: Check if pointer address is aligned to 8 bytes (address % 8 == 0)

    bool ok = is_aligned_8;
    reportStatus("Problem 64: 8-byte pointer alignment check", ok);
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
