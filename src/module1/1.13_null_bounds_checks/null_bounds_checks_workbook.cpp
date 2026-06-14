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
    std::cout << "--- WORKBOOK: Null, Bounds & Alignment Check (Problems 61-65) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P61: Null pointer check
  {
    int *ptr = nullptr;
    bool is_safe = true;
    // TODO: If ptr is null, set is_safe = false

    bool ok = (!is_safe);
    reportStatus("Problem 61: Safe null check verification", ok);
    if (ok)
      passed++;
    total++;
  }

  // P62: Out of bounds check
  {
    int size = 5;
    int index = 5;
    bool safe_to_access = true;
    // TODO: Set safe_to_access = false if index is out of bounds [0, size-1]

    bool ok = (!safe_to_access);
    reportStatus("Problem 62: Out-of-bounds pointer prevention", ok);
    if (ok)
      passed++;
    total++;
  }

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

  // P65: Alignment Check (64-byte)
  {
    alignas(64) int aligned_val = 100;
    int *ptr = &aligned_val;
    bool is_aligned_64 = false;
    // TODO: Check if ptr address is aligned to 64 bytes

    bool ok = is_aligned_64;
    reportStatus("Problem 65: 64-byte pointer alignment check", ok);
    if (ok)
      passed++;
    total++;
  }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "--- SCORECARD: Null, Bounds & Alignment Check ---" << std::endl;
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
