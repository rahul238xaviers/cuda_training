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
    std::cout << "--- WORKBOOK: Pointer Relations & Copying (Problems 76-80) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P76: Pointer comparison in array
  {
    int arr[5] = {10, 20, 30, 40, 50};
    int *ptr_a = &arr[1];
    int *ptr_b = &arr[4];
    bool is_a_before_b = false;
    // TODO: Compare ptr_a and ptr_b (set is_a_before_b = true if ptr_a is
    // before ptr_b)

    bool ok = is_a_before_b;
    reportStatus("Problem 76: Pointer comparison logic", ok);
    if (ok)
      passed++;
    total++;
  }

  // P77: Safe memcpy overlap detection
  {
    char buffer[10] = "abcdefghi";
    char *src = buffer;
    char *dst = buffer + 2;
    bool overlap = false;
    // TODO: Detect if copying 5 bytes from src to dst overlaps
    // Set overlap = true if (src < dst && src + 5 > dst)
    if (src < dst && src + 5 > dst)
      overlap = true;

    bool ok = overlap;
    reportStatus("Problem 77: Overlapping memory detection", ok);
    if (ok)
      passed++;
    total++;
  }

  // P78: Dangling pointer prevention
  {
    int *ptr = new int(10);
    delete ptr;
    // TODO: Set ptr to nullptr to prevent it from remaining dangling

    bool ok = (ptr == nullptr);
    reportStatus("Problem 78: Dangling pointer sanitization", ok);
    if (ok)
      passed++;
    total++;
  }

  // P79: Memory copy safe helper
  {
    char src[5] = "GPU";
    char dst[5] = {0};
    // TODO: Copy 4 bytes of src to dst using std::memcpy

    bool ok = (std::strcmp(dst, "GPU") == 0);
    reportStatus("Problem 79: Standard memcpy utility usage", ok);
    if (ok)
      passed++;
    total++;
  }

  // P80: Alignment offset helper
  {
    uintptr_t addr = 1003;
    uintptr_t aligned_addr = 0;
    // TODO: Round addr up to next multiple of 8 using bitwise operations

    bool ok = (aligned_addr == 1008);
    reportStatus("Problem 80: Pointer alignment bitwise rounding", ok);
    if (ok)
      passed++;
    total++;
  }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "--- SCORECARD: Pointer Relations & Copying ---" << std::endl;
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
