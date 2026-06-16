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
    std::cout << "--- WORKBOOK: Multi-Dim & Alignment (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P48: Correct deallocation of pointer-of-pointers
  {
    int **arr = new int *[3];
    for (int i = 0; i < 3; ++i)
      arr[i] = new int[4];
    bool deallocated = false;
    // TODO: Deallocate arr correctly and set deallocated to true

    bool ok = deallocated;
    reportStatus("Problem 48: 2D pointer-of-pointers deallocation", ok);
    if (ok)
      passed++;
    total++;
  }

  // P49: Custom Pointer Alignment (aligned_alloc)
  {
    void *ptr = nullptr;
    size_t alignment = 64;
    size_t size = 256;
    // TODO: Allocate size bytes aligned to alignment using std::aligned_alloc
    // (or posix_memalign)

    bool ok = (ptr != nullptr && ((uintptr_t)ptr % alignment == 0));
    if (ptr)
      std::free(ptr);
    reportStatus("Problem 49: Aligned heap allocation", ok);
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
