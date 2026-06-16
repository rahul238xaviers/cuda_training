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
    std::cout << "--- WORKBOOK: Arenas & Placements (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P53: Placement new
  {
    char buffer[sizeof(Point2D)];
    Point2D *ptr = nullptr;
    // TODO: Use placement new to construct Point2D inside buffer, setting r=5,
    // c=15

    bool ok = (ptr == (Point2D *)buffer && ptr->r == 5 && ptr->c == 15);
    reportStatus("Problem 53: Placement new construction", ok);
    if (ok)
      passed++;
    total++;
  }

  // P54: Custom Pool Allocator
  {
    char pool[1024];
    char *pool_ptr = pool;
    void *alloc1 = nullptr;
    // TODO: Allocate 128 bytes from pool by advancing pool_ptr. Assign
    // allocated memory to alloc1.

    bool ok = (alloc1 == pool && pool_ptr == pool + 128);
    reportStatus("Problem 54: Custom arena pool allocation", ok);
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
