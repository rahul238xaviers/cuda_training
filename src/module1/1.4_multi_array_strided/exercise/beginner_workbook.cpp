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
    std::cout << "--- WORKBOOK: Multi-array & Strided Ops (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P16: Interleave two arrays using pointers
  {
    int a[3] = {1, 3, 5};
    int b[3] = {2, 4, 6};
    int c[6] = {0}; // Should be {1, 2, 3, 4, 5, 6}
    // TODO: Interleave a and b into c using pointers to scan each array

    bool ok = (c[0] == 1 && c[1] == 2 && c[4] == 5 && c[5] == 6);
    reportStatus("Problem 16: Interleave arrays via pointers", ok);
    if (ok)
      passed++;
    total++;
  }

  // P17: Reverse array in-place via two pointers
  {
    int arr[5] = {1, 2, 3, 4, 5};
    // TODO: Reverse arr in-place using two pointers (start pointing to head,
    // end pointing to tail)

    bool ok = (arr[0] == 5 && arr[4] == 1 && arr[2] == 3);
    reportStatus("Problem 17: In-place array reversal using two pointers", ok);
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
