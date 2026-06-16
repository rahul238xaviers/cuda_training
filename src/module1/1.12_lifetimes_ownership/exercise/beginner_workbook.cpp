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
    std::cout << "--- WORKBOOK: Lifetimes, Ownership & Resize (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P56: Deep copy dynamic structs
  {
    CustomMLBatch src;
    src.batch_id = 9;
    src.data = new float[3]{1.0f, 2.0f, 3.0f};
    CustomMLBatch dst{0, nullptr};
    // TODO: Perform deep copy of src to dst (allocating new memory for
    // dst.data)

    bool ok = (dst.batch_id == 9 && dst.data != nullptr &&
               dst.data != src.data && dst.data[1] == 2.0f);
    delete[] src.data;
    delete[] dst.data;
    reportStatus("Problem 56: Struct deep copy validation", ok);
    if (ok)
      passed++;
    total++;
  }

  // P57: Shared buffer ownership tracker
  {
    int *ref_count = new int(1);
    int *data = new int[5]{1, 2, 3, 4, 5};
    // TODO: Simulate adding a new reference. Increment ref_count.

    bool ok = (*ref_count == 2);
    delete ref_count;
    delete[] data;
    reportStatus("Problem 57: Manual reference counting", ok);
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
