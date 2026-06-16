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
    std::cout << "--- WORKBOOK: Type Casts & Double Pointers (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P73: const int* vs int* const
  {
    int x = 10, y = 20;
    const int *ptr1 = &x;
    int *const ptr2 = &x;
    // TODO: Determine which statement is valid:
    // A) ptr1 = &y;  is valid, but *ptr1 = 5; is invalid
    // B) ptr2 = &y;  is valid, but *ptr2 = 5; is invalid
    // Assign choice = 'A' or 'B'
    char choice = ' ';

    bool ok = (choice == 'A');
    reportStatus("Problem 73: Const pointer categorization", ok);
    if (ok)
      passed++;
    total++;
  }

  // P74: Double pointer dereference
  {
    int x = 50;
    int *p = &x;
    int **pp = &p;
    int val = 0;
    // TODO: Read value of x using pp double pointer

    bool ok = (val == 50);
    reportStatus("Problem 74: Double pointer dereference", ok);
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
