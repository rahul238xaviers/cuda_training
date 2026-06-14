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
    std::cout << "--- WORKBOOK: Strides & Indirection (Problems 6-10) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P6: Postfix increment on pointer
  {
    int arr[5] = {10, 20, 30, 40, 50};
    int *ptr = &arr[0];
    int val = 0;
    // TODO: Assign the dereferenced value of ptr to val, then increment ptr
    // using postfix (ptr++)

    bool ok = (ptr == &arr[1] && val == 10);
    reportStatus("Problem 6: Postfix pointer increment", ok);
    if (ok)
      passed++;
    total++;
  }

  // P7: Strided stepping (step = 2)
  {
    int arr[10] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
    int *ptr = arr;
    // TODO: Advance ptr by 4 elements using pointer arithmetic to get to
    // element 40

    bool ok = (*ptr == 40);
    reportStatus("Problem 7: Strided pointer advancement", ok);
    if (ok)
      passed++;
    total++;
  }

  // P8: Stepping backwards
  {
    int arr[5] = {1, 2, 3, 4, 5};
    int *ptr = &arr[4];
    // TODO: Decrement ptr by 3 elements

    bool ok = (ptr == &arr[1]);
    reportStatus("Problem 8: Pointer backward stepping", ok);
    if (ok)
      passed++;
    total++;
  }

  // P9: Modify value via pointer
  {
    int val = 42;
    int *ptr = &val;
    // TODO: Change the value of val to 100 using ptr

    bool ok = (val == 100);
    reportStatus("Problem 9: Write through pointer", ok);
    if (ok)
      passed++;
    total++;
  }

  // P10: Bracketless indexing
  {
    int arr[5] = {100, 200, 300, 400, 500};
    int *ptr = arr;
    int val = 0;
    // TODO: Retrieve the element at index 3 using offset syntax *(ptr + i)
    // without brackets []

    bool ok = (val == 400);
    reportStatus("Problem 10: Bracketless indexing", ok);
    if (ok)
      passed++;
    total++;
  }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "--- SCORECARD: Strides & Indirection ---" << std::endl;
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
