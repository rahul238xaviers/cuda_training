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
    std::cout << "--- WORKBOOK: Type Casts & Double Pointers (Problems 71-75) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P71: Union pointer check
  {
    union FloatInt {
      float f;
      int i;
    } val;
    val.f = 2.0f;
    int read_i = 0;
    // TODO: Read the bit representation using the union's integer field

    bool ok = (read_i == 0x40000000);
    reportStatus("Problem 71: Union representation check", ok);
    if (ok)
      passed++;
    total++;
  }

  // P72: Array decay verification
  {
    int arr[5] = {1, 2, 3, 4, 5};
    size_t decayed_size = 0;
    auto helper = [](int *p) { return sizeof(p); };
    decayed_size = helper(arr);
    // TODO: Verify decayed_size matches sizeof(int*)

    bool ok = (decayed_size == sizeof(int *));
    reportStatus("Problem 72: Array parameter decay check", ok);
    if (ok)
      passed++;
    total++;
  }

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

  // P75: Void* cast
  {
    float val = 4.5f;
    void *vptr = &val;
    float result = 0.0f;
    // TODO: Dereference vptr to read the float value

    bool ok = (std::abs(result - 4.5f) < 1e-5);
    reportStatus("Problem 75: Void pointer casting", ok);
    if (ok)
      passed++;
    total++;
  }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "--- SCORECARD: Type Casts & Double Pointers ---" << std::endl;
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
