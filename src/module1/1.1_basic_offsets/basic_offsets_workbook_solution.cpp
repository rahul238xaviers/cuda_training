#include <cstdint>
#include <iomanip>
#include <iostream>
#include <ostream>

// Global status reporter
void reportStatus(const std::string &name, bool passed) {
  std::cout << "  " << std::left << std::setw(65) << name;
  if (passed) {
    std::cout << "\033[1;32m[PASSED]\033[0m" << std::endl;
  } else {
    std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
  }
}

// Structs used across exercises
struct Point2D {
  int r;
  int c;
};
struct Point3D {
  int d;
  int h;
  int w;
};
struct Tensor4D {
  int b;
  int c;
  int h;
  int w;
};

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
  float *data;
};

int main() {
  std::cout
      << "================================================================="
      << std::endl;
  std::cout << "--- WORKBOOK: Basic Offset & Increment (Problems 1-5) ---"
            << std::endl;
  std::cout
      << "================================================================="
      << std::endl;

  int passed = 0;
  int total = 0;

  // P1: Add offset to int*
  {
    int arr[10] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    int *ptr = arr;
    int *res = nullptr;
    // TODO: Make res point to the element at index 5 using pointer arithmetic
    // ptr point to the memory address of the start of the array. Adding 5 will
    // move the address to index 5.
    res = (ptr + 5);

    std::cout << "The value of pointer *ptr is " << *ptr << std::endl;

    bool ok = (res == &arr[5]);
    reportStatus("Problem 1: Add offset to int*", ok);
    if (ok)
      passed++;
    total++;
  }

  // P2: Subtract offset from float*
  {
    float arr[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                     6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float *ptr = &arr[9];
    float *res = nullptr;
    // TODO: Make res point to the element at index 3 using pointer subtraction

    res = ptr - 6;

    bool ok = (res == &arr[3]);
    reportStatus("Problem 2: Subtract offset from float*", ok);
    if (ok)
      passed++;
    total++;
  }

  // P3: Difference between pointers
  {
    double arr[10] = {0};
    double *ptr_a = &arr[2];
    double *ptr_b = &arr[8];
    ptrdiff_t diff = 0;
    // TODO: Compute distance (in elements) between ptr_b and ptr_a (ptr_b -
    // ptr_a)

    diff = ptr_b - ptr_a;

    /* std::cout << "Ptra value is " << ptr_a << "Ptrb value is " << ptr_b
               << "and the difference is " << diff << std::endl;*/

    bool ok = (diff == 6);
    reportStatus("Problem 3: Pointer difference (distance in elements)", ok);
    if (ok)
      passed++;
    total++;
  }

  // P4: Distance in bytes
  {
    int arr[10] = {0};
    int *ptr_a = &arr[1];
    int *ptr_b = &arr[6];
    ptrdiff_t byte_diff = 0;
    // TODO: Calculate distance in bytes between ptr_b and ptr_a by casting to
    // char*

    std::cout << "The byte value of ptr_b is "
              << reinterpret_cast<std::uintptr_t>(ptr_b) -
                     reinterpret_cast<std::uintptr_t>(ptr_a)
              << std::endl;

    byte_diff = reinterpret_cast<std::uintptr_t>(ptr_b) -
                reinterpret_cast<std::uintptr_t>(ptr_a);

    bool ok = (byte_diff == 5 * sizeof(int));
    reportStatus("Problem 4: Pointer distance in bytes", ok);
    if (ok)
      passed++;
    total++;
  }

  // P5: Prefix increment on pointer
  {
    int arr[5] = {10, 20, 30, 40, 50};
    int *ptr = &arr[0];
    int val = 0;
    // TODO: Increment ptr using prefix (++ptr) and assign dereferenced value to
    // val (in one line or two)

    val = *(++ptr);

    std::cout << "The value of incremented pointer is " << val << std::endl;

    bool ok = (ptr == &arr[1] && val == 20);
    reportStatus("Problem 5: Prefix pointer increment", ok);
    if (ok)
      passed++;
    total++;
  }

  std::cout
      << "\n================================================================="
      << std::endl;
  std::cout << "--- SCORECARD: Basic Offset & Increment ---" << std::endl;
  std::cout
      << "================================================================="
      << std::endl;
  std::cout << "  Passed: " << passed << " / " << total << " tests."
            << std::endl;
  if (passed == total) {
    std::cout << "\033[1;32m  [STATUS] ALL " << total
              << " TESTS PASSED! \033[0m" << std::endl;
  } else {
    std::cout << "\033[1;31m  [STATUS] INCOMPLETE (" << (total - passed)
              << " tests failed) \033[0m" << std::endl;
  }
  std::cout
      << "================================================================="
      << std::endl;

  return (passed == total) ? 0 : 1;
}
