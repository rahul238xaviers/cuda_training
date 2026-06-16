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
    std::cout << "--- WORKBOOK: Lifetimes, Ownership & Resize (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P58: Stack vs Heap allocation benchmarking
  {
    // Simple verification that stack addresses are local
    int stack_var = 10;
    int *heap_ptr = new int(10);
    bool stack_is_local = false;
    // TODO: Check if address of stack_var is local to stack (higher memory
    // region than heap_ptr in most Linux architectures) Set stack_is_local =
    // true if &stack_var > heap_ptr
    if (&stack_var > heap_ptr)
      stack_is_local = true;

    bool ok = stack_is_local;
    delete heap_ptr;
    reportStatus("Problem 58: Stack address detection", ok);
    if (ok)
      passed++;
    total++;
  }

  // P59: Dynamic vector growth simulation
  {
    int *arr = new int[2]{10, 20};
    int capacity = 2;
    // TODO: Resize the array to size 4, copy old values, add {30, 40}, free old
    // array

    bool ok = (arr != nullptr && capacity == 4 && arr[0] == 10 && arr[3] == 40);
    delete[] arr;
    reportStatus("Problem 59: Dynamic array resize simulation", ok);
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
