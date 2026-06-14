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
    std::cout << "--- WORKBOOK: Size, Padding & Punning (Problems 66-70) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P66: alignas Struct size
  {
    size_t size = sizeof(AlignedStruct);
    // TODO: Check if the size of AlignedStruct is a multiple of 16 due to
    // alignas(16)
    bool ok = (size % 16 == 0);
    reportStatus("Problem 66: alignas size verification", ok);
    if (ok)
      passed++;
    total++;
  }

  // P67: Struct padding check
  {
    size_t size = sizeof(UnalignedStruct);
    // TODO: Read layout size of UnalignedStruct. Set is_padded = true if it's
    // larger than the sum of its member sizes (1 + 8 + 4 = 13 bytes).
    bool is_padded = false;
    if (size > 13)
      is_padded = true;

    bool ok = is_padded;
    reportStatus("Problem 67: Struct padding calculation", ok);
    if (ok)
      passed++;
    total++;
  }

  // P68: sizeof vs elements check
  {
    int arr[10] = {0};
    size_t array_elements = 0;
    // TODO: Compute number of elements in arr using sizeof operator
    // (sizeof(arr) / sizeof(arr[0]))

    bool ok = (array_elements == 10);
    reportStatus("Problem 68: Sizeof array elements verification", ok);
    if (ok)
      passed++;
    total++;
  }

  // P69: Reinterpret float to int bits
  {
    float f = 1.0f; // Bit representation in IEEE-754: 0x3f800000
    int bits = 0;
    // TODO: Cast address of f to int* using reinterpret_cast and read the
    // integer bits

    bool ok = (bits == 0x3f800000);
    reportStatus("Problem 69: Reinterpret bit cast", ok);
    if (ok)
      passed++;
    total++;
  }

  // P70: Type punning inspection
  {
    float f = -1.0f;
    unsigned char bytes[4] = {0};
    // TODO: Copy the bytes of f directly into bytes array using pointer casts
    // (memcpy or char pointers)

    bool ok = (bytes[3] == 0xbf && bytes[2] == 0x80); // Sign bit in high byte
    reportStatus("Problem 70: Type punning via char pointer", ok);
    if (ok)
      passed++;
    total++;
  }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "--- SCORECARD: Size, Padding & Punning ---" << std::endl;
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
