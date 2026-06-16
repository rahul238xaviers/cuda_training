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
    std::cout << "--- WORKBOOK: ML Layer Offsets (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P93: Group Normalization Channel Index mapping
  {
    // Batch N, Channels C, Groups G. Get group index for channel c.
    // C = 16, G = 4. Group index = c / (C/G). For c = 9.
    int C = 16, G = 4, c = 9;
    int group_idx = -1;
    // TODO: Compute group_idx for channel c

    bool ok = (group_idx == 2);
    reportStatus("Problem 93: Group norm channel group mapping", ok);
    if (ok)
      passed++;
    total++;
  }

  // P94: Layer Normalization Channel Offset
  {
    // LN input tensor of size (Batch B, SeqLen S, HiddenDim H)
    // Find pointer to starting hidden layer element of Batch b=1, Seq s=5
    int B = 2, S = 10, H = 128;
    float input_tensor[2 * 10 * 128];
    float *cell_ptr = nullptr;
    // TODO: Get pointer to starting address of (b=1, s=5)

    bool ok = (cell_ptr == &input_tensor[1 * (S * H) + 5 * H]);
    reportStatus("Problem 94: Layer norm feature pointer mapping", ok);
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
