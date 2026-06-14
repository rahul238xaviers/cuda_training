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
    std::cout << "--- WORKBOOK: ML Layer Offsets (Problems 91-95) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P91: Gradient Descent Weight Update Step
  {
    float W[3] = {1.0f, 2.0f, 3.0f};
    float G[3] = {0.1f, 0.2f, 0.3f};
    float lr = 0.1f;
    // TODO: Perform weight update in-place W = W - lr * G using pointer
    // increments

    bool ok = (std::abs(W[0] - 0.99f) < 1e-5 && std::abs(W[2] - 2.97f) < 1e-5);
    reportStatus("Problem 91: Gradient descent weight step updates", ok);
    if (ok)
      passed++;
    total++;
  }

  // P92: 2D Bias Addition Strides
  {
    // Add bias vector of size C=2 to matrix X of size N=3, C=2
    // X is stored row-major. Add bias[c] to X[n*C + c].
    float X[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float bias[2] = {0.5f, 1.5f};
    // TODO: Implement bias addition in-place on X using pointers

    bool ok = (X[0] == 1.5f && X[1] == 3.5f && X[4] == 5.5f && X[5] == 7.5f);
    reportStatus("Problem 92: Row-bias additions with stride offsets", ok);
    if (ok)
      passed++;
    total++;
  }

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

  // P95: Im2Col Output Coordinate Map
  {
    // Map 2D output feature map coordinate (out_h, out_w) to linear index
    // Dimensions: Out_H = 14, Out_W = 14
    int out_h = 2, out_w = 5;
    int flat_idx = -1;
    // TODO: Compute flat output index

    bool ok = (flat_idx == 33);
    reportStatus("Problem 95: Im2Col output flat coordinate mapping", ok);
    if (ok)
      passed++;
    total++;
  }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "--- SCORECARD: ML Layer Offsets ---" << std::endl;
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
