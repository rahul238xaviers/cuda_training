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
    std::cout << "--- WORKBOOK: ML Layer Offsets (Beginner Level) ---" << std::endl;
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
