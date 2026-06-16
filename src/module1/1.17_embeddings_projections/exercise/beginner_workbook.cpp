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
    std::cout << "--- WORKBOOK: Embeddings & Projections (Beginner Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

  // P81: Embedding Lookup
  {
    float embeddings[4 * 8] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
                               1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f,
                               2.0f, 2.1f, 2.2f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f,
                               3.0f, 3.1f, 3.2f, 3.3f, 3.4f, 3.5f, 3.6f, 3.7f};
    int token_id = 2;
    int embedding_dim = 8;
    float *lookup_ptr = nullptr;
    // TODO: Get pointer to the embedding vector of token_id

    bool ok = (lookup_ptr == &embeddings[16] && lookup_ptr[1] == 2.1f);
    reportStatus("Problem 81: Embedding table lookup offset", ok);
    if (ok)
      passed++;
    total++;
  }

  // P82: Batched Embedding Lookup
  {
    float embeddings[4 * 4] = {0.0f, 0.1f, 0.2f, 0.3f, 1.0f, 1.1f, 1.2f, 1.3f,
                               2.0f, 2.1f, 2.2f, 2.3f, 3.0f, 3.1f, 3.2f, 3.3f};
    int batch_tokens[2] = {3, 1};
    float *batch_ptrs[2] = {nullptr, nullptr};
    // TODO: Store pointer for each token in batch_tokens into batch_ptrs

    bool ok =
        (batch_ptrs[0] == &embeddings[12] && batch_ptrs[1] == &embeddings[4]);
    reportStatus("Problem 82: Batched embedding row resolution", ok);
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
