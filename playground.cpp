#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace std;

int main() {
  // =========================================================================
  // Problem 3: Cache-Blocked 2D Matrix Transposition
  //
  // Context:
  // Matrix size is 1024 x 1024 floats (4 MB total).
  // Naive transpose writes down columns across 1,024 different cache lines,
  // causing continuous L1 cache evictions.
  //
  // The Golden Formula gave us:
  //   TILE = 32 (32x32 floats = 4 KB, comfortably staying hot in L1 cache).
  //
  // Task:
  // Transpose `src` of shape [1024, 1024] into `dst` of shape [1024, 1024]
  // using 2D cache tiles of size TILE x TILE.
  //
  // For each element at logical position (r, c):
  //   Source index:      r * Cols + c
  //   Destination index: c * Rows + r
  // =========================================================================

  const int Rows = 1024;
  const int Cols = 1024;
  const int TILE = 32;
  const size_t total_elements = (size_t)Rows * Cols;

  std::vector<float> src(total_elements);
  for (size_t i = 0; i < total_elements; ++i) {
    src[i] = static_cast<float>(i + 1);
  }
  std::vector<float> dst(total_elements, 0.0f);

  auto start = std::chrono::high_resolution_clock::now();

  // TODO: Implement the 2D Tiled Transpose:
  //   1. Outer loop 'r0' steps from 0 to Rows with step TILE
  //   2. Next loop 'c0' steps from 0 to Cols with step TILE
  //   3. Inner loop 'r' goes from r0 to r0 + TILE
  //   4. Inner loop 'c' goes from c0 to c0 + TILE
  //   5. Copy: dst[c * Rows + r] = src[r * Cols + c];
  // --- YOUR CODE STARTS HERE ---
  // Step 1: Step through tile grid by TILE increments

  // --- YOUR CODE ENDS HERE ---

  auto end = std::chrono::high_resolution_clock::now();
  double elapsed_sec = std::chrono::duration<double>(end - start).count();
  double bytes_moved = 2.0 * total_elements * sizeof(float);
  double throughput = (bytes_moved / elapsed_sec) / 1e9;

  bool passed = true;
  for (int r = 0; r < Rows && passed; ++r) {
    for (int c = 0; c < Cols; ++c) {
      if (dst[c * Rows + r] != src[r * Cols + c]) {
        passed = false;
        break;
      }
    }
  }

  cout << fixed << setprecision(2);
  cout << "========================================\n";
  cout << "Problem 3: Cache-Blocked 2D Transpose\n";
  cout << "========================================\n";
  cout << "Matrix Size: " << Rows << " x " << Cols << " ("
       << (total_elements * sizeof(float)) / (1024 * 1024) << " MB)\n";
  cout << "Tile Size:   " << TILE << " x " << TILE << " ("
       << (TILE * TILE * sizeof(float)) / 1024 << " KB)\n";
  cout << "Time:        " << elapsed_sec * 1000.0 << " ms\n";
  cout << "Throughput:  " << throughput << " GB/s\n\n";

  if (passed) {
    cout
        << "\033[1;32m[TEST PASSED] Matrix Transpose is 100% correct!\033[0m\n";
  } else {
    cout << "\033[1;31m[TEST FAILED] Transpose values do not match "
            "expected.\033[0m\n";
  }

  return passed ? 0 : 1;
}
