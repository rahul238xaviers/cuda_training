#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

// =========================================================================
// CHAMPION WORKBOOK: Strides & Indirection
//
// Module: 1.2 - Strided Indexing & Indirection Patterns
// Level:  Champion / High-Performance Systems Engineer
//
// Focus: CSR compression & decompression, CSR SpMV indirection, generalized
//        3D tensor permutation, and cache-blocked 2D matrix transposition.
//
// Compilation:
//   clang++ -std=c++20 -O3 exercise/champion_workbook.cpp -o champion_test
//   ./champion_test
// =========================================================================

void reportStatus(const std::string &name, bool passed,
                  double throughputGBs = -1.0) {
  std::cout << "  " << std::left << std::setw(60) << name;
  if (passed) {
    std::cout << "\033[1;32m[PASSED]\033[0m";
    if (throughputGBs > 0.0) {
      std::cout << " (" << std::fixed << std::setprecision(2) << throughputGBs
                << " GB/s)";
    }
    std::cout << std::endl;
  } else {
    std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
  }
}

int main() {
  std::cout
      << "================================================================="
      << std::endl;
  std::cout << "--- WORKBOOK: Strides & Indirection (Champion Masterclass) ---"
            << std::endl;
  std::cout
      << "================================================================="
      << std::endl;

  int passed = 0;
  int total = 0;

  // Shared 4x4 Sparse Matrix for Problems 1, 2, and 3:
  //
  //   Row 0: [ 10,   0,  20,   0 ]  -> (2 non-zeros at col 0, col 2)
  //   Row 1: [  0,  30,   0,   0 ]  -> (1 non-zero  at col 1)
  //   Row 2: [ 40,   0,   0,  50 ]  -> (2 non-zeros at col 0, col 3)
  //   Row 3: [  0,   0,  60,   0 ]  -> (1 non-zero  at col 2)
  const int M = 4;
  const int N = 4;
  const std::vector<float> dense_matrix = {
      10.0f,  0.0f, 20.0f,  0.0f,
       0.0f, 30.0f,  0.0f,  0.0f,
      40.0f,  0.0f,  0.0f, 50.0f,
       0.0f,  0.0f, 60.0f,  0.0f
  };

  // -------------------------------------------------------------------------
  // PROBLEM 1: CSR Matrix Compression (Dense -> CSR Packing)
  //
  // Context: Pruning neural networks zeros out 50-80% of weights. Before
  //          saving the checkpoint, we compress the 2D dense matrix into
  //          three compact 1D CSR arrays: row_ptr, col_indices, and values.
  //
  // Given:
  //   - Dense matrix `dense_matrix` of shape M = 4, N = 4 (16 floats).
  //   - Output CSR arrays: `row_ptr`, `col_indices`, and `values`.
  //
  // Goal:
  //   - Compress non-zero elements into `values`, record column indices in `col_indices`,
  //     and build the row offset array `row_ptr`.
  //
  // Reference:
  //   - See Section 3 (Pointer Indirection: The Two Mental Models) of theory.md.
  // -------------------------------------------------------------------------
  std::vector<int> row_ptr;
  std::vector<int> col_indices;
  std::vector<float> values;
  {
    // TODO: Compress the dense matrix into CSR format (row_ptr, col_indices, values).
    // --- YOUR CODE STARTS HERE ---

    // --- YOUR CODE ENDS HERE ---

    std::vector<int> expected_row_ptr = {0, 2, 3, 5, 6};
    std::vector<int> expected_col_indices = {0, 2, 1, 0, 3, 2};
    std::vector<float> expected_values = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

    bool p1_passed = (row_ptr == expected_row_ptr &&
                      col_indices == expected_col_indices &&
                      values == expected_values);

    reportStatus("Problem 1: CSR Matrix Compression (Dense -> CSR Packing)", p1_passed);
    if (p1_passed) passed++;
    total++;
  }

  // -------------------------------------------------------------------------
  // PROBLEM 2: CSR Matrix Decompression (CSR -> Dense Unpacking)
  //
  // Context: To inspect pruned layers, debug sparsity, or execute dense GEMM,
  //          we decompress the 3 CSR arrays back into a full 2D dense tensor.
  //
  // Given:
  //   - Compressed CSR format: `row_ptr`, `col_indices`, and `values`.
  //   - Matrix dimensions: M = 4, N = 4.
  //   - Output dense matrix buffer `reconstructed` (size M * N).
  //
  // Goal:
  //   - Decompress the CSR representation back into full 2D dense layout in `reconstructed`.
  //
  // Reference:
  //   - See Section 3 (Pointer Indirection: The Two Mental Models) of theory.md.
  // -------------------------------------------------------------------------
  {
    std::vector<float> reconstructed(M * N, 0.0f);

    // TODO: Decompress CSR representation back into full 2D dense matrix format.
    // --- YOUR CODE STARTS HERE ---

    // --- YOUR CODE ENDS HERE ---

    bool p2_passed = (reconstructed == dense_matrix);
    reportStatus("Problem 2: CSR Matrix Decompression (CSR -> Dense Unpacking)", p2_passed);
    if (p2_passed) passed++;
    total++;
  }

  // -------------------------------------------------------------------------
  // PROBLEM 3: Compressed Sparse Row (CSR) Sparse Matrix-Vector Multiply (SpMV)
  //
  // Context: In Graph Neural Networks and sparse inference, SpMV computes
  //          y = Matrix * x directly from the compressed CSR format without
  //          multiplying any zero elements.
  //
  // Given:
  //   - Compressed matrix in CSR format: `row_ptr`, `col_indices`, and `values`.
  //   - Dense input vector `x` of size N = 4.
  //   - Output vector `y` of size M = 4.
  //
  // Goal:
  //   - Compute y = A * x directly from the CSR format without visiting zero elements.
  //
  // Reference:
  //   - See Section 3 (Pointer Indirection) and Section 5 (Cache Lines & Bandwidth Waste) of theory.md.
  // -------------------------------------------------------------------------
  {
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> y(M, 0.0f);

    // TODO: Compute SpMV product y = A * x directly from CSR compressed representation.
    // --- YOUR CODE STARTS HERE ---

    // --- YOUR CODE ENDS HERE ---

    // Expected:
    // Row 0: 10*1 + 20*3 = 70
    // Row 1: 30*2 = 60
    // Row 2: 40*1 + 50*4 = 240
    // Row 3: 60*3 = 180
    std::vector<float> expected_y = {70.0f, 60.0f, 240.0f, 180.0f};

    bool p3_passed = true;
    for (int r = 0; r < M; ++r) {
      if (std::abs(y[r] - expected_y[r]) > 1e-4f) p3_passed = false;
    }

    reportStatus("Problem 3: Compressed Sparse Row (CSR) SpMV Indirection", p3_passed);
    if (p3_passed) passed++;
    total++;
  }

  // -------------------------------------------------------------------------
  // PROBLEM 4: Generalized 3D Tensor Stride Permutation (D0, D1, D2 -> D0, D2, D1)
  //
  // Context: In GPU tensor frameworks, permuting dimensions requires calculating
  //          generalized strides:
  //            stride[dim] = prod_{d > dim} shape[d]
  //
  // Given:
  //   - Tensor `src` of shape [D0=4, D1=8, D2=16] (512 floats).
  //   - Destination tensor `dst` of shape [D0=4, D2=16, D1=8].
  //
  // Goal:
  //   - Permute dimensions (D0, D1, D2) -> (D0, D2, D1) using multi-dimensional stride mapping.
  //   - Benchmark memory bandwidth throughput in GB/s.
  //
  // Reference:
  //   - See Section 1 (Flat Address Mapping Formula) and Section 2 (Row-Major vs Column-Major Layouts) of theory.md.
  // -------------------------------------------------------------------------
  {
    const int D0 = 4, D1 = 8, D2 = 16;
    const size_t total_elements = D0 * D1 * D2;
    std::vector<float> src(total_elements);
    for (size_t i = 0; i < total_elements; ++i)
      src[i] = static_cast<float>(i + 1);

    std::vector<float> dst(total_elements, -1.0f);

    auto start = std::chrono::high_resolution_clock::now();

    // TODO: Permute 3D tensor dimensions (D0, D1, D2) -> (D0, D2, D1) using stride equations.
    // --- YOUR CODE STARTS HERE ---

    // --- YOUR CODE ENDS HERE ---

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();
    double bytes_moved = 2.0 * total_elements * sizeof(float);
    double throughput = (bytes_moved / elapsed_sec) / 1e9;

    bool p4_passed = true;
    for (int d0 = 0; d0 < D0 && p4_passed; ++d0) {
      for (int d1 = 0; d1 < D1 && p4_passed; ++d1) {
        for (int d2 = 0; d2 < D2; ++d2) {
          size_t src_idx = (size_t)d0 * (D1 * D2) + (size_t)d1 * D2 + d2;
          size_t dst_idx = (size_t)d0 * (D2 * D1) + (size_t)d2 * D1 + d1;
          if (dst[dst_idx] != src[src_idx]) {
            p4_passed = false;
            break;
          }
        }
      }
    }

    reportStatus("Problem 4: Generalized 3D Tensor Stride Permutation",
                 p4_passed, p4_passed ? throughput : -1.0);
    if (p4_passed) passed++;
    total++;
  }

  // -------------------------------------------------------------------------
  // PROBLEM 5: Cache-Blocked 2D Matrix Transposition (Golden Tile Rule: TILE=32)
  //
  // Context: In GPU & CPU systems, naive matrix transpose causes strided writes
  //          that thrash cache lines across memory. Using the Golden Formula:
  //            T <= sqrt( (alpha * C_L1) / (2 * sizeof(float)) ) = 32
  //          We decompose a 1024x1024 matrix into 32x32 tiles that stay 100%
  //          resident inside L1 cache lines with zero eviction thrashing.
  //
  // Given:
  //   - Matrix `src` of shape [Rows=1024, Cols=1024] (1,048,576 floats).
  //   - Destination buffer `dst` of shape [Cols=1024, Rows=1024].
  //   - Golden tile dimension: TILE = 32.
  //
  // Goal:
  //   - Transpose `src` into `dst` using 32x32 cache-blocked tiles to eliminate cache thrashing.
  //   - Benchmark memory bandwidth throughput in GB/s.
  //
  // Reference:
  //   - See Section 5 (The Golden Tile Sizing Formula: In-Place vs. Out-of-Place) of theory.md.
  // -------------------------------------------------------------------------
  {
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

    // TODO: Transpose matrix using 32x32 cache-blocked tiles to eliminate cache thrashing.
    // --- YOUR CODE STARTS HERE ---

    // --- YOUR CODE ENDS HERE ---

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();
    double bytes_moved = 2.0 * total_elements * sizeof(float);
    double throughput = (bytes_moved / elapsed_sec) / 1e9;

    bool p5_passed = true;
    for (int r = 0; r < Rows && p5_passed; ++r) {
      for (int c = 0; c < Cols; ++c) {
        if (dst[c * Rows + r] != src[r * Cols + c]) {
          p5_passed = false;
          break;
        }
      }
    }

    reportStatus("Problem 5: Cache-Blocked 2D Matrix Transpose (TILE=32)",
                 p5_passed, p5_passed ? throughput : -1.0);
    if (p5_passed) passed++;
    total++;
  }

  std::cout
      << "\n================================================================="
      << std::endl;
  std::cout << "--- SCORECARD ---" << std::endl;
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
