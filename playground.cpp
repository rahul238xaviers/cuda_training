#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace std;

int main() {
  // -------------------------------------------------------------------------
  // Problem 3: Memory Stride & Cache Thrashing Benchmark
  //
  // Context:
  // Your Apple M-series CPU has a cache line size of 128 bytes (32 floats).
  // When STRIDE = 32, every read jumps 128 bytes, landing on a completely
  // different cache line!
  //
  // Task:
  // Touch every element in `buffer` across 32 strided passes:
  //   Pass 0 (s = 0):  0,  32,  64,  96, ...
  //   Pass 1 (s = 1):  1,  33,  65,  97, ...
  //   ...
  //   Pass 31 (s = 31): 31, 63, 95, 127, ...
  //
  // Total elements visited = N (524,288).
  // Since each element is 1.0f, the expected sum is 524288.0f.
  // -------------------------------------------------------------------------

  const int N = 524288;
  const int STRIDE = 32;
  std::vector<float> buffer(N, 1.0f);

  float strided_sum = 0.0f;

  auto start = std::chrono::high_resolution_clock::now();

  // TODO: Implement the two nested loops:
  //   1. Outer loop over offset 's' from 0 to STRIDE - 1
  //   2. Inner loop over index 'i' starting from 's', up to N, stepping by
  //   STRIDE
  //   3. Accumulate buffer[i] into strided_sum
  // --- YOUR CODE STARTS HERE ---

  for (int s = 0; s <= STRIDE - 1; s++) {
    for (int i = s; i < N; i += STRIDE) {
      strided_sum += buffer[i];
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsed_sec = std::chrono::duration<double>(end - start).count();
  double throughput_gbps = (N * sizeof(float) / elapsed_sec) / 1e9;

  // --- Contiguous Benchmark (Stride = 1) ---
  float contiguous_sum = 0.0f;
  auto start_contig = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < N; ++i) {
    contiguous_sum += buffer[i];
  }

  auto end_contig = std::chrono::high_resolution_clock::now();
  double elapsed_contig_sec =
      std::chrono::duration<double>(end_contig - start_contig).count();
  double throughput_contig_gbps =
      (N * sizeof(float) / elapsed_contig_sec) / 1e9;

  cout << fixed << setprecision(4);
  cout << "========================================\n";
  cout << "1. STRIDED TRAVERSAL (STRIDE = 32):\n";
  cout << "   Time:       " << elapsed_sec * 1000.0 << " ms\n";
  cout << "   Throughput: " << throughput_gbps << " GB/s\n";
  cout << "----------------------------------------\n";
  cout << "2. CONTIGUOUS TRAVERSAL (STRIDE = 1):\n";
  cout << "   Time:       " << elapsed_contig_sec * 1000.0 << " ms\n";
  cout << "   Throughput: " << throughput_contig_gbps << " GB/s\n";
  cout << "----------------------------------------\n";
  cout << "   Speedup:    " << (elapsed_sec / elapsed_contig_sec) << "x faster!\n";
  cout << "========================================\n\n";

  bool passed = (std::abs(strided_sum - static_cast<float>(N)) < 1e-3f);
  if (passed) {
    cout << "\033[1;32m[TEST PASSED] Cache-thrashing strided sum is "
            "correct!\033[0m\n";
  } else {
    cout
        << "\033[1;31m[TEST FAILED] Sum did not match expected value.\033[0m\n";
  }

  return 0;
}
