#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
using namespace std;

int main() {
  const int M = 4;
  const int NNZ = 6;

  // CSR Sparse Matrix representation:
  // Row 0: [ 10,   0,  20,   0 ]
  // Row 1: [  0,  30,   0,   0 ]
  // Row 2: [ 40,   0,   0,  50 ]
  // Row 3: [  0,   0,  60,   0 ]
  std::vector<int> row_ptr = {0, 2, 3, 5, 6};
  std::vector<int> col_indices = {0, 2, 1, 0, 3, 2};
  std::vector<float> values = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

  // Dense input vector x:
  std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};

  // Output vector y (initially all 0.0f):
  std::vector<float> y(M, 0.0f);

  // Expected output:
  // Row 0: 10*1 + 20*3 = 70
  // Row 1: 30*2 = 60
  // Row 2: 40*1 + 50*4 = 240
  // Row 3: 60*3 = 180
  std::vector<float> expected_y = {70.0f, 60.0f, 240.0f, 180.0f};

  // TODO: Compute y[r] using CSR SpMV traversal.
  // --- YOUR CODE STARTS HERE ---

  for (int row_ptr_incr = 0; row_ptr_incr < 4; row_ptr_incr++) {

    int current_row_ptr_index_value = row_ptr[row_ptr_incr];
    int next_row_ptr_index_value = row_ptr[row_ptr_incr + 1];
    float verctorMultiplicationValue = 0.0f;
    for (int start = current_row_ptr_index_value;
         start < next_row_ptr_index_value; start++) {

      verctorMultiplicationValue += values[start] * x[col_indices[start]];
    }

    y[row_ptr_incr] = verctorMultiplicationValue;
  }

  // --- YOUR CODE ENDS HERE ---

  // Print results:
  cout << "Computed y: [ ";
  for (float val : y)
    cout << val << " ";
  cout << "]\n";

  cout << "Expected y: [ ";
  for (float val : expected_y)
    cout << val << " ";
  cout << "]\n\n";

  bool passed = true;
  for (int r = 0; r < M; ++r) {
    if (std::abs(y[r] - expected_y[r]) > 1e-4f)
      passed = false;
  }

  if (passed) {
    cout << "[TEST PASSED] CSR SpMV matches expected output!\n";
  } else {
    cout << "[TEST FAILED] Output does not match expected.\n";
  }

  return 0;
}
