#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>
using namespace std;
int main() {
  // Your playground: write and test code here
  const int ROWS = 16;
  const int COLS = 8;
  std::vector<float> mat(ROWS * COLS);
  for (int r = 0; r < ROWS; ++r) {
    for (int c = 0; c < COLS; ++c) {
      mat[r * COLS + c] = static_cast<float>(r * 100 + c);
    }
  }

  const int target_col = 3;
  std::vector<float> col_vec(ROWS, 0.0f);
  float *col_vec_ptr = col_vec.data();

  // TODO: Extract column target_col using pointer stepping with stride
  // COLS.
  // --- YOUR CODE STARTS HERE ---
  for (int i = 3; i < 128; i += 8) {
    *col_vec_ptr = mat[i];
    col_vec_ptr++;
  }
  // --- YOUR CODE ENDS HERE ---

  for (int i = 0; i < 16; i++) {
    cout << col_vec[i] << " ";
    cout << endl;
  }

  return 0;
}
