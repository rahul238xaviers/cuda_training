#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>
using namespace std;
int main() {
  // Your playground: write and test code here
  int DIM = 4;
  int TILE_SIZE = 2;

  const int total_elements = DIM * DIM;
  vector<float> src(total_elements);
  vector<float> dest(total_elements);

  for (int i = 0; i < total_elements; ++i) {
    src[i] = static_cast<float>(i);
  }

  for (int tile_number = 0; tile_number < DIM; tile_number++) {

    int global_tile_start_row = (tile_number / 2) * 2;
    int global_tile_start_col = (tile_number % 2) * 2;

    for (int current_row = global_tile_start_row;
         current_row < global_tile_start_row + 2; current_row++) {

      for (int current_col = global_tile_start_col;
           current_col < global_tile_start_col + 2; current_col++) {
        dest[current_col * DIM + current_row] =
            src[current_row * DIM + current_col];
      }
    }
  }

  for (int i = 0; i < total_elements; i++) {
    cout << dest[i] << " ";
    if ((i + 1) % DIM == 0 && i != 0) {
      cout << endl;
    }
  }

  return 0;
}
