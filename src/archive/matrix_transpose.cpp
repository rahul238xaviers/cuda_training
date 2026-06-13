#include <iostream>
#include <vector>
#include <algorithm>
#include <execution>

int main() {
    // A 4x4 matrix dimensions
    const int rows = 4;
    const int cols = 4;

    // A flat 1D vector holding our 2D matrix data sequentially
    std::vector<int> source = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16
    };
    std::vector<int> dest(rows * cols, 0);

    // Create an index vector for the rows: [0, 1, 2, 3]
    std::vector<int> row_indices(rows);
    for (int i = 0; i < rows; ++i) row_indices[i] = i;

    // Parallel execution across rows using your CPU cores via TBB
    std::for_each(std::execution::par, row_indices.begin(), row_indices.end(), [&](int r) {
        for (int c = 0; c < cols; ++c) {
            // Standard 2D to 1D flat mapping calculation:
            int src_index  = r * cols + c;
            int dest_index = c * rows + r; // Flipped dimensions

            dest[dest_index] = source[src_index];
        }
    });

    // Print the result grid to verify
    std::cout << "Transposed Matrix (1D Flat Array Math):\n";
    for (int r = 0; r < cols; ++r) {
        for (int c = 0; c < rows; ++c) {
            std::cout << dest[r * rows + c] << "\t";
        }
        std::cout << "\n";
    }

    return 0;
}