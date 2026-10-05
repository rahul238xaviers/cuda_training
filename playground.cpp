#include <iostream>
#include <vector>

void print_array(const std::string &label, const std::vector<int> &arr) {
  std::cout << label << " [ ";
  for (int x : arr) {
    std::cout << x << " ";
  }
  std::cout << "]" << std::endl;
}

int main() {
  std::cout << "=========================================" << std::endl;
  std::cout << "--- PLAYGROUND: In-Place Tree Reduction ---" << std::endl;
  std::cout << "=========================================" << std::endl;

  // -------------------------------------------------------------------------
  // CHALLENGE: In-Place Tree Reduction
  //
  // Given an array `arr` of size N = 8:
  // Reduce all elements in-place such that `arr[0]` holds the total sum
  // using the CUDA-style tree reduction pattern.
  // -------------------------------------------------------------------------
  const int N = 8;
  std::vector<int> arr = {1, 2, 3, 4, 5, 6, 7, 8};

  print_array("Initial array: ", arr);

  // --- YOUR CODE STARTS HERE ---

  for (int s = N >> 1; s > 0; s = s >> 1) {
    for (int i = 0; i < s; i++) {
      arr[i] += arr[i + s];
    }
  }

  // --- YOUR CODE ENDS HERE ---

  print_array("Final array:   ", arr);
  std::cout << "Total Sum at arr[0] = " << arr[0] << " (Expected: 36)"
            << std::endl;

  return 0;
}
