#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  {
    const int N = 125;
    std::vector<float> data(N);
    float expected_sum = 0.0f;
    for (int i = 0; i < N; ++i) {
      data[i] = static_cast<float>(i * 0.1f + 0.5f);
      expected_sum += data[i];
    }

    // TODO: In-place tree reduction with odd-length boundary folding.
    // --- YOUR CODE STARTS HERE ---

    int oldStride = N;
    for (int stride = N >> 1; stride > 0; stride = stride >> 1) {
      if (stride * 2 != oldStride) {
        data[0] = data[0] + data[oldStride - 1];
      }
      oldStride = stride;
      for (int i = 0; i < stride; i++) {
        data[i] = data[i] + data[i + stride];
      }
    }
    cout << "Calculated value of the entire array after tree reduction is "
         << data[0] << endl;
    cout << "Expected value of the data is " << expected_sum << endl;
    // --- YOUR CODE ENDS HERE ---

    // bool p1_passed = (std::abs(data[0] - expected_sum) < 1e-2f);
    //  std::cout << "Problem 3 Status: " << (p1_passed ? "[PASSED]" :
    //  "[FAILED]") << std::endl;
  }

  return 0;
}
