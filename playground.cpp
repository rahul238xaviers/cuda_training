#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std;


struct WelfordAcc {
  int count = 0;
  float mean = 0.0f;
  float M2 = 0.0f;
};

int main() {
  {
    const int N = 1024;
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = N / BLOCK_SIZE;

    vector<float> activations(N);
    double gt_sum = 0.0;
    for (int i = 0; i < N; ++i) {
      activations[i] = 10000.0f + static_cast<float>((i % 7) - 3) * 0.125f;
      gt_sum += activations[i];
    }

    double gt_mean = gt_sum / N;
    double gt_var = 0.0;
    for (int i = 0; i < N; ++i) {
      double diff = activations[i] - gt_mean;
      gt_var += diff * diff;
    }
    gt_var /= N;
    double gt_std = sqrt(gt_var);

    struct WelfordAcc {
      int count = 0;
      float mean = 0.0f;
      float M2 = 0.0f;
    };

    // Each block pre-computes its local Welford statistics
    vector<WelfordAcc> block_accs(NUM_BLOCKS);
    for (int b = 0; b < NUM_BLOCKS; ++b) {
      int count = 0;
      float mean = 0.0f;
      float M2 = 0.0f;
      for (int i = 0; i < BLOCK_SIZE; ++i) {
        float x = activations[b * BLOCK_SIZE + i];
        count++;
        float delta1 = x - mean;
        mean += delta1 / count;
        float delta2 = x - mean;
        M2 += delta1 * delta2;
      }
      block_accs[b] = {count, mean, M2};
    }

    float global_mean = 0.0f;
    float global_variance = 0.0f;
    float global_std = 0.0f;

    auto start = chrono::high_resolution_clock::now();

    // TODO: Merge the 4 block_accs into a single global accumulator using
    // Chan's merge formula.
    // --- YOUR CODE STARTS HERE ---

    int global_count = 0;
    float M2 = 0.0f;
    float globalM2 = 0.0f;

    for (int i = 0; i < NUM_BLOCKS; i++) {
      WelfordAcc welfordAcc = block_accs[i];
      int current_count = welfordAcc.count;
      float current_mean = welfordAcc.mean;
      float curren_M2 = welfordAcc.M2;
      global_count += current_count;

      float delta = current_mean - global_mean;

      global_mean = global_mean + (delta * current_count / global_count);

      globalM2 = (globalM2 + curren_M2 +
                  ((delta * delta) * (global_count - current_count) *
                   current_count / global_count));
    }
    global_variance = globalM2 / global_count;
    global_std = sqrt(global_variance);

    // --- YOUR CODE ENDS HERE ---

    auto end = chrono::high_resolution_clock::now();
    double elapsed_sec = chrono::duration<double>(end - start).count();
    double bytes_moved = NUM_BLOCKS * sizeof(WelfordAcc);
    double throughput = (bytes_moved / elapsed_sec) / 1e9;

    bool p4_passed =
        (abs(global_mean - static_cast<float>(gt_mean)) < 1e-2f) &&
        (abs(global_variance - static_cast<float>(gt_var)) < 1e-4f) &&
        (abs(global_std - static_cast<float>(gt_std)) < 1e-4f);

    cout << "The case status is " << p4_passed << endl;
  }
}
