#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

int main() {
    {
        const int N = 64;
        const float PIVOT = 2.0f;
        std::vector<float> logits(N);
        int expected_above = 0;
        for (int i = 0; i < N; ++i) {
            logits[i] = static_cast<float>((i % 11) - 5) * 0.75f;
            if (logits[i] >= PIVOT) expected_above++;
        }

        int num_above = 0;
        float* left = logits.data();
        float* right = logits.data() + (N - 1);

        while (left < right) {
            if (*left >= PIVOT) {
                left++;
                num_above++;
            }
            if (*right < PIVOT) {
                right--;
            }

            if (*left < PIVOT && *right >= PIVOT && left < right) {
                cout << "swap is happening left is " << *left << " and the right is " << *right
                     << endl;
                swap(*left, *right);
                num_above++;
                left++;
                right--;
            }
        }
        bool p2_passed = (num_above == expected_above);
        for (int i = 0; i < num_above && p2_passed; ++i) {
            if (logits[i] < PIVOT) p2_passed = false;
        }
        for (int i = num_above; i < N && p2_passed; ++i) {
            if (logits[i] >= PIVOT) p2_passed = false;
        }
        cout << "The test has passed " << p2_passed << endl;

        return 0;
    }
}
