#include <iostream>
#include <vector>

using namespace std;

struct WelfordAcc {
  int count = 0;
  float mean = 0.0f;
  float M2 = 0.0f;
};

int main() {
  {
    vector<float> blockA = {2.0, 4.0, 6.0, 8.0};
    vector<WelfordAcc> welfordAcc(4);

    float mean = 0.0f;
    int count = 0;
    float M2 = 0.0f;
    float oldMean = 0.0f;
    for (int i = 0; i < (int)blockA.size(); i++) {
      float a = blockA[i];
      mean = ((mean * count) + a) / (count + 1);
      M2 = M2 + (a - oldMean) * (a - mean);
      count++;
      welfordAcc[i].count = count;
      welfordAcc[i].mean = mean;
      welfordAcc[i].M2 = M2;
      oldMean = mean;
    }

    cout << "Calculated mean is " << mean << endl;
    cout << "Calculated M2 is " << M2 << endl;

    for (int i = 0; i < (int)welfordAcc.size(); i++) {

      WelfordAcc current_welfordAcc = welfordAcc[i];

      cout << "The count at step  " << i << " is " << current_welfordAcc.count
           << endl;
      cout << "The mean at step " << i << " is " << current_welfordAcc.mean
           << endl;
      cout << "The M2 at step " << i << " is " << current_welfordAcc.M2 << endl;
    }

    float variance = M2 / count;
    float standard_deviation = sqrt(variance);

    cout << "The variance is " << variance << endl;
    cout << "The standard deviation is  " << standard_deviation << endl;
  }
}
