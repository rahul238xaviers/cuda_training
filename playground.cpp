#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace std;

// Method 1: Scaled Indexing WITH __restrict__
__attribute__((noinline))
void run_with_restrict(
    const float* __restrict__ x, int stride_x,
    const float* __restrict__ y, int stride_y,
    float* __restrict__ z, int n
) {
    for (int i = 0; i < n; ++i) {
        z[i] = x[i * stride_x] * y[i * stride_y];
    }
}

// Method 2: Scaled Indexing WITHOUT __restrict__ (Compiler suspects aliasing!)
__attribute__((noinline))
void run_without_restrict(
    const float* x, int stride_x,
    const float* y, int stride_y,
    float* z, int n
) {
    for (int i = 0; i < n; ++i) {
        z[i] = x[i * stride_x] * y[i * stride_y];
    }
}

// Method 3: Pointer Stepping WITH __restrict__
__attribute__((noinline))
void run_pointer_stepping(
    const float* __restrict__ x, int stride_x,
    const float* __restrict__ y, int stride_y,
    float* __restrict__ z, int n
) {
    for (int i = 0; i < n; ++i) {
        *z++ = (*x) * (*y);
        x += stride_x;
        y += stride_y;
    }
}

int main() {
    const int OUT_N = 2000000;
    int stride_x = 2;
    int stride_y = 3;

    vector<float> X(OUT_N * stride_x, 1.01f);
    vector<float> Y(OUT_N * stride_y, 0.99f);
    vector<float> Z(OUT_N, 0.0f);

    const int RUNS = 30;

    // Benchmark 1: WITH restrict
    auto t0 = chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; ++r) {
        run_with_restrict(X.data(), stride_x, Y.data(), stride_y, Z.data(), OUT_N);
    }
    auto t1 = chrono::high_resolution_clock::now();
    double ms_restrict = chrono::duration<double, milli>(t1 - t0).count() / RUNS;

    // Benchmark 2: WITHOUT restrict
    auto t2 = chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; ++r) {
        run_without_restrict(X.data(), stride_x, Y.data(), stride_y, Z.data(), OUT_N);
    }
    auto t3 = chrono::high_resolution_clock::now();
    double ms_no_restrict = chrono::duration<double, milli>(t3 - t2).count() / RUNS;

    // Benchmark 3: Pointer Stepping
    auto t4 = chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; ++r) {
        run_pointer_stepping(X.data(), stride_x, Y.data(), stride_y, Z.data(), OUT_N);
    }
    auto t5 = chrono::high_resolution_clock::now();
    double ms_pointer = chrono::duration<double, milli>(t5 - t4).count() / RUNS;

    cout << fixed << setprecision(3);
    cout << "========================================================\n";
    cout << "  PLAYGROUND: THE PHYSICAL REALITY OF STRIDING & RESTRICT\n";
    cout << "========================================================\n";
    cout << "  1. Indexing WITH __restrict__   : " << ms_restrict << " ms\n";
    cout << "  2. Indexing WITHOUT __restrict__: " << ms_no_restrict << " ms\n";
    cout << "  3. Pointer Stepping (Explicit)  : " << ms_pointer << " ms\n";
    cout << "========================================================\n";

    return 0;
}
