#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>

using namespace std;

// =========================================================================
// BEGINNER WORKBOOK: Multi-Array & Strided Ops
//
// Module: 1.4 - Multi-Array Linear Algebra & Memory Traversal
// Level:  Beginner
//
// Focus: BLAS Level-1 SAXPY throughput, strided Hadamard products, and
//        strided cosine similarity computation.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.4_beginner
//   ../../../output/1.4_beginner
// =========================================================================

void reportStatus(const string& name, bool passed, double throughputGBs = -1.0) {
    cout << "  " << left << setw(60) << name;
    if (passed) {
        cout << "\033[1;32m[PASSED]\033[0m";
        if (throughputGBs > 0.0) {
            cout << " (" << fixed << setprecision(2) << throughputGBs << " GB/s)";
        }
        cout << endl;
    } else {
        cout << "\033[1;31m[FAILED]\033[0m" << endl;
    }
}

int main() {
    cout << "=================================================================" << endl;
    cout << "--- WORKBOOK: Multi-Array & Strided Ops (Beginner Level) ---" << endl;
    cout << "=================================================================" << endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: High-Performance SAXPY (Single-Precision A * X Plus Y)
    //
    // Context: BLAS-1 SAXPY (y = alpha * x + y) is the standard baseline benchmark
    //          for GPU memory streaming bandwidth.
    //
    // Task: Given vectors `X` and `Y` of size N=131,072 floats (512 KB each) and alpha=2.5f:
    //       Update Y in-place: Y[i] = alpha * X[i] + Y[i].
    //       Benchmark memory throughput in GB/s!
    // -------------------------------------------------------------------------
    {
        const int N = 131072;
        const float alpha = 2.5f;
        vector<float> X(N), Y(N);
        for (int i = 0; i < N; ++i) {
            X[i] = static_cast<float>(i * 0.01f);
            Y[i] = static_cast<float>((i % 100) * 0.5f);
        }
        vector<float> original_Y = Y;

        auto start = chrono::high_resolution_clock::now();

        // Scope block isolating raw pointer arithmetic:
        {
            // Extract raw memory pointers with non-aliasing __restrict__ contract:
            const float* __restrict__ pX = X.data();
            float* __restrict__ pY = Y.data();

            // Shadow high-level vector objects X and Y so naive vector indexing fails to compile:
            struct DisallowDirectVectorSubscripting {} X, Y;
            (void)X; (void)Y;

            // TODO: In-place SAXPY using pointer arithmetic on pX and pY:
            //       Update Y in-place: *pY = alpha * (*pX) + *pY; pX++; pY++;
            //       (or via pointer offset: pY[i] = alpha * pX[i] + pY[i];)
            // --- YOUR CODE STARTS HERE ---

            // --- YOUR CODE ENDS HERE ---
        }

        auto end = chrono::high_resolution_clock::now();
        double elapsed_sec = chrono::duration<double>(end - start).count();
        double bytes_moved = 3.0 * N * sizeof(float); // Read X, Read Y, Write Y
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        bool p1_passed = true;
        for (int i = 0; i < N; ++i) {
            float expected = alpha * X[i] + original_Y[i];
            if (abs(Y[i] - expected) > 1e-3f) {
                p1_passed = false;
                break;
            }
        }

        reportStatus("Problem 1: BLAS Level-1 SAXPY Vector Stream", p1_passed, throughput);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Strided Hadamard Product (Element-Wise Multiplication)
    //
    // Context: In gating mechanisms (e.g. GRU / LSTM gates), two activations are
    //          multiplied elementwise with stride skips: Z[i] = X[i * stride_x] * Y[i * stride_y].
    //
    // Task: Given `X` of size 128 (stride_x = 2) and `Y` of size 192 (stride_y = 3):
    //       Compute 64 elementwise products into `Z` [64].
    // -------------------------------------------------------------------------
    {
        const int OUT_N = 64;
        const int STRIDE_X = 2;
        const int STRIDE_Y = 3;

        vector<float> X(OUT_N * STRIDE_X);
        vector<float> Y(OUT_N * STRIDE_Y);
        for (size_t i = 0; i < X.size(); ++i) X[i] = static_cast<float>(i + 1);
        for (size_t i = 0; i < Y.size(); ++i) Y[i] = static_cast<float>((i % 10) + 1);

        vector<float> Z(OUT_N, 0.0f);

        {
            const float* __restrict__ pX = X.data();
            const float* __restrict__ pY = Y.data();
            float* __restrict__ pZ = Z.data();

            struct DisallowDirectVectorSubscripting {} X, Y, Z;
            (void)X; (void)Y; (void)Z;

            // TODO: Compute Z[i] = X[i * STRIDE_X] * Y[i * STRIDE_Y] using strided pointer arithmetic:
            //       Advance pX by STRIDE_X and pY by STRIDE_Y on each iteration.
            // --- YOUR CODE STARTS HERE ---

            // --- YOUR CODE ENDS HERE ---
        }

        bool p2_passed = true;
        for (int i = 0; i < OUT_N; ++i) {
            float expected = X[i * STRIDE_X] * Y[i * STRIDE_Y];
            if (abs(Z[i] - expected) > 1e-4f) {
                p2_passed = false;
                break;
            }
        }

        reportStatus("Problem 2: Strided Hadamard Element-Wise Product", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Strided Cosine Similarity
    //
    // Context: In vector databases (RAG embeddings), cosine similarity measures
    //          the angle between embedding vectors:
    //            cos(A, B) = (A . B) / (||A||_2 * ||B||_2)
    //
    // Task: Given two vectors `A` and `B` of size N=128:
    //       Compute their dot product, both L2 norms, and cosine similarity.
    // -------------------------------------------------------------------------
    {
        const int N = 128;
        vector<float> A(N), B(N);
        for (int i = 0; i < N; ++i) {
            A[i] = static_cast<float>((i % 7) - 3);
            B[i] = static_cast<float>((i % 5) - 2);
        }

        float cosine_sim = 0.0f;

        {
            const float* __restrict__ pA = A.data();
            const float* __restrict__ pB = B.data();

            struct DisallowDirectVectorSubscripting {} A, B;
            (void)A; (void)B;

            // TODO: Single-pass dual-vector cosine similarity using pointers pA and pB:
            //       Accumulate dot, norm_a, norm_b in local registers in one pass over memory,
            //       then calculate cosine_sim = dot / (sqrt(norm_a) * sqrt(norm_b)).
            // --- YOUR CODE STARTS HERE ---

            // --- YOUR CODE ENDS HERE ---
        }

        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (int i = 0; i < N; ++i) {
            dot += A[i] * B[i];
            norm_a += A[i] * A[i];
            norm_b += B[i] * B[i];
        }
        float expected_sim = dot / (sqrt(norm_a) * sqrt(norm_b));

        bool p3_passed = (abs(cosine_sim - expected_sim) < 1e-4f);
        reportStatus("Problem 3: Dual-Vector Cosine Similarity", p3_passed);
        if (p3_passed) passed++;
        total++;
    }

    cout << "\n=================================================================" << endl;
    cout << "--- SCORECARD ---" << endl;
    cout << "=================================================================" << endl;
    cout << "  Passed: " << passed << " / " << total << " tests." << endl;
    if (passed == total) {
        cout << "\033[1;32m  [STATUS] ALL " << total << " TESTS PASSED! \033[0m" << endl;
    } else {
        cout << "\033[1;31m  [STATUS] INCOMPLETE (" << (total - passed) << " tests failed) \033[0m" << endl;
    }
    cout << "=================================================================" << endl;

    return (passed == total) ? 0 : 1;
}
