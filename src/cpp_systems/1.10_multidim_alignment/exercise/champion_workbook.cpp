#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <cstdint>

// =========================================================================
// CHAMPION WORKBOOK: Multi-Dim & Alignment
//
// Module: 1.10 - Dynamic Heap Allocation & Memory Lifecycle
// Level:  Champion / High-Performance Systems Engineer
//
// Focus: 3D Pitched volume subgrid slicers, AoS vs SoA vectorization
//        memory bandwidth benchmark, and dynamic 4D strided tensor views.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.10_champion
//   ../../../output/1.10_champion
// =========================================================================

void reportStatus(const std::string& name, bool passed, double throughputGBs = -1.0) {
    std::cout << "  " << std::left << std::setw(60) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m";
        if (throughputGBs > 0.0) {
            std::cout << " (" << std::fixed << std::setprecision(2) << throughputGBs << " GB/s)";
        }
        std::cout << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: Multi-Dim & Alignment (Champion Masterclass) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Complete 3D Pitched Volume Allocator & Subvolume Slicer
    //
    // Context: In `cudaMalloc3D` and volumetric imaging, raw memory is padded
    //          both per-row (pitch) and per-slice (slice_pitch) so all subvolume
    //          offsets align with hardware cache lines (256 bytes = 64 floats).
    //
    // Task: Implement `PitchedVolume3D` for dimensions WIDTH=50, HEIGHT=20, DEPTH=8 floats:
    //       1. Compute PITCH_FLOATS = (WIDTH + 63) & ~63 = 64 floats (256 bytes)
    //       2. Compute SLICE_FLOATS = PITCH_FLOATS * HEIGHT = 1280 floats
    //       3. Allocate flat buffer and populate with f(d, h, w) = d*10000 + h*100 + w.
    //       4. Extract a subvolume [CROP_D=4, CROP_H=8, CROP_W=16] at (d0=2, h0=4, w0=10)
    //          into a compact contiguous buffer `subvolume_out`.
    // -------------------------------------------------------------------------
    {
        const int WIDTH = 50, HEIGHT = 20, DEPTH = 8;
        const int PITCH_FLOATS = (WIDTH + 63) & ~63; // 64
        const int SLICE_FLOATS = PITCH_FLOATS * HEIGHT; // 1280
        const size_t total_elements = DEPTH * SLICE_FLOATS;

        std::vector<float> pitched_pool(total_elements, 0.0f);
        for (int d = 0; d < DEPTH; ++d) {
            for (int h = 0; h < HEIGHT; ++h) {
                for (int w = 0; w < WIDTH; ++w) {
                    pitched_pool[d * SLICE_FLOATS + h * PITCH_FLOATS + w] = static_cast<float>(d * 10000 + h * 100 + w);
                }
            }
        }

        const int d0 = 2, h0 = 4, w0 = 10;
        const int CROP_D = 4, CROP_H = 8, CROP_W = 16;
        std::vector<float> subvolume_out(CROP_D * CROP_H * CROP_W, -1.0f);

        // TODO: Extract the subvolume into subvolume_out in contiguous order.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p1_passed = true;
        for (int cd = 0; cd < CROP_D && p1_passed; ++cd) {
            for (int ch = 0; ch < CROP_H && p1_passed; ++ch) {
                for (int cw = 0; cw < CROP_W; ++cw) {
                    float expected = static_cast<float>((d0 + cd) * 10000 + (h0 + ch) * 100 + (w0 + cw));
                    size_t out_idx = cd * (CROP_H * CROP_W) + ch * CROP_W + cw;
                    if (subvolume_out[out_idx] != expected) {
                        p1_passed = false;
                        break;
                    }
                }
            }
        }

        reportStatus("Problem 1: 3D Pitched Volume Allocator & Subvolume Slicer", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: AoS vs SoA Memory Bandwidth Vectorization Benchmark
    //
    // Context: When a kernel only reads 2 components (x and w) out of a 4-component
    //          struct (x, y, z, w), AoS wastes 50% of the fetched cache lines loading
    //          unused y and z components. In SoA, only the relevant arrays are loaded.
    //
    // Task: Process N=262,144 particles (1 MB):
    //       Compute sum = sum(x[i] * w[i]) using Structure-of-Arrays (SoA).
    //       Benchmark memory throughput in GB/s!
    // -------------------------------------------------------------------------
    {
        const int N = 262144;
        std::vector<float> soa_x(N, 1.5f);
        std::vector<float> soa_w(N, 2.0f);

        float total_dot = 0.0f;
        auto start = std::chrono::high_resolution_clock::now();

        // TODO: Compute dot product of soa_x and soa_w.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double bytes_moved = 2.0 * N * sizeof(float);
        double throughput = (bytes_moved / elapsed_sec) / 1e9;

        float expected_dot = N * (1.5f * 2.0f);
        bool p2_passed = (std::abs(total_dot - expected_dot) < 1e-1f);

        reportStatus("Problem 2: SoA Vectorized Memory Bandwidth Benchmark", p2_passed, throughput);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Dynamic 4D Strided Tensor View Descriptor
    //
    // Context: In deep learning frameworks (PyTorch `at::Tensor`), a tensor view
    //          consists of a data pointer, a shape array, and a stride array.
    //          Formula for element (b, c, h, w):
    //            offset = b * stride[0] + c * stride[1] + h * stride[2] + w * stride[3]
    //
    // Task: Implement a 4D tensor view over a contiguous pool of 1024 floats:
    //       Shape: [B=2, C=4, H=8, W=16]
    //       Strides: [512, 128, 16, 1]
    //       Populate `view_sum` by summing all elements using dynamic strides.
    // -------------------------------------------------------------------------
    {
        const int B = 2, C = 4, H = 8, W = 16;
        const size_t total_elements = B * C * H * W; // 1024
        std::vector<float> pool(total_elements);
        float expected_sum = 0.0f;
        for (size_t i = 0; i < total_elements; ++i) {
            pool[i] = static_cast<float>(i * 0.1f);
            expected_sum += pool[i];
        }

        const int strides[4] = {512, 128, 16, 1};
        float view_sum = 0.0f;

        // TODO: Sum all elements in pool by traversing 4 nested loops (b, c, h, w)
        // using the dynamic strides array.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = (std::abs(view_sum - expected_sum) < 1e-2f);
        reportStatus("Problem 3: Dynamic 4D Strided Tensor View Descriptor", p3_passed);
        if (p3_passed) passed++;
        total++;
    }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "--- SCORECARD ---" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  Passed: " << passed << " / " << total << " tests." << std::endl;
    if (passed == total) {
        std::cout << "\033[1;32m  [STATUS] ALL " << total << " TESTS PASSED! \033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m  [STATUS] INCOMPLETE (" << (total - passed) << " tests failed) \033[0m" << std::endl;
    }
    std::cout << "=================================================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
