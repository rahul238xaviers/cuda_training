#include <iostream>
#include <vector>
#include <execution>
#include <algorithm>
#include <ranges>
#include <cmath>
#include <iomanip>

// =========================================================================
// FILE NAME: 16.1_level_stdpar_coord_mapping_workbook.cpp
//
// SUMMARY: A 10-problem advanced workbook dedicated entirely to mastering
//          multidimensional coordinate translation, transpositions, and
//          structured spatial indexing inside parallel std::transform pipelines.
//
// HOW TO COMPILE:
//   g++ -std=c++20 -O3 16.1_level_stdpar_coord_mapping_workbook.cpp -ltbb -o mapping_run
//   ./fused_run  (or ./mapping_run)
// =========================================================================

// --- Structural definitions for our multi-dimensional exercises ---

struct Point2D {
    int r; // Row index
    int c; // Column index
};

struct Point3D {
    int d; // Depth (Page)
    int h; // Height (Row)
    int w; // Width (Column)
};

struct Tensor4D {
    int b; // Batch index
    int c; // Channel index
    int h; // Height (Row)
    int w; // Width (Column)
};

// Verification status reporter
void reportStatus(const std::string& name, bool passed) {
    std::cout << "  " << std::left << std::setw(50) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- Level 16.1: Advanced Parallel Coordinate Mapping Workbook ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    const int N = 1000;
    std::vector<float> destination(N);
    bool allPassed = true;

    // -------------------------------------------------------------------------
    // PROBLEM 1: 2D Row-Major Flattening
    // Instruction: Take a vector of 'Point2D' coordinates. Flatten each coordinate
    //              into its standard 1D Row-Major index inside a matrix of width W = 64.
    //              Formula: index = r * W + c
    // -------------------------------------------------------------------------
    std::fill(destination.begin(), destination.end(), -1.0f);
    std::vector<Point2D> coords2D(N);
    int W1 = 64;
    for (int i = 0; i < N; ++i) {
        coords2D[i] = { i / W1, i % W1 };
    }

    // TODO: Write your parallel std::transform statement here:
    // std::transform(std::execution::par, coords2D.begin(), coords2D.end(), destination.begin(), ...);

    std::transform(std::execution::par, coords2D.begin(), coords2D.end(),destination.begin(), [W1](const Point2D& point2D){
        auto [r,c] = point2D;
       float index = static_cast<float>(r*W1 + c);
       return index;
        
    } );


    // Verification
    bool p1_passed = true;
    for (int i = 0; i < N; ++i) {
        float expected = static_cast<float>(coords2D[i].r * W1 + coords2D[i].c);
        if (std::abs(destination[i] - expected) > 1e-4) p1_passed = false;
    }
    reportStatus("Problem 1: 2D Row-Major Index Mapping", p1_passed);
    allPassed &= p1_passed;


    // -------------------------------------------------------------------------
    // PROBLEM 2: 2D Column-Major Flattening
    // Instruction: Some libraries (like Fortran, MATLAB, or CUDA's cuBLAS) store 
    //              matrices in Column-Major order (columns packed contiguously).
    //              Formula: index = c * H + r
    //              Take 'coords2D' and map each to its 1D Column-Major index for H = 32.
    // -------------------------------------------------------------------------
    std::fill(destination.begin(), destination.end(), -1.0f);
    int H2 = 32;
    // Regenerating coordinates to align with H=32 height boundaries
    for (int i = 0; i < N; ++i) {
        coords2D[i] = { i % H2, i / H2 };
    }

    // TODO: Write your parallel std::transform statement here:   

    std::transform(std::execution::par, coords2D.begin(), coords2D.end(), destination.begin(), [H2](Point2D& point2D){

        auto[r,c] = point2D;

        return static_cast<float>(H2 * c + r);

    });


    // Verification
    bool p2_passed = true;
    for (int i = 0; i < N; ++i) {
        float expected = static_cast<float>(coords2D[i].c * H2 + coords2D[i].r);
        if (std::abs(destination[i] - expected) > 1e-4) p2_passed = false;
    }
    reportStatus("Problem 2: 2D Column-Major Index Mapping", p2_passed);
    allPassed &= p2_passed;


    // -------------------------------------------------------------------------
    // PROBLEM 3: 3D Row-Major Volumetric Flattening
    // Instruction: Take a vector of 3D voxel coordinates ('Point3D'). Map each coordinate
    //              to its flat 1D index inside a 3D grid of Height H = 8 and Width W = 16.
    //              Formula: index = d * (H * W) + h * W + w
    // -------------------------------------------------------------------------
    std::fill(destination.begin(), destination.end(), -1.0f);
    std::vector<Point3D> coords3D(N);
    int H3 = 8;
    int W3 = 16;
    for (int i = 0; i < N; ++i) {
        int temp = i;
        int w = temp % W3;
        temp /= W3;
        int h = temp % H3;
        int d = temp / H3;
        coords3D[i] = { d, h, w };
    }

    // TODO: Write your parallel std::transform statement here:

    std::transform(std::execution::par, coords3D.begin(), coords3D.end(), destination.begin(), [H3, W3](const Point3D& point3D) {

        auto [d,h, w] = point3D;
        return static_cast<float>((d * (H3 * W3)) + (h * W3) + w);

    });


    // Verification
    bool p3_passed = true;
    for (int i = 0; i < N; ++i) {
        float expected = static_cast<float>(coords3D[i].d * (H3 * W3) + coords3D[i].h * W3 + coords3D[i].w);
        if (std::abs(destination[i] - expected) > 1e-4) p3_passed = false;
    }
    reportStatus("Problem 3: 3D Volumetric Spatial Mapping", p3_passed);
    allPassed &= p3_passed;


    // -------------------------------------------------------------------------
    // PROBLEM 4: 2D Flat Transposition Index Calculator
    // Instruction: In matrix transpositions, we map a source element (r, c) to (c, r).
    //              Given a list of original 1D Row-Major indices of a 3x5 matrix,
    //              calculate what its flat index would be inside the *transposed* 5x3 matrix.
    //              Steps inside lambda: 
    //                1. Decode original index to (r, c) using W_src = 5.
    //                2. Calculate new transposed flat index using W_dst = 3: index = c * W_dst + r.
    // -------------------------------------------------------------------------
    std::fill(destination.begin(), destination.end(), -1.0f);
    std::vector<int> originalFlatIndices(N);
    int W_src = 5;
    int W_dst = 3;
    for (int i = 0; i < N; ++i) {
        originalFlatIndices[i] = i % 15; // Keeps indices in range of 3x5 grid
    }

    // TODO: Write your parallel std::transform statement here:

    std::transform(std::execution::par,originalFlatIndices.begin(), originalFlatIndices.end(), destination.begin(), [W_src, W_dst](const int& val){

         int r = val % W_src;
         int c = val / W_src;
        return static_cast<float>(r * W_dst + c);
    } );

    // Verification
    bool p4_passed = true;
    for (int i = 0; i < N; ++i) {
        int orig_idx = originalFlatIndices[i];
        int r = orig_idx / W_src;
        int c = orig_idx % W_src;
        float expected = static_cast<float>(c * W_dst + r);
        if (std::abs(destination[i] - expected) > 1e-4) p4_passed = false;
    }
    reportStatus("Problem 4: Transposed Flat Index Translator", p4_passed);
    allPassed &= p4_passed;


    // -------------------------------------------------------------------------
    // PROBLEM 5: 3D Sub-Volume / Voxel Crop Offsets
    // Instruction: In scientific rendering, we crop small sub-volumes out of large grids.
    //              Given local coordinate (d, h, w) inside a crop block, calculate its
    //              global 1D index inside a larger parent volume of size H_parent = 128,
    //              W_parent = 128, starting at global offsets (d0=10, h0=20, w0=30).
    //              Formula: global_d = d + d0, global_h = h + h0, global_w = w + w0.
    //                       global_idx = global_d * (H_p * W_p) + global_h * W_p + global_w.
    // -------------------------------------------------------------------------
    std::fill(destination.begin(), destination.end(), -1.0f);
    int H_parent = 128;
    int W_parent = 128;
    int d0 = 10, h0 = 20, w0 = 30;

    // Crop coordinates (each bounded strictly inside a tiny crop space)
    for (int i = 0; i < N; ++i) {
        coords3D[i] = { i / 100, (i / 10) % 10, i % 10 }; 
    }

    // TODO: Write your parallel std::transform statement here:

    std::transform(std::execution::par, coords3D.begin(), coords3D.end(), destination.begin(), [d0, h0, w0, H_parent, W_parent](const Point3D point3D){

        auto [d, h, w] = point3D;
        int global_d = d0 + d;
        int global_h = h0 + h;
        int global_w = w0 + w;

        int global_idx = global_d * (H_parent * W_parent) +  global_h * W_parent + global_w;
        return global_idx;

    } );

    // Verification 
    bool p5_passed = true;
    for (int i = 0; i < N; ++i) {
        int gd = coords3D[i].d + d0;
        int gh = coords3D[i].h + h0;
        int gw = coords3D[i].w + w0;
        float expected = static_cast<float>(gd * (H_parent * W_parent) + gh * W_parent + gw);
        if (std::abs(destination[i] - expected) > 1e-4) p5_passed = false;
    }
    reportStatus("Problem 5: Sub-Grid Crop Volumetric Mapping", p5_passed);
    allPassed &= p5_passed;


    // -------------------------------------------------------------------------
    // PROBLEM 6: 2D Boundary Padding / Halo Cell Translation
    // Instruction: In graphics and stencil operations, we pad matrices with a 
    //              1-pixel "halo" border to run filter kernels safely.
    //              Take standard unpadded coordinates (r, c) inside a matrix of size HxW,
    //              and calculate their flat 1D indices inside a padded matrix of width (W + 2).
    //              Steps:
    //                1. Shift the coordinates past the pad: padded_r = r + 1, padded_c = c + 1
    //                2. Map to flat offset with padded width: padded_idx = padded_r * (W + 2) + padded_c
    // -------------------------------------------------------------------------
    std::fill(destination.begin(), destination.end(), -1.0f);
    int W6 = 32; // Unpadded Width
    for (int i = 0; i < N; ++i) {
        coords2D[i] = { i / W6, i % W6 };
    }

    // TODO: Write your parallel std::transform statement here:

    std::transform(std::execution::par, coords2D.begin(), coords2D.end(), destination.begin(), [W6] (const Point2D point2D ) {

        auto [r, c] = point2D;
        int padded_r = r + 1;
        int padded_c = c + 1;
        int padded_idx = padded_r * (W6 + 2) + padded_c;

        return padded_idx;
    });


    // Verification
    bool p6_passed = true;
    for (int i = 0; i < N; ++i) {
        int pr = coords2D[i].r + 1;
        int pc = coords2D[i].c + 1;
        float expected = static_cast<float>(pr * (W6 + 2) + pc);
        if (std::abs(destination[i] - expected) > 1e-4) p6_passed = false;
    }
    reportStatus("Problem 6: 2D Boundary Padding Offset Mapping", p6_passed);
    allPassed &= p6_passed;


    // -------------------------------------------------------------------------
    // PROBLEM 7: 4D Tensor Batch-Channel-Height-Width Flattening
    // Instruction: In PyTorch and Stable Diffusion, arrays are processed as 4D tensors
    //              of shape (Batch B, Channels C, Height H, Width W).
    //              Take 'Tensor4D' coordinates and flatten each into a 1D linear offset.
    //              Formula: index = b * (C * H * W) + c * (H * W) + h * W + w
    //              Dimensions: C = 3 (RGB), H = 16, W = 16.
    // -------------------------------------------------------------------------
    std::fill(destination.begin(), destination.end(), -1.0f);
    std::vector<Tensor4D> coords4D(N);
    int C7 = 3;
    int H7 = 16;
    int W7 = 16;
    for (int i = 0; i < N; ++i) {
        int temp = i;
        int w = temp % W7;
        temp /= W7;
        int h = temp % H7;
        temp /= H7;
        int c = temp % C7; 
        int b = temp / C7;
        coords4D[i] = { b, c, h, w };
    }

    // TODO: Write your parallel std::transform statement here:

    std::transform(std::execution::par, coords4D.begin() , coords4D.end(), destination.begin(), [C7, H7, W7](const Tensor4D tensor4D) {

        auto [b, c, h, w] = tensor4D;
        int index = b * (C7 * H7 * W7) + c * (H7 * W7) + h * W7 + w;

        return index;

    });

    // Verification
    bool p7_passed = true;
    for (int i = 0; i < N; ++i) {
        float expected = static_cast<float>(
            coords4D[i].b * (C7 * H7 * W7) +
            coords4D[i].c * (H7 * W7) +
            coords4D[i].h * W7 +
            coords4D[i].w
        );
        if (std::abs(destination[i] - expected) > 1e-4) p7_passed = false;
    }
    reportStatus("Problem 7: 4D Tensor Batch-Channel Flattening", p7_passed);
    allPassed &= p7_passed;


    // -------------------------------------------------------------------------
    // PROBLEM 8: Row-Major to Column-Major Transposition Mapper
    // Instruction: Translate a raw Row-Major index directly into its matching 
    //              Column-Major index in a matrix of size H = 10, W = 100.
    //              Steps:
    //                1. Decode the input index to row-major coords: r = idx / W, c = idx % W.
    //                2. Map those coordinates to column-major index: col_major_idx = c * H + r.
    // -------------------------------------------------------------------------
    std::fill(destination.begin(), destination.end(), -1.0f);
    std::vector<int> rowMajorIndices(N);
    int H8 = 10;
    int W8 = 100;
    for (int i = 0; i < N; ++i) {
        rowMajorIndices[i] = i % 1000; // Constrain within 10x100 grid bounds
    }

    // TODO: Write your parallel std::transform statement here:

    std::transform(std::execution::par, rowMajorIndices.begin(), rowMajorIndices.end(), destination.begin(), [H8, W8](const int idx){

        int r = idx / W8;
        int c = idx % W8;

        int  col_major_idx = c * H8 + r;
        return col_major_idx;

    });

    // Verification
    bool p8_passed = true;
    for (int i = 0; i < N; ++i) {
        int idx = rowMajorIndices[i];
        int r = idx / W8;
        int c = idx % W8;
        float expected = static_cast<float>(c * H8 + r);
        if (std::abs(destination[i] - expected) > 1e-4) p8_passed = false;
    }
    reportStatus("Problem 8: Row-Major to Column-Major Translator", p8_passed);
    allPassed &= p8_passed;


    // -------------------------------------------------------------------------
    // PROBLEM 9: 2D Block-Cyclic Matrix Partitioning Map
    // Instruction: Distributed solvers partition matrices into blocks of size Br x Bc.
    //              Given standard coordinates (r, c), map each element to its physical
    //              Local Block Coordinator index: (block_row, block_col).
    //              Formulas: block_row = r / Br, block_col = c / Bc.
    //              Calculate a 1D block index using a grid of blocks with block_width = 8.
    //              Formula: block_index = block_row * block_width + block_col.
    //              Block dimensions: Br = 8, Bc = 8. Block Width = 8.
    // -------------------------------------------------------------------------
    std::fill(destination.begin(), destination.end(), -1.0f);
    int Br = 8, Bc = 8;
    int block_width = 8;
    for (int i = 0; i < N; ++i) {
        coords2D[i] = { i / 64, i % 64 };
    }

    // TODO: Write your parallel std::transform statement here:

    std::transform(std::execution::par, coords2D.begin(), coords2D.end(), destination.begin(), [Br, Bc, block_width](const Point2D point2D){

        auto [r, c] = point2D;

        int block_row = r / Br;
        int block_col = c / Bc;
        int block_index = block_row * block_width + block_col;
        return block_index;

    });


    // Verification
    bool p9_passed = true;
    for (int i = 0; i < N; ++i) {
        int brow = coords2D[i].r / Br;
        int bcol = coords2D[i].c / Bc;
        float expected = static_cast<float>(brow * block_width + bcol);
        if (std::abs(destination[i] - expected) > 1e-4) p9_passed = false;
    }
    reportStatus("Problem 9: Distributed Block-Cyclic Coordinate Map", p9_passed);
    allPassed &= p9_passed;


    // -------------------------------------------------------------------------
    // PROBLEM 10: 3D Spherical to Cartesian Voxel Index Mapping
    // Instruction: Modern simulation engines map spherical sensor parameters to Cartesian volumes.
    //              You are given coordinates representing: radius_slice r, theta_slice t, phi_slice p.
    //              Convert these logical coordinates into a Cartesian index using custom widths:
    //              Formula: cartesian_x = r, cartesian_y = t, cartesian_z = p
    //              Compute the flat voxel index inside a cartesian workspace of Height Y = 16, Width X = 32.
    //              Formula: voxel_idx = cartesian_z * (Y * X) + cartesian_y * X + cartesian_x.
    //              Dimensions: Y = 16, X = 32.
    // -------------------------------------------------------------------------
    std::fill(destination.begin(), destination.end(), -1.0f);
    int Y10 = 16;
    int X10 = 32;
    std::vector<Point3D> sphericalCoords(N);
    for (int i = 0; i < N; ++i) {
        sphericalCoords[i] = { i % 32, (i / 32) % 16, i / 512 }; // r, t, p
    }

    // TODO: Write your parallel std::transform statement here:

    std::transform(std::execution::par, sphericalCoords.begin(), sphericalCoords.end(), destination.begin(), [Y10, X10](const Point3D point3D){

        auto[r, t, p] = point3D;
        int voxel_idx = p * (Y10 * X10) + t * X10 + r;
        return voxel_idx;
    });


    // Verification
    bool p10_passed = true;
    for (int i = 0; i < N; ++i) {
        int cx = sphericalCoords[i].d; // r maps to x
        int cy = sphericalCoords[i].h; // t maps to y
        int cz = sphericalCoords[i].w; // p maps to z
        float expected = static_cast<float>(cz * (Y10 * X10) + cy * X10 + cx);
        if (std::abs(destination[i] - expected) > 1e-4) p10_passed = false;
    }
    reportStatus("Problem 10: Spherical-to-Cartesian Voxel Map", p10_passed);
    allPassed &= p10_passed;


    // -------------------------------------------------------------------------
    // FINAL GRADE
    // -------------------------------------------------------------------------
    std::cout << "=================================================================" << std::endl;
    if (allPassed) {
        std::cout << "\033[1;32m      CONGRATULATIONS! ALL 10 CO-ORDINATE SCENARIOS PASSED! \033[0m" << std::endl;
        std::cout << "      You have officially championed GPGPU Coordinate Mapping!" << std::endl;
    } else {
        std::cout << "\033[1;31m      WORKBOOK STATUS: INCOMPLETE (Some mapping tests failed) \033[0m" << std::endl;
        std::cout << "      Inspect your loop operators and re-verify your mapping logic." << std::endl;
    }
    std::cout << "=================================================================" << std::endl;

    return 0;
}