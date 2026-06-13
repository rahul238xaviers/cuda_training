#include <iostream>
#include <vector>
#include <execution>
#include <algorithm>
#include <ranges>
#include <cmath>
#include <iomanip>

// =========================================================================
// FILE NAME: 16_level_stdpar_transform_workbook.cpp
//
// SUMMARY: A 10-problem comprehensive workbook for C++ Standard Parallelism
//          (stdpar) transformations. This file will help you cement your
//          understanding of standard parallel mapping pipelines.
//
// HOW TO COMPILE:
//   g++ -std=c++20 -O3 16_level_stdpar_transform_workbook.cpp -ltbb -o workbook_run
//   ./workbook_run
// =========================================================================

// Structured coordinates for Problem 10
struct GridCoord {
    int row;
    int col;
};

// Verification status reporter
void reportStatus(const std::string& name, bool passed) {
    std::cout << "  " << std::left << std::setw(45) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- Level 16.5: C++ Parallel std::transform 10-Problem Workbook ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    const int N = 1000;
    
    // Allocating testing resources
    std::vector<float> sourceA(N);
    std::vector<float> sourceB(N);
    std::vector<float> destination(N);

    // Initialize source arrays with sequential base patterns
    for (int i = 0; i < N; i++) {
        sourceA[i] = static_cast<float>(i);
        sourceB[i] = static_cast<float>(N - i);
        destination[i] = -999.0f; // Clear output
    }

    bool allPassed = true;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Standard Scaling Map ($y_i = x_i \times 3.5$)
    // Instruction: Take elements from 'sourceA', scale by 3.5f, write to 'destination'.
    // -------------------------------------------------------------------------
    float scaleFactor = 3.5f;
    
    // TODO: Write your std::transform pipeline here:
    // std::transform(...);

    std::transform(std::execution::par, sourceA.begin(), sourceA.end(), destination.begin(), [](float& val){
        return val * 3.5f;
    });

    // Verification
    bool p1_passed = true;
    for (int i = 0; i < N; i++) {
        if (std::abs(destination[i] - (sourceA[i] * 3.5f)) > 1e-4) p1_passed = false;
    }
    reportStatus("Problem 1: Vector Scalar Scaling", p1_passed);
    allPassed &= p1_passed;


    // -------------------------------------------------------------------------
    // PROBLEM 2: Mathematical Reciprocal ($y_i = 1.0 / x_i$)
    // Instruction: Map elements of 'sourceA' to their reciprocal. (Handle division by 0
    // by assigning 0.0f if the input value is exactly 0.0f).
    // -------------------------------------------------------------------------
    std::fill(destination.begin(), destination.end(), -999.0f); // Reset

    // TODO: Write your std::transform pipeline here:
    // std::transform(...);

    std::transform(std::execution::par, sourceA.begin(), sourceA.end(), destination.begin(), [](float& x_i) {
        if(x_i == 0.0f){
            return 0.0f;
        }
        else{
            return (1.0f /x_i);
        }
    } );

    // Verification
    bool p2_passed = true;
    for (int i = 0; i < N; i++) {
        float expected = (sourceA[i] == 0.0f) ? 0.0f : (1.0f / sourceA[i]);
        if (std::abs(destination[i] - expected) > 1e-4) p2_passed = false;
    }
    reportStatus("Problem 2: Non-Zero Reciprocal Mapping", p2_passed);
    allPassed &= p2_passed;


    // -------------------------------------------------------------------------
    // PROBLEM 3: Celsius to Fahrenheit Conversion ($F = C \times 1.8 + 32$)
    // Instruction: Assume 'sourceA' represents Celsius values. Write a parallel
    // transformation to map them to Fahrenheit in 'destination'.
    // -------------------------------------------------------------------------
    std::fill(destination.begin(), destination.end(), -999.0f); // Reset

    // TODO: Write your std::transform pipeline here:
    // std::transform(...);

    std::transform(std::execution::par, sourceA.begin(), sourceA.end(), destination.begin(), [](float val){
        return ((val * 1.8f) + 32.0f);
    });
    // Verification
    bool p3_passed = true;
    for (int i = 0; i < N; i++) {
        float expected = sourceA[i] * 1.8f + 32.0f;
        if (std::abs(destination[i] - expected) > 1e-4) p3_passed = false;
    }
    reportStatus("Problem 3: Thermal Conversion Map", p3_passed);
    allPassed &= p3_passed;


    // -------------------------------------------------------------------------
    // PROBLEM 4: Binary Array Difference ($C[i] = |A[i] - B[i]|$)
    // Instruction: Use standard parallel binary transform to calculate the absolute
    // difference between elements of 'sourceA' and 'sourceB', and stream to 'destination'.
    //
    // Hint: Standard binary transform signature:
    // std::transform(Policy, First1, Last1, First2, D_first, BinaryOp);
    // -------------------------------------------------------------------------
    std::fill(destination.begin(), destination.end(), -999.0f); // Reset

    // TODO: Write your std::transform pipeline here:
    // std::transform(...);

    std::transform(std::execution::par, sourceA.begin(), sourceA.end(), sourceB.begin(), destination.begin(), [](float a, float b){
        return std::abs(a - b);
    });

    // Verification
    bool p4_passed = true;
    for (int i = 0; i < N; i++) {
        float expected = std::abs(sourceA[i] - sourceB[i]);
        if (std::abs(destination[i] - expected) > 1e-4) p4_passed = false;
    }
    reportStatus("Problem 4: Binary Absolute Difference", p4_passed);
    allPassed &= p4_passed;


    // -------------------------------------------------------------------------
    // PROBLEM 5: Vector Magnitude Calculations ($M[i] = \sqrt{A[i]^2 + B[i]^2}$)
    // Instruction: Treat 'sourceA' as coordinate X and 'sourceB' as coordinate Y.
    // Stream the physical hypotenuse length into 'destination' using parallel binary transform.
    // -------------------------------------------------------------------------
    std::fill(destination.begin(), destination.end(), -999.0f); // Reset

    // TODO: Write your std::transform pipeline here:
    // std::transform(...);

    std::transform(std::execution::par, sourceA.begin(), sourceA.end(), sourceB.begin(), destination.begin(), [](float a, float b){

        float norm_calc = std::sqrt(a * a + b * b);
        return norm_calc;
    });


    // Verification
    bool p5_passed = true;
    for (int i = 0; i < N; i++) {
        float expected = std::sqrt(sourceA[i]*sourceA[i] + sourceB[i]*sourceB[i]);
        if (std::abs(destination[i] - expected) > 1e-4) p5_passed = false;
    }
    reportStatus("Problem 5: Euclidian Coordinate Norms", p5_passed);
    allPassed &= p5_passed;


    // -------------------------------------------------------------------------
    // PROBLEM 6: Rectified Linear Unit (ReLU) Activation ($\text{ReLU}(x) = \max(0.0f, x)$)
    // Instruction: AI kernels use activations. Subtract 500.0f from 'sourceA' (meaning
    // some elements will be negative), compute the parallel ReLU activation, and write to 'destination'.
    // -------------------------------------------------------------------------
    std::fill(destination.begin(), destination.end(), -999.0f); // Reset

    // TODO: Write your std::transform pipeline here:
    // std::transform(...);

    std::transform(std::execution::par, sourceA.begin(), sourceA.end(), destination.begin(), [](float val){
       return std::max(0.0f, val - 500.0f);
    });

    // Verification
    bool p6_passed = true;
    for (int i = 0; i < N; i++) {
        float val = sourceA[i] - 500.0f;
        float expected = std::max(0.0f, val);
        if (std::abs(destination[i] - expected) > 1e-4) p6_passed = false;
    }
    reportStatus("Problem 6: Neural Network ReLU Mapping", p6_passed);
    allPassed &= p6_passed;


    // -------------------------------------------------------------------------
    // PROBLEM 7: Sigmoid Activation function ($S(x) = \frac{1.0f}{1.0f + e^{-x}}$)
    // Instruction: Map normalized float elements (scale sourceA down by dividing by N,
    // so values sit in [0, 1]) through the Sigmoid activation and write to 'destination'.
    // -------------------------------------------------------------------------
    std::fill(destination.begin(), destination.end(), -999.0f); // Reset

    // TODO: Write your std::transform pipeline here:
    // std::transform(...);

    std::transform(std::execution::par, sourceA.begin(), sourceA.end(), destination.begin(), [N](float val){
        float normX = val/static_cast<float> (N);
        float sigmoidValue = 1.0f / (1.0f + std::exp(-normX));
       return sigmoidValue;
    });

    // Verification
    bool p7_passed = true;
    for (int i = 0; i < N; i++) {
        float norm_x = sourceA[i] / static_cast<float>(N);
        float expected = 1.0f / (1.0f + std::exp(-norm_x));
        if (std::abs(destination[i] - expected) > 1e-4) p7_passed = false;
    }
    reportStatus("Problem 7: Logistic Sigmoid Activations", p7_passed);
    allPassed &= p7_passed;


    // -------------------------------------------------------------------------
    // PROBLEM 8: Clamping Signal values to Range $[100.0f, 200.0f]$
    // Instruction: Restrict values of 'sourceA' to sit strictly inside 100.0f and 200.0f.
    // If val < 100.0f set to 100.0f; if val > 200.0f set to 200.0f. Write to 'destination'.
    // -------------------------------------------------------------------------
    std::fill(destination.begin(), destination.end(), -999.0f); // Reset

    // TODO: Write your std::transform pipeline here:
    std::transform(std::execution::par, sourceA.begin(), sourceA.end(), destination.begin(), [N](float val){

        return std::clamp(val, 100.0f, 200.0f);
      
    });

    // Verification
    bool p8_passed = true;
    for (int i = 0; i < N; i++) {
        float expected = std::clamp(sourceA[i], 100.0f, 200.0f);
        if (std::abs(destination[i] - expected) > 1e-4) p8_passed = false;
    }
    reportStatus("Problem 8: Hard Threshold Signal Clamping", p8_passed);
    allPassed &= p8_passed;


    // -------------------------------------------------------------------------
    // PROBLEM 9: Boolean Mask Generator
    // Instruction: Stream true (1.0f) to 'destination' if 'sourceA' element is odd,
    // and false (0.0f) if the element is even.
    // -------------------------------------------------------------------------
    std::fill(destination.begin(), destination.end(), -999.0f); // Reset

    // TODO: Write your std::transform pipeline here:
    // std::transform(...);

        std::transform(std::execution::par, sourceA.begin(), sourceA.end(), destination.begin(), [](float val){

        return fmodf(val, 2.0f) == 0.0f ? 0.0f : 1.0f;
      
    });

    // Verification
    bool p9_passed = true;
    for (int i = 0; i < N; i++) {
        float expected = (static_cast<int>(sourceA[i]) % 2 != 0) ? 1.0f : 0.0f;
        if (std::abs(destination[i] - expected) > 1e-4) p9_passed = false;
    }
    reportStatus("Problem 9: Modular Boolean Condition Masking", p9_passed);
    allPassed &= p9_passed;


    // -------------------------------------------------------------------------
    // PROBLEM 10: 2D Coordinate Row-Major Flatten Mapping 
    // Instruction: You are given a vector of 'GridCoord' inputs. For each coordinate, 
    // calculate its flat 1D index inside a matrix of width W = 32. 
    // Stream the resulting indices into 'destination'.
    // -------------------------------------------------------------------------
    std::fill(destination.begin(), destination.end(), -999.0f); // Reset
    std::vector<GridCoord> coords(N);
    for (int i = 0; i < N; i++) {
        coords[i] = { i / 32, i % 32 };
    }
    int matrixWidth = 32; 

    // TODO: Write your std::transform pipeline over 'coords' here:
    // Note: Input iterator is over 'coords.begin()' to 'coords.end()', output to 'destination.begin()'.
    // std::transform(...);

    std::transform(std::execution::par, coords.begin(), coords.end(), destination.begin(),[matrixWidth](const GridCoord& currCoord){

        float index = static_cast<float>(currCoord.row * matrixWidth + currCoord.col );

        return index;

    });

    // Verification
    bool p10_passed = true;
    for (int i = 0; i < N; i++) {
        float expected = static_cast<float>(coords[i].row * matrixWidth + coords[i].col);
        if (std::abs(destination[i] - expected) > 1e-4) p10_passed = false;
    }
    reportStatus("Problem 10: Multi-dimensional Coord Transpose Mapping", p10_passed);
    allPassed &= p10_passed;


    // -------------------------------------------------------------------------
    // FINAL GRADE
    // -------------------------------------------------------------------------
    std::cout << "=================================================================" << std::endl;
    if (allPassed) {
        std::cout << "\033[1;32m      CONGRATULATIONS! ALL 10 CONCURRENT SCENARIOS PASSED! \033[0m" << std::endl;
        std::cout << "      You have officially championed the std::transform API!" << std::endl;
    } else {
        std::cout << "\033[1;31m      WORKBOOK STATUS: INCOMPLETE (Some test cases failed) \033[0m" << std::endl;
        std::cout << "      Inspect your loop operators and re-verify your mapping logic." << std::endl;
    }
    std::cout << "=================================================================" << std::endl;

    return 0;
}