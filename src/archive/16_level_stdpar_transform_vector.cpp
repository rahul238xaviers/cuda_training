#include <iostream>
#include <vector>
#include <execution>
#include <algorithm>
#include <ranges>

// =========================================================================
// FILE NAME: 16_level_stdpar_transform_vector.cpp
//
// SUMMARY: Take an array of input values, compute a non-destructive math 
//          operation, and stream the results out to a separate buffer.
// =========================================================================

int main() {
    std::cout << "--- stdpar Level 16: Parallel Array Transformations ---" << std::endl;

    int N = 1000;
    std::vector<float> source(N);
    std::vector<float> destination(N);

    // 1. Initialize the source vector sequentially using our index iota view
    auto indices = std::views::iota(0, N);
    float* src_ptr = source.data();
    std::for_each(std::execution::par, indices.begin(), indices.end(), [src_ptr](int i) {
        src_ptr[i] = static_cast<float>(i + 1); // Fills with 1.0f, 2.0f, 3.0f...
    });

    // =========================================================================
    // TODO: Task: Use std::transform with std::execution::par to process 
    //             every element in 'source', calculate its mathematical reciprocal
    //             (1.0f / val), and stream the output directly into 'destination'.
    //
    // Constraints & Reminders:
    // - Pass the parallel execution policy as the first parameter.
    // - Supply the boundary iterators for the input space ('source.begin()' and 'source.end()').
    // - Supply the starting iterator for the target output space ('destination.begin()').
    // - Since we are only reading the input, your lambda parameter should accept
    //   it as a read-only variable: (float val) or (const float& val).
    // - Crucially, your lambda MUST return the calculated float result!
    // =========================================================================
    
    // ---> WRITE YOUR PARALLEL std::transform OPERATION HERE <---

    std::transform(std::execution::par ,source.begin(), source.end(), destination.begin(), [](float val){
        return 1.0f/val;
    } );


    // =========================================================================
    // Verification Engine (Do not alter)
    // =========================================================================
    std::cout << "\n--- Verification ---" << std::endl;
    std::cout << "Checking streamed pipeline output bounds:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "source[" << i << "]: " << source[i] 
                  << " -> destination[" << i << "]: " << destination[i] 
                  << " (Expected: " << (1.0f / source[i]) << ")" << std::endl;
    }

    return 0;
}