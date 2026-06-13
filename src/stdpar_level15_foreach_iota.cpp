#include <iostream>
#include <vector>
#include <execution>
#include <algorithm>
#include <ranges>
#include <cmath>

// =========================================================================
// STDPAR LEVEL 15: Parallel std::for_each and C++20 std::views::iota
//
// CONCEPT CLASS:
// 1. std::views::iota(start, end)
//    This is a "lazy view". It represents a virtual sequence of numbers from 
//    'start' to 'end - 1'. Crucially, it does NOT allocate any memory. It
//    generates the numbers on-the-fly as we iterate through them.
// 
// 2. std::execution::par
//    This is an "Execution Policy". It tells the compiler's standard library:
//    "Execute this algorithm in parallel across multiple hardware cores."
//    - On GCC (CPU): It splits the loop across your CPU threads via Intel TBB.
//    - On NVC++ (GPU): It compiles your loop into a GPU kernel launch.
//
// 3. std::for_each
//    Replaces the traditional serial loop. When combined with 'std::execution::par'
//    and a range of indices from 'std::views::iota', it acts as an 
//    embarrassingly parallel grid launch where each thread processes its own index.
//
// YOUR TASK:
// 1. Initialize a vector of size N = 1000 with sequential float values from 
//    10.0f up to (10.0f + N - 1). You must use a parallel std::for_each over a 
//    std::views::iota range.
// 2. Multiply every element in your vector by 2.5f using a second parallel 
//    std::for_each loop.
// 3. Print the first 5 elements to verify your math!
// =========================================================================

int main() {
    std::cout << "--- stdpar Level 15: std::for_each & std::views::iota ---" << std::endl;

    int N = 1000;
    std::vector<float> dataset(N);

    // Create a virtual range of indices: [0, N)
    auto indices = std::views::iota(0, N);

    // TODO: 1. Initialize 'dataset' using std::for_each with std::execution::par.
    // Inside your lambda, each index 'i' should assign: dataset[i] = 10.0f + static_cast<float>(i);
    //
    // Hint:
    // std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i) {
    //     // Your code here
    // });
    
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i) {

        dataset[i] = 10.0f + static_cast<float> (i);

    } );




    // TODO: 2. Scale every element of 'dataset' by 2.5f using std::for_each with std::execution::par.
    //
    // Hint: You can iterate over the indices range, or run std::for_each directly 
    // over the vector's iterators: dataset.begin() to dataset.end()!
    
    std::for_each(std::execution::par, dataset.begin(), dataset.end(), [](float& val) {
        val *= 2.5f;       

    } );

    // Verification (Do not modify)
    std::cout << "\n--- Verification ---" << std::endl;
    std::cout << "First 5 elements of dataset:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "dataset[" << i << "]: " << dataset[i] << " (Expected: " << (10.0f + i) * 2.5f << ")" << std::endl;
    }

    return 0;
}