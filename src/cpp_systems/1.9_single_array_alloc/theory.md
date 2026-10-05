# Module Chapter: Single & Array Allocations

## Introduction
Focuses on stack vs heap differences, allocating dynamically using `new` and `malloc`, and clean deallocation.

## Relevance to Machine Learning & CUDA
ML datasets and neural networks are too large for stack frames. C++ and CUDA code relies on dynamic heap allocations (such as `cudaMalloc` and `cudaMallocManaged`) to store activations and weights.

---

## Exercise Workbooks
This chapter features three separate practice workbooks matching problem complexity. Navigate to the `exercise/` folder to access them:
1. **Beginner Level (`exercise/beginner_workbook.cpp`):** Focuses on basic concepts, syntax, and foundational patterns.
2. **Intermediate Level (`exercise/intermediate_workbook.cpp`):** Introduces structural complexity, strides, and dimensional offsets.
3. **Champion Level (`exercise/champion_workbook.cpp`):** Covers complex edge cases, ML tensor scenarios, and performance-minded indexing patterns.

To compile and verify your exercises:
```bash
g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.9_single_array_alloc_beginner
../../../output/1.9_single_array_alloc_beginner
```
