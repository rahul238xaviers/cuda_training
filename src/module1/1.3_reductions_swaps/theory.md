# Module Chapter: Array Reductions & Swaps

## Introduction
Focuses on aggregating values (sum, max) and swapping memory contents using pointer traversal.

## Relevance to Machine Learning & CUDA
Reductions are critical for algorithms like Softmax, LayerNorm, and Attention mechanisms. In CUDA, block-level reductions require stepping pointers across threads in shared memory.

---

## Exercise Workbooks
This chapter features three separate practice workbooks matching problem complexity. Navigate to the `exercise/` folder to access them:
1. **Beginner Level (`exercise/beginner_workbook.cpp`):** Focuses on basic concepts, syntax, and foundational patterns.
2. **Intermediate Level (`exercise/intermediate_workbook.cpp`):** Introduces structural complexity, strides, and dimensional offsets.
3. **Champion Level (`exercise/champion_workbook.cpp`):** Covers complex edge cases, ML tensor scenarios, and performance-minded indexing patterns.

To compile and verify your exercises:
```bash
g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.3_reductions_swaps_beginner
../../../output/1.3_reductions_swaps_beginner
```
