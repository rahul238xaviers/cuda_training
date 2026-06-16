# Module Chapter: Multi-array & Strided Ops

## Introduction
Covers operations running across multiple separate arrays using custom stride intervals.

## Relevance to Machine Learning & CUDA
Vector operations like element-wise addition, scaling, and dot products (SAXPY/DAXPY) require streaming data from multiple independent buffers concurrently using offset calculations.

---

## Exercise Workbooks
This chapter features three separate practice workbooks matching problem complexity. Navigate to the `exercise/` folder to access them:
1. **Beginner Level (`exercise/beginner_workbook.cpp`):** Focuses on basic concepts, syntax, and foundational patterns.
2. **Intermediate Level (`exercise/intermediate_workbook.cpp`):** Introduces structural complexity, strides, and dimensional offsets.
3. **Champion Level (`exercise/champion_workbook.cpp`):** Covers complex edge cases, ML tensor scenarios, and performance-minded indexing patterns.

To compile and verify your exercises:
```bash
g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.4_multi_array_strided_beginner
../../../output/1.4_multi_array_strided_beginner
```
