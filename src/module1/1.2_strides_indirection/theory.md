# Module Chapter: Strides & Indirection

## Introduction
Focuses on stepping through arrays using custom step-sizes (strides) and accessing pointers indirectly.

## Relevance to Machine Learning & CUDA
Machine learning tensors are represented as flat memory arrays with specific strides (e.g. PyTorch layouts). Stride-based pointer math is heavily used in CUDA GEMM (Matrix Multiplication) and Conv2D kernels.

---

## Exercise Workbooks
This chapter features three separate practice workbooks matching problem complexity. Navigate to the `exercise/` folder to access them:
1. **Beginner Level (`exercise/beginner_workbook.cpp`):** Focuses on basic concepts, syntax, and foundational patterns.
2. **Intermediate Level (`exercise/intermediate_workbook.cpp`):** Introduces structural complexity, strides, and dimensional offsets.
3. **Champion Level (`exercise/champion_workbook.cpp`):** Covers complex edge cases, ML tensor scenarios, and performance-minded indexing patterns.

To compile and verify your exercises:
```bash
g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.2_strides_indirection_beginner
../../../output/1.2_strides_indirection_beginner
```
