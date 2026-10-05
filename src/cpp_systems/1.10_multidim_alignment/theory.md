# Module Chapter: Multi-Dim & Alignment

## Introduction
Covers dynamic 2D array allocations (pointer-of-pointers), flat 2D arrays, and memory alignment.

## Relevance to Machine Learning & CUDA
Misaligned memory addresses lead to inefficient GPU memory transactions. CUDA utilizes coalesced memory loads which require dynamic tensors to be aligned to specific boundaries (e.g. 128 bytes).

---

## Exercise Workbooks
This chapter features three separate practice workbooks matching problem complexity. Navigate to the `exercise/` folder to access them:
1. **Beginner Level (`exercise/beginner_workbook.cpp`):** Focuses on basic concepts, syntax, and foundational patterns.
2. **Intermediate Level (`exercise/intermediate_workbook.cpp`):** Introduces structural complexity, strides, and dimensional offsets.
3. **Champion Level (`exercise/champion_workbook.cpp`):** Covers complex edge cases, ML tensor scenarios, and performance-minded indexing patterns.

To compile and verify your exercises:
```bash
g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.10_multidim_alignment_beginner
../../../output/1.10_multidim_alignment_beginner
```
