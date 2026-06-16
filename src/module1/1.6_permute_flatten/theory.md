# Module Chapter: 3D/4D Permute & Flatten

## Introduction
Covers indexing math for 3D and 4D tensors, layout transposition, and dimensional permutation.

## Relevance to Machine Learning & CUDA
Transformers and CNNs frequently reshape and permute tensor axes (e.g. transposing Batch-Seq-Head-Dim to Batch-Head-Seq-Dim). Knowing how to compute these permuted flat indices is crucial for attention kernels.

---

## Exercise Workbooks
This chapter features three separate practice workbooks matching problem complexity. Navigate to the `exercise/` folder to access them:
1. **Beginner Level (`exercise/beginner_workbook.cpp`):** Focuses on basic concepts, syntax, and foundational patterns.
2. **Intermediate Level (`exercise/intermediate_workbook.cpp`):** Introduces structural complexity, strides, and dimensional offsets.
3. **Champion Level (`exercise/champion_workbook.cpp`):** Covers complex edge cases, ML tensor scenarios, and performance-minded indexing patterns.

To compile and verify your exercises:
```bash
g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.6_permute_flatten_beginner
../../../output/1.6_permute_flatten_beginner
```
