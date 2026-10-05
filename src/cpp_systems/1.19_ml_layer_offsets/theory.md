# Module Chapter: ML Layer Offsets

## Introduction
Covers weight descent updates, 2D bias offsets, GroupNorm mapping, and LayerNorm pointers.

## Relevance to Machine Learning & CUDA
Optimizers, layer normalization, and gradient updates rely heavily on parallel offset sweeps. Iterating across normalization channels requires precise pointer striding.

---

## Exercise Workbooks
This chapter features three separate practice workbooks matching problem complexity. Navigate to the `exercise/` folder to access them:
1. **Beginner Level (`exercise/beginner_workbook.cpp`):** Focuses on basic concepts, syntax, and foundational patterns.
2. **Intermediate Level (`exercise/intermediate_workbook.cpp`):** Introduces structural complexity, strides, and dimensional offsets.
3. **Champion Level (`exercise/champion_workbook.cpp`):** Covers complex edge cases, ML tensor scenarios, and performance-minded indexing patterns.

To compile and verify your exercises:
```bash
g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.19_ml_layer_offsets_beginner
../../../output/1.19_ml_layer_offsets_beginner
```
