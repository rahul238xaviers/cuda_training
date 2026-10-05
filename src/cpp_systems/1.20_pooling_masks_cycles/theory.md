# Module Chapter: Pooling, Masks & Cycles

## Introduction
Covers max pooling index calculations, dropout masks, and pointer-chasing benchmarks.

## Relevance to Machine Learning & CUDA
Max-pooling requires saving active indexes to route gradients in backward passes. Dropout layers rely on pointer increments to apply randomized masks.

---

## Exercise Workbooks
This chapter features three separate practice workbooks matching problem complexity. Navigate to the `exercise/` folder to access them:
1. **Beginner Level (`exercise/beginner_workbook.cpp`):** Focuses on basic concepts, syntax, and foundational patterns.
2. **Intermediate Level (`exercise/intermediate_workbook.cpp`):** Introduces structural complexity, strides, and dimensional offsets.
3. **Champion Level (`exercise/champion_workbook.cpp`):** Covers complex edge cases, ML tensor scenarios, and performance-minded indexing patterns.

To compile and verify your exercises:
```bash
g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.20_pooling_masks_cycles_beginner
../../../output/1.20_pooling_masks_cycles_beginner
```
