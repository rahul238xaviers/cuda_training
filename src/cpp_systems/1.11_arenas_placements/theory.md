# Module Chapter: Arenas & Placements

## Introduction
Covers custom memory arenas, pool allocators, and placement new.

## Relevance to Machine Learning & CUDA
Frequently allocating and freeing GPU memory introduces massive driver overhead. High-performance ML frameworks (like PyTorch) use custom caching arena allocators to bypass CUDA runtime overhead.

---

## Exercise Workbooks
This chapter features three separate practice workbooks matching problem complexity. Navigate to the `exercise/` folder to access them:
1. **Beginner Level (`exercise/beginner_workbook.cpp`):** Focuses on basic concepts, syntax, and foundational patterns.
2. **Intermediate Level (`exercise/intermediate_workbook.cpp`):** Introduces structural complexity, strides, and dimensional offsets.
3. **Champion Level (`exercise/champion_workbook.cpp`):** Covers complex edge cases, ML tensor scenarios, and performance-minded indexing patterns.

To compile and verify your exercises:
```bash
g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.11_arenas_placements_beginner
../../../output/1.11_arenas_placements_beginner
```
