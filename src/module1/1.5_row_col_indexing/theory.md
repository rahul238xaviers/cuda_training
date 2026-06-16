# Module Chapter: Row/Col-Major Indexing

## Introduction
Covers mapping 2D coordinate space (row, col) to flat 1D linear indexes and decoding indexes back to coordinates.

## Relevance to Machine Learning & CUDA
GPUs process multi-dimensional tensors using flat linear memory. In a CUDA kernel, threads map their coordinates (x, y) to flat indexes using row-major layouts to load and store data.

---

## Exercise Workbooks
This chapter features three separate practice workbooks matching problem complexity. Navigate to the `exercise/` folder to access them:
1. **Beginner Level (`exercise/beginner_workbook.cpp`):** Focuses on basic concepts, syntax, and foundational patterns.
2. **Intermediate Level (`exercise/intermediate_workbook.cpp`):** Introduces structural complexity, strides, and dimensional offsets.
3. **Champion Level (`exercise/champion_workbook.cpp`):** Covers complex edge cases, ML tensor scenarios, and performance-minded indexing patterns.

To compile and verify your exercises:
```bash
g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.5_row_col_indexing_beginner
../../../output/1.5_row_col_indexing_beginner
```
