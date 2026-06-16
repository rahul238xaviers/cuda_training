# Module Chapter: Subgrids, Padding & Diagonals

## Introduction
Covers sub-volume slicing, boundary padding offsets (halos), and diagonal elements indexing.

## Relevance to Machine Learning & CUDA
Padding is required in convolutional layers to maintain spatial dimensions. Halo indexing is also crucial in GPGPU stencil computations where adjacent thread blocks share boundary cells.

---

## Exercise Workbooks
This chapter features three separate practice workbooks matching problem complexity. Navigate to the `exercise/` folder to access them:
1. **Beginner Level (`exercise/beginner_workbook.cpp`):** Focuses on basic concepts, syntax, and foundational patterns.
2. **Intermediate Level (`exercise/intermediate_workbook.cpp`):** Introduces structural complexity, strides, and dimensional offsets.
3. **Champion Level (`exercise/champion_workbook.cpp`):** Covers complex edge cases, ML tensor scenarios, and performance-minded indexing patterns.

To compile and verify your exercises:
```bash
g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.7_subgrids_padding_beginner
../../../output/1.7_subgrids_padding_beginner
```
