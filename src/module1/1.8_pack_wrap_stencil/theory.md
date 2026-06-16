# Module Chapter: Pack, Wrap, Stencil & Crops

## Introduction
Covers upper triangular packing, circular buffer indexes, cyclic partitioning, and stencil offsets.

## Relevance to Machine Learning & CUDA
Upper-triangular indices are used to pack symmetric similarity matrices in attention layers. Block-cyclic partitioning is the standard way to distribute tensor weights across multi-GPU setups.

---

## Exercise Workbooks
This chapter features three separate practice workbooks matching problem complexity. Navigate to the `exercise/` folder to access them:
1. **Beginner Level (`exercise/beginner_workbook.cpp`):** Focuses on basic concepts, syntax, and foundational patterns.
2. **Intermediate Level (`exercise/intermediate_workbook.cpp`):** Introduces structural complexity, strides, and dimensional offsets.
3. **Champion Level (`exercise/champion_workbook.cpp`):** Covers complex edge cases, ML tensor scenarios, and performance-minded indexing patterns.

To compile and verify your exercises:
```bash
g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.8_pack_wrap_stencil_beginner
../../../output/1.8_pack_wrap_stencil_beginner
```
