# Module Chapter: Null, Bounds & Alignment Check

## Introduction
Covers runtime pointer validation, boundary checking, and alignment offset detection.

## Relevance to Machine Learning & CUDA
An out-of-bounds pointer dereference inside a CUDA thread causes an Illegal Memory Access error, crashing the host program. Runtime checks are vital to debugging GPU kernels.

---

## Exercise Workbooks
This chapter features three separate practice workbooks matching problem complexity. Navigate to the `exercise/` folder to access them:
1. **Beginner Level (`exercise/beginner_workbook.cpp`):** Focuses on basic concepts, syntax, and foundational patterns.
2. **Intermediate Level (`exercise/intermediate_workbook.cpp`):** Introduces structural complexity, strides, and dimensional offsets.
3. **Champion Level (`exercise/champion_workbook.cpp`):** Covers complex edge cases, ML tensor scenarios, and performance-minded indexing patterns.

To compile and verify your exercises:
```bash
g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.13_null_bounds_checks_beginner
../../../output/1.13_null_bounds_checks_beginner
```
