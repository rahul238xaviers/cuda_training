# Module Chapter: Type Casts & Double Pointers

## Introduction
Covers pointer typecasting, void pointers representation, and double-pointer indirection.

## Relevance to Machine Learning & CUDA
CUDA API functions (like `cudaMalloc`) accept `void**` parameters to output allocations. Reinterpret casts are commonly used to cast general GPU buffers to typed float arrays.

---

## Exercise Workbooks
This chapter features three separate practice workbooks matching problem complexity. Navigate to the `exercise/` folder to access them:
1. **Beginner Level (`exercise/beginner_workbook.cpp`):** Focuses on basic concepts, syntax, and foundational patterns.
2. **Intermediate Level (`exercise/intermediate_workbook.cpp`):** Introduces structural complexity, strides, and dimensional offsets.
3. **Champion Level (`exercise/champion_workbook.cpp`):** Covers complex edge cases, ML tensor scenarios, and performance-minded indexing patterns.

To compile and verify your exercises:
```bash
g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.15_casts_double_pointers_beginner
../../../output/1.15_casts_double_pointers_beginner
```
