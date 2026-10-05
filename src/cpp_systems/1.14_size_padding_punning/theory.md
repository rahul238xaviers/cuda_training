# Module Chapter: Size, Padding & Punning

## Introduction
Covers structure padding verification, type punning, and direct bitwise casting.

## Relevance to Machine Learning & CUDA
Deep learning models are optimized using low-precision datatypes (e.g., FP16, BF16, FP8). Type punning and bit-shifts are used to convert raw binary states to floating-point values.

---

## Exercise Workbooks
This chapter features three separate practice workbooks matching problem complexity. Navigate to the `exercise/` folder to access them:
1. **Beginner Level (`exercise/beginner_workbook.cpp`):** Focuses on basic concepts, syntax, and foundational patterns.
2. **Intermediate Level (`exercise/intermediate_workbook.cpp`):** Introduces structural complexity, strides, and dimensional offsets.
3. **Champion Level (`exercise/champion_workbook.cpp`):** Covers complex edge cases, ML tensor scenarios, and performance-minded indexing patterns.

To compile and verify your exercises:
```bash
g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.14_size_padding_punning_beginner
../../../output/1.14_size_padding_punning_beginner
```
