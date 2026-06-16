# Module Chapter: Lifetimes, Ownership & Resize

## Introduction
Covers reference-counted ownership models, dynamic resizing, deep copying, and memory leak prevention.

## Relevance to Machine Learning & CUDA
Training models for millions of steps means even a small leak will quickly crash the GPU with Out-of-Memory (OOM) errors. Strict RAII lifetime management of GPU buffers is essential.

---

## Exercise Workbooks
This chapter features three separate practice workbooks matching problem complexity. Navigate to the `exercise/` folder to access them:
1. **Beginner Level (`exercise/beginner_workbook.cpp`):** Focuses on basic concepts, syntax, and foundational patterns.
2. **Intermediate Level (`exercise/intermediate_workbook.cpp`):** Introduces structural complexity, strides, and dimensional offsets.
3. **Champion Level (`exercise/champion_workbook.cpp`):** Covers complex edge cases, ML tensor scenarios, and performance-minded indexing patterns.

To compile and verify your exercises:
```bash
g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.12_lifetimes_ownership_beginner
../../../output/1.12_lifetimes_ownership_beginner
```
