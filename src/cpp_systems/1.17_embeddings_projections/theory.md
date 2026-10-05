# Module Chapter: Embeddings & Projections

## Introduction
Simulates embedding lookups and head key-value offset projection indexing.

## Relevance to Machine Learning & CUDA
Embedding layers convert discrete token indexes into continuous vectors. Embedding lookups and multi-head queries map directly to parallel stride indexing on the GPU.

---

## Exercise Workbooks
This chapter features three separate practice workbooks matching problem complexity. Navigate to the `exercise/` folder to access them:
1. **Beginner Level (`exercise/beginner_workbook.cpp`):** Focuses on basic concepts, syntax, and foundational patterns.
2. **Intermediate Level (`exercise/intermediate_workbook.cpp`):** Introduces structural complexity, strides, and dimensional offsets.
3. **Champion Level (`exercise/champion_workbook.cpp`):** Covers complex edge cases, ML tensor scenarios, and performance-minded indexing patterns.

To compile and verify your exercises:
```bash
g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.17_embeddings_projections_beginner
../../../output/1.17_embeddings_projections_beginner
```
