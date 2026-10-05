# Module Chapter: Attention & Quantization

## Introduction
Covers sliding window attention limits, similarity index mapping, and INT8 to FP32 weight reconstruction.

## Relevance to Machine Learning & CUDA
Attention scaling and quantized weight updates require strided pointer lookups inside CUDA cores.

---

## Exercise Workbooks
This chapter features three separate practice workbooks matching problem complexity. Navigate to the `exercise/` folder to access them:
1. **Beginner Level (`exercise/beginner_workbook.cpp`):** Focuses on basic concepts, syntax, and foundational patterns.
2. **Intermediate Level (`exercise/intermediate_workbook.cpp`):** Introduces structural complexity, strides, and dimensional offsets.
3. **Champion Level (`exercise/champion_workbook.cpp`):** Covers complex edge cases, ML tensor scenarios, and performance-minded indexing patterns.

To compile and verify your exercises:
```bash
g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.18_attention_quantization_beginner
../../../output/1.18_attention_quantization_beginner
```
