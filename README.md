# CUDA C++ for Machine Learning & LLM GPU Kernel Engineering

[![CUDA 12+](https://img.shields.io/badge/CUDA-12.x%20%7C%2011.8+-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![C++17/C++20](https://img.shields.io/badge/C%2B%2B-17%20%2F%2020-00599C?logo=c%2B%2B)](https://en.cppreference.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A self-contained, exercise-driven curriculum engineered to take developers from C++ pointer arithmetic and memory layouts to authoring production-grade CUDA kernels for Large Language Model (LLM) training and inference engines.

---

## 📖 Curriculum Overview

The curriculum is divided into **7 Core Modules** spanning **42 subtopics** and **126 hands-on workbooks** across three difficulty tiers: **Beginner**, **Intermediate**, and **Champion**.

| Module | Focus Area | Stages | Workbooks | Status |
| :--- | :--- | :---: | :---: | :---: |
| [**Module 1**](src/module1/) | **C++ Fundamentals Refresh**: Pointer arithmetic, striding, tensor flattening, memory arenas. | 20 | 60 | ✅ Available |
| [**Module 2**](src/module2/) | **Modern C++ for GPU Readiness**: Device lambdas, SFINAE/traits, RAII stream guards, tensor views, host pipelines. | 5 | 15 | ✅ Available |
| [**Module 3**](src/module3/) | **CUDA Architecture & Execution Model**: Threads, registers, warp shuffles, shared memory banks, occupancy. | 4 | 12 | ✅ Available |
| [**Module 4**](src/module4/) | **Memory Hierarchy & Tiling**: Global coalescing, read-only cache, 2D shared-memory matrix tiling. | 2 | 6 | ✅ Available |
| [**Module 5**](src/module5/) | **Parallel Primitives & Numerical Stability**: Two-pass reductions, atomics, Log-Sum-Exp softmax. | 3 | 9 | ✅ Available |
| [**Module 6**](src/module6/) | **Precision, Intrinsics & Tensor Cores**: `__nv_bfloat16`, vectorized types, WMMA Tensor Core matrix math. | 2 | 6 | ✅ Available |
| [**Module 7**](src/module7/) | **The Master 24 LLM Training Kernels Suite**: Full PyTorch-equivalent CUDA kernel suite for Transformer training. | 6 | 18 | ✅ Available |
| **Total** | **End-to-End GPU Kernel Mastery** | **42** | **126** | **100% Ready** |

---

## 🛠️ Prerequisites & Setup

### Requirements
- **NVIDIA GPU**: Turing (compute capability `sm_75`), Ampere (`sm_80`, `sm_86`), Ada Lovelace (`sm_89`), or Hopper (`sm_90`).
- **CUDA Toolkit**: 11.8 or 12.x installed with `nvcc` in your `PATH`.
- **C++ Compiler**: GCC 9+ / Clang with C++17 and C++20 support.
- **Operating System**: Linux (Ubuntu 20.04/22.04/24.04 recommended) or Windows via WSL 2.

### Environment Verification
```bash
# Verify NVIDIA GPU driver
nvidia-smi

# Verify CUDA Compiler
nvcc --version
```

---

## 🚀 How to Use This Repository

Every subtopic in this repository follows a clean, standardized 3-part layout:
```
src/moduleX/<topic_name>/
├── theory.md               # Detailed conceptual guide with math, diagrams & hardware internals
├── exercise/               # Template workbooks with // TODO gaps and unit tests
│   ├── beginner_workbook.cu (or .cpp)
│   ├── intermediate_workbook.cu
│   └── champion_workbook.cu
└── solution/               # Your personal workspace (git-ignored: only .gitkeep is tracked)
```

### 1. Step-by-Step Learning Workflow

1. **Read the Theory**: Open `theory.md` in the subtopic folder to understand the algorithmic and hardware requirements.
2. **Open the Exercise**: Start with `exercise/beginner_workbook.cu`. Each workbook includes 5 targeted exercises and an integrated test harness.
3. **Fill the `// TODO` Gaps**: Implement the requested functions and kernels.
4. **Compile & Test**: Compile the workbook using `nvcc` (or `g++` for Module 1).
5. **Verify the Scorecard**: The test harness prints a pass/fail summary:
   ```
   [Test 1: Kernel Output] PASSED
   [Test 2: Edge Cases]    PASSED
   [Test 3: GPU Memory]    PASSED
   [Test 4: Performance]  PASSED
   [Test 5: Numerical]    PASSED
   Passed: 5 / 5 tests.
   ```
6. **Progress to Higher Tiers**: Once Beginner passes 5/5, move to `intermediate_workbook.cu`, then challenge yourself with `champion_workbook.cu`.

---

## 💻 Compilation Commands

### 1. Compiling a Single CUDA Workbook (`.cu`)
```bash
# Single workbook compilation with NVCC
nvcc -O3 -std=c++17 -arch=sm_75 --extended-lambda \
  src/module2/2.1_lambdas_function_objects/exercise/beginner_workbook.cu \
  -o /tmp/workbook_test && /tmp/workbook_test
```
*(Note: Replace `-arch=sm_75` with your GPU architecture, e.g. `sm_80` for A100, `sm_86` for RTX 30-series, `sm_89` for RTX 40-series).*

### 2. Compiling a Single C++ Fundamentals Workbook (`.cpp`)
```bash
g++ -std=c++20 -O3 \
  src/module1/1.1_basic_offsets/exercise/beginner_workbook.cpp \
  -o /tmp/cpp_test && /tmp/cpp_test
```

### 3. Automated Test Runners
We provide two automated test runners to evaluate workbooks:

```bash
# Test all C++ Fundamentals workbooks (Module 1)
bash src/module1/compile_and_run_all.sh

# Test all CUDA workbooks (Modules 2 to 7)
bash src/compile_and_run_cuda.sh all

# Test a specific module or stage
bash src/compile_and_run_cuda.sh module2
bash src/compile_and_run_cuda.sh 7.6_flash_attention
```

---

## 📊 Actionable Exercise Tracker

Copy and paste this checklist into your personal GitHub Issue, pull request, Notion page, or Markdown notes to track your progress through all 126 exercises.

```markdown
# 🚀 CUDA ML Training Curriculum Progress Tracker

## Module 1: C++ Fundamentals Refresh (60 Exercises)
### Section 1: Pointer Arithmetic & Offsets
- [ ] 1.1 Basic Offset & Increment — Beginner
- [ ] 1.1 Basic Offset & Increment — Intermediate
- [ ] 1.1 Basic Offset & Increment — Champion
- [ ] 1.2 Strides & Indirection — Beginner
- [ ] 1.2 Strides & Indirection — Intermediate
- [ ] 1.2 Strides & Indirection — Champion
- [ ] 1.3 Array Reductions & Swaps — Beginner
- [ ] 1.3 Array Reductions & Swaps — Intermediate
- [ ] 1.3 Array Reductions & Swaps — Champion
- [ ] 1.4 Multi-array & Strided Ops — Beginner
- [ ] 1.4 Multi-array & Strided Ops — Intermediate
- [ ] 1.4 Multi-array & Strided Ops — Champion

### Section 2: 2D/3D Tensor Flattening & Strides
- [ ] 1.5 Row/Col-Major Indexing — Beginner
- [ ] 1.5 Row/Col-Major Indexing — Intermediate
- [ ] 1.5 Row/Col-Major Indexing — Champion
- [ ] 1.6 3D/4D Permute & Flatten — Beginner
- [ ] 1.6 3D/4D Permute & Flatten — Intermediate
- [ ] 1.6 3D/4D Permute & Flatten — Champion
- [ ] 1.7 Subgrids, Padding & Diagonals — Beginner
- [ ] 1.7 Subgrids, Padding & Diagonals — Intermediate
- [ ] 1.7 Subgrids, Padding & Diagonals — Champion
- [ ] 1.8 Pack, Wrap, Stencil & Crops — Beginner
- [ ] 1.8 Pack, Wrap, Stencil & Crops — Intermediate
- [ ] 1.8 Pack, Wrap, Stencil & Crops — Champion

### Section 3: Memory Management & Arenas
- [ ] 1.9 Single & Array Allocations — Beginner
- [ ] 1.9 Single & Array Allocations — Intermediate
- [ ] 1.9 Single & Array Allocations — Champion
- [ ] 1.10 Multi-Dim & Alignment — Beginner
- [ ] 1.10 Multi-Dim & Alignment — Intermediate
- [ ] 1.10 Multi-Dim & Alignment — Champion
- [ ] 1.11 Arenas & Placements — Beginner
- [ ] 1.11 Arenas & Placements — Intermediate
- [ ] 1.11 Arenas & Placements — Champion
- [ ] 1.12 Lifetimes, Ownership & Resize — Beginner
- [ ] 1.12 Lifetimes, Ownership & Resize — Intermediate
- [ ] 1.12 Lifetimes, Ownership & Resize — Champion

### Section 4: Advanced Indexing & Bit Manipulation
- [ ] 1.13 Bitwise Flags & Masks — Beginner
- [ ] 1.13 Bitwise Flags & Masks — Intermediate
- [ ] 1.13 Bitwise Flags & Masks — Champion
- [ ] 1.14 Circular Buffers & Ring Queues — Beginner
- [ ] 1.14 Circular Buffers & Ring Queues — Intermediate
- [ ] 1.14 Circular Buffers & Ring Queues — Champion
- [ ] 1.15 Sparse Representations — Beginner
- [ ] 1.15 Sparse Representations — Intermediate
- [ ] 1.15 Sparse Representations — Champion
- [ ] 1.16 Custom Allocators — Beginner
- [ ] 1.16 Custom Allocators — Intermediate
- [ ] 1.16 Custom Allocators — Champion

### Section 5: Machine Learning GPU Prep & Simulations
- [ ] 1.17 Embeddings & Projections — Beginner
- [ ] 1.17 Embeddings & Projections — Intermediate
- [ ] 1.17 Embeddings & Projections — Champion
- [ ] 1.18 Attention & Quantization — Beginner
- [ ] 1.18 Attention & Quantization — Intermediate
- [ ] 1.18 Attention & Quantization — Champion
- [ ] 1.19 ML Layer Offsets — Beginner
- [ ] 1.19 ML Layer Offsets — Intermediate
- [ ] 1.19 ML Layer Offsets — Champion
- [ ] 1.20 Pooling, Masks & Cycles — Beginner
- [ ] 1.20 Pooling, Masks & Cycles — Intermediate
- [ ] 1.20 Pooling, Masks & Cycles — Champion

---

## Module 2: Modern C++ for GPU Readiness (15 Exercises)
- [ ] 2.1 Lambdas & Function Objects — Beginner
- [ ] 2.1 Lambdas & Function Objects — Intermediate
- [ ] 2.1 Lambdas & Function Objects — Champion
- [ ] 2.2 Templates & Type Traits — Beginner
- [ ] 2.2 Templates & Type Traits — Intermediate
- [ ] 2.2 Templates & Type Traits — Champion
- [ ] 2.3 RAII & Stream Managers — Beginner
- [ ] 2.3 RAII & Stream Managers — Intermediate
- [ ] 2.3 RAII & Stream Managers — Champion
- [ ] 2.4 Move Semantics & Tensor Views — Beginner
- [ ] 2.4 Move Semantics & Tensor Views — Intermediate
- [ ] 2.4 Move Semantics & Tensor Views — Champion
- [ ] 2.5 Host Parallel Execution — Beginner
- [ ] 2.5 Host Parallel Execution — Intermediate
- [ ] 2.5 Host Parallel Execution — Champion

---

## Module 3: CUDA Architecture & Execution Model (12 Exercises)
- [ ] 3.1 Thread Registers & Local Spilling — Beginner
- [ ] 3.1 Thread Registers & Local Spilling — Intermediate
- [ ] 3.1 Thread Registers & Local Spilling — Champion
- [ ] 3.2 Warps, Divergence & Shuffles — Beginner
- [ ] 3.2 Warps, Divergence & Shuffles — Intermediate
- [ ] 3.2 Warps, Divergence & Shuffles — Champion
- [ ] 3.3 Blocks, Shared Memory & Bank Conflicts — Beginner
- [ ] 3.3 Blocks, Shared Memory & Bank Conflicts — Intermediate
- [ ] 3.3 Blocks, Shared Memory & Bank Conflicts — Champion
- [ ] 3.4 Grids, Occupancy & Grid-Stride Loops — Beginner
- [ ] 3.4 Grids, Occupancy & Grid-Stride Loops — Intermediate
- [ ] 3.4 Grids, Occupancy & Grid-Stride Loops — Champion

---

## Module 4: Memory Hierarchy & Tiling (6 Exercises)
- [ ] 4.1 Global Memory Coalescing & Read-Only Cache — Beginner
- [ ] 4.1 Global Memory Coalescing & Read-Only Cache — Intermediate
- [ ] 4.1 Global Memory Coalescing & Read-Only Cache — Champion
- [ ] 4.2 Shared Memory Matrix Tiling — Beginner
- [ ] 4.2 Shared Memory Matrix Tiling — Intermediate
- [ ] 4.2 Shared Memory Matrix Tiling — Champion

---

## Module 5: Parallel Primitives & Numerical Stability (9 Exercises)
- [ ] 5.1 Warp & Block Reductions — Beginner
- [ ] 5.1 Warp & Block Reductions — Intermediate
- [ ] 5.1 Warp & Block Reductions — Champion
- [ ] 5.2 Global Atomic Operations — Beginner
- [ ] 5.2 Global Atomic Operations — Intermediate
- [ ] 5.2 Global Atomic Operations — Champion
- [ ] 5.3 Numerically Stable Reductions (Softmax & Loss) — Beginner
- [ ] 5.3 Numerically Stable Reductions (Softmax & Loss) — Intermediate
- [ ] 5.3 Numerically Stable Reductions (Softmax & Loss) — Champion

---

## Module 6: Precision, Intrinsics & Tensor Cores (6 Exercises)
- [ ] 6.1 BFloat16 (`__nv_bfloat16`) Arithmetic & Intrinsics — Beginner
- [ ] 6.1 BFloat16 (`__nv_bfloat16`) Arithmetic & Intrinsics — Intermediate
- [ ] 6.1 BFloat16 (`__nv_bfloat16`) Arithmetic & Intrinsics — Champion
- [ ] 6.2 Tensor Cores & WMMA API — Beginner
- [ ] 6.2 Tensor Cores & WMMA API — Intermediate
- [ ] 6.2 Tensor Cores & WMMA API — Champion

---

## Module 7: Master 24 LLM Training Kernels Suite (18 Exercises)
- [ ] 7.1 Elementwise & Reshaping (`reshape_3d`, `reshape_4d`, `residual_add`, `swiglu_fwd/bwd`) — Beginner
- [ ] 7.1 Elementwise & Reshaping (`reshape_3d`, `reshape_4d`, `residual_add`, `swiglu_fwd/bwd`) — Intermediate
- [ ] 7.1 Elementwise & Reshaping (`reshape_3d`, `reshape_4d`, `residual_add`, `swiglu_fwd/bwd`) — Champion
- [ ] 7.2 Token Embedding Lookup (`embedding_forward`) — Beginner
- [ ] 7.2 Token Embedding Lookup (`embedding_forward`) — Intermediate
- [ ] 7.2 Token Embedding Lookup (`embedding_forward`) — Champion
- [ ] 7.3 Normalization & RoPE (`rms_norm_fwd/bwd`, `rope_fwd/bwd`, `fused_add_norm_fwd/bwd`) — Beginner
- [ ] 7.3 Normalization & RoPE (`rms_norm_fwd/bwd`, `rope_fwd/bwd`, `fused_add_norm_fwd/bwd`) — Intermediate
- [ ] 7.3 Normalization & RoPE (`rms_norm_fwd/bwd`, `rope_fwd/bwd`, `fused_add_norm_fwd/bwd`) — Champion
- [ ] 7.4 Loss Functions & AdamW Optimizer (`cross_entropy`, `compute_loss`, `adamw_step`) — Beginner
- [ ] 7.4 Loss Functions & AdamW Optimizer (`cross_entropy`, `compute_loss`, `adamw_step`) — Intermediate
- [ ] 7.4 Loss Functions & AdamW Optimizer (`cross_entropy`, `compute_loss`, `adamw_step`) — Champion
- [ ] 7.5 GEMM Projections (`gemm_proj`, `gemm_bf16`, `gemm_gqa`, `gemm_ffn`, `fused_swiglu_gemm`) — Beginner
- [ ] 7.5 GEMM Projections (`gemm_proj`, `gemm_bf16`, `gemm_gqa`, `gemm_ffn`, `fused_swiglu_gemm`) — Intermediate
- [ ] 7.5 GEMM Projections (`gemm_proj`, `gemm_bf16`, `gemm_gqa`, `gemm_ffn`, `fused_swiglu_gemm`) — Champion
- [ ] 7.6 Tiled FlashAttention-2 Suite (`flash_attn_fwd`, `fused_attn_bwd`) — Beginner
- [ ] 7.6 Tiled FlashAttention-2 Suite (`flash_attn_fwd`, `fused_attn_bwd`) — Intermediate
- [ ] 7.6 Tiled FlashAttention-2 Suite (`flash_attn_fwd`, `fused_attn_bwd`) — Champion
```

---

## 🛡️ License & Contributing

Distributed under the MIT License. Contributions, additional architecture optimizations, and bug reports are welcome via Pull Requests.
