# 📚 High-Performance Systems & CUDA LLM Engine: The Digital Book

Welcome to your interactive curriculum and personal digital textbook. This repository bridges modern C++ systems programming, GPU architecture, and high-performance LLM training/inference engines from first principles.

---

## 🔖 Reading Ribbon & Active Bookmark

```text
========================================================================================
CURRENT STATION : Volume 1, Chapter 1.2 — Strides & Pointer Indirection
STATUS          : Ready to Begin
PREVIOUS        : Chapter 1.1: Basic Offsets & Pointer Arithmetic (100% Passed 🏆)
ACTIVE LAB      : playground.cpp
========================================================================================
```

### 📊 Overall Course Progress
```text
Phase 1: Systems Foundations  [██░░░░░░░░░░░░░░░░░░]  5% (1/20 Chapters)
Phase 2: CUDA Architecture    [░░░░░░░░░░░░░░░░░░░░]  0% (0/3 Modules)
Phase 3: 22 LLM CUDA Kernels  [░░░░░░░░░░░░░░░░░░░░]  0% (0/22 Kernels)
Total Verified Exercises      : 10 / 10 Passed (100% on Chapter 1.1)
```

---

## 🧭 Master Table of Contents

- [Volume 1: C++ Low-Level Memory Foundations](#volume-1-c-low-level-memory-foundations)
  - [Chapter 1.1: Basic Offsets & Pointer Arithmetic](#chapter-11-basic-memory-offsets--pointer-arithmetic) `[PASSED 🏆]`
  - [Chapter 1.2: Strides & Pointer Indirection](#chapter-12-strides--pointer-indirection) `[CURRENT 📍]`
  - [Chapter 1.3: Reductions & Swaps](#chapter-13-reductions--swaps)
  - [Chapter 1.4: Multi-Array Strided Traversal](#chapter-14-multi-array-strided-traversal)
  - [Chapter 1.5: Row/Column Matrix Indexing](#chapter-15-rowcolumn-matrix-indexing)
  - [Chapter 1.6: Permute & Flatten Operations](#chapter-16-permute--flatten-operations)
  - [Chapter 1.7: Subgrids & Padding](#chapter-17-subgrids--padding)
  - [Chapter 1.8: Packing, Wrapping & Stencils](#chapter-18-packing-wrapping--stencils)
  - [Chapter 1.9: Single Array Allocation Schemes](#chapter-19-single-array-allocation-schemes)
  - [Chapter 1.10: Multidimensional Alignment](#chapter-110-multidimensional-alignment)
  - [Chapter 1.11: Arenas & Custom Placements](#chapter-111-arenas--custom-placements)
  - [Chapter 1.12: Lifetimes & Ownership Semantics](#chapter-112-lifetimes--ownership-semantics)
  - [Chapter 1.13: Null & Bounds Verification](#chapter-113-null--bounds-verification)
  - [Chapter 1.14: Size, Padding & Type Punning](#chapter-114-size-padding--type-punning)
  - [Chapter 1.15: Casts & Double Pointers](#chapter-115-casts--double-pointers)
  - [Chapter 1.16: Pointer Relations & Copying](#chapter-116-pointer-relations--copying)
  - [Chapter 1.17: Embeddings & Projections](#chapter-117-embeddings--projections)
  - [Chapter 1.18: Attention & Quantization Packing](#chapter-118-attention--quantization-packing)
  - [Chapter 1.19: ML Layer Offsets & Layouts](#chapter-119-ml-layer-offsets--layouts)
  - [Chapter 1.20: Pooling, Masks & Cyclic Buffers](#chapter-120-pooling-masks--cyclic-buffers)
- [Volume 2: CUDA Architecture & Execution Model](#volume-2-cuda-architecture--execution-model)
  - [Module 2: CUDA Threading Hierarchy (Threads, Warps, Blocks, Grids)](#module-2-cuda-execution-model)
  - [Module 3: CUDA Memory Hierarchy (Global, Shared SRAM, Registers)](#module-3-cuda-memory-hierarchy)
  - [Module 4: High-Performance Parallel Primitives (GEMM, Reductions, Softmax)](#module-4-parallel-primitives)
- [Volume 3: The 22 Production LLM CUDA Kernels](#volume-3-the-22-production-llm-cuda-kernels)

---

# Volume 1: C++ Low-Level Memory Foundations

### Chapter 1.1: Basic Memory Offsets & Pointer Arithmetic
> **Synopsis**: Understand the physical nature of memory addresses in 64-bit hardware. Master pointer scaling, raw byte interpretation, cache-line alignment math, vectorized memory loads, and custom bump memory pools.
* **Key Mental Models**: Pointer scaling laws (`ptr + N`), physical RAM vs. type lenses, the warehouse forklift cache-line model, bitwise alignment (`(addr + 255) & ~255`), double-buffering pointer ping-pong.
* **Study & Practice**:
  - 📖 [Read Chapter Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.1_basic_offsets/theory.md)
  - ⚡ [Quick Revision Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.1_basic_offsets/cheat_sheet.md)
  - 🧪 [Active Practice Playground](file:///Users/rahulkumar/dev/cuda_training/playground.cpp)
* **Workbook Tiers**:
  - `[x]` **Beginner**: 4/4 Passed (Strided stepping, binary weight deserialization, in-place reversal, ring buffer wrapping).
  - `[x]` **Intermediate**: 3/3 Passed (Pitched 2D memory, Float4 128-bit streaming @ 46+ GB/s, bitwise cache padding).
  - `[x]` **Champion**: 3/3 Passed (Aligned bump allocator, double-buffered pipeline, 16x16 tiled transpose @ 10.70 GB/s).
* **Status**: `[COMPLETED 🏆]` (Completed on 2026-10-01)

---

### Chapter 1.2: Strides & Pointer Indirection
> **Synopsis**: How multidimensional tensors (2D, 3D, 4D) map onto flat 1D memory. Learn how strides define non-contiguous tensor views, transposed slices, and pointer indirection arrays without copying data.
* **Hardware Motivation**: PyTorch tensor operations like `.view()`, `.transpose()`, and slicing do not copy data—they merely change the strides. Understanding stride math is mandatory for writing GEMM and Attention kernels.
* **Study & Practice**:
  - 📖 [Read Chapter Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.2_strides_indirection/theory.md)
  - 🧪 [Practice in Playground](file:///Users/rahulkumar/dev/cuda_training/playground.cpp)
  - 📝 [Beginner Workbook](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.2_strides_indirection/exercise/beginner_workbook.cpp)
  - 📝 [Intermediate Workbook](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.2_strides_indirection/exercise/intermediate_workbook.cpp)
  - 📝 [Champion Workbook](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.2_strides_indirection/exercise/champion_workbook.cpp)
* **Status**: `[CURRENT STATION 📍]`

---

### Chapter 1.3: Reductions & Swaps
> **Synopsis**: Algorithmic pointer patterns for aggregating data across contiguous streams (sum, max, argmax) and zero-cost memory permutations via pointer swaps.
* **Hardware Motivation**: Foundation for GPU Warp Reductions and Softmax normalization in LLM attention layers.
* **Status**: `[QUEUED ⏳]`

---

### Chapter 1.4: Multi-Array Strided Traversal
> **Synopsis**: Coordinating simultaneous pointer movement across multiple independent arrays with mismatched strides and data types.
* **Hardware Motivation**: Multi-input tensor elementwise operations (Residual Add, Fused Add-Norm, Bias Addition).
* **Status**: `[QUEUED ⏳]`

---

### Chapter 1.5: Row/Column Matrix Indexing
> **Synopsis**: Flat memory indexing rules for row-major vs. column-major layouts, contiguous strides vs. leading dimensions (`LDC`, `LDA`).
* **Hardware Motivation**: Foundation for BLAS matrix multiplication standards used by NVIDIA cuBLAS and CUTLASS.
* **Status**: `[QUEUED ⏳]`

---

### Chapter 1.6: Permute & Flatten Operations
> **Synopsis**: Transforming 4D tensor layouts `[Batch, Heads, SeqLen, HeadDim]` into contiguous memory blocks for multi-head attention.
* **Hardware Motivation**: Preparing Query, Key, and Value projections for FlashAttention computation.
* **Status**: `[QUEUED ⏳]`

---

### Chapter 1.7: Subgrids & Padding
> **Synopsis**: Extracting sub-tensors, handling matrix boundary conditions, and padding edges to avoid unaligned hardware access.
* **Hardware Motivation**: Kernel boundary guards preventing out-of-bounds memory faults on irregular sequence lengths.
* **Status**: `[QUEUED ⏳]`

---

### Chapter 1.8: Packing, Wrapping & Stencils
> **Synopsis**: Spatial memory access patterns, cyclic ring buffers, and packing disjoint memory blocks into contiguous buffers.
* **Hardware Motivation**: KV-Cache management in autoregressive token generation.
* **Status**: `[QUEUED ⏳]`

---

### Chapter 1.9: Single Array Allocation Schemes
> **Synopsis**: Partitioning one large contiguous heap buffer into dozens of heterogeneous arrays with zero allocator overhead.
* **Hardware Motivation**: Static memory allocation for LLM inference graphs (allocating all model activations in one buffer).
* **Status**: `[QUEUED ⏳]`

---

### Chapter 1.10: Multidimensional Alignment
> **Synopsis**: Enforcing hardware cache-line alignment across 3D and 4D tensor strides and dimensions.
* **Hardware Motivation**: Ensuring every attention head and row starts at a hardware-coalesced boundary.
* **Status**: `[QUEUED ⏳]`

---

### Chapter 1.11: Arenas & Custom Placements
> **Synopsis**: Advanced arena allocators, placement `new`, and memory recycling strategies without OS syscalls.
* **Hardware Motivation**: PyTorch caching allocator architecture (`c10::CUDAAllocator`).
* **Status**: `[QUEUED ⏳]`

---

### Chapter 1.12: Lifetimes & Ownership Semantics
> **Synopsis**: RAII, object lifetimes, stack vs. heap ownership, and avoiding dangling pointer hazards.
* **Hardware Motivation**: Managing GPU VRAM device pointers safely across multiple host threads.
* **Status**: `[QUEUED ⏳]`

---

### Chapter 1.13: Null & Bounds Verification
> **Synopsis**: Defensive systems programming, memory safety verification, and hardware illegal address trap prevention.
* **Hardware Motivation**: Eliminating `CUDA_ERROR_ILLEGAL_ADDRESS` exceptions that halt GPU execution.
* **Status**: `[QUEUED ⏳]`

---

### Chapter 1.14: Size, Padding & Type Punning
> **Synopsis**: Bitwise union casting, struct memory packing rules, and inspecting raw representation of floating-point values.
* **Hardware Motivation**: FP16 / BF16 / FP8 / INT4 low-precision quantization formats.
* **Status**: `[QUEUED ⏳]`

---

### Chapter 1.15: Casts & Double Pointers
> **Synopsis**: Pointer to pointer indirection (`void**`, `float**`), ragged arrays, and type conversions.
* **Hardware Motivation**: Managing device pointer tables and batched GEMM array-of-pointers interfaces.
* **Status**: `[QUEUED ⏳]`

---

### Chapter 1.16: Pointer Relations & Copying
> **Synopsis**: Overlapping memory regions, pointer aliasing, `std::memcpy` vs `std::memmove`, and compiler `__restrict__` optimizations.
* **Hardware Motivation**: Informing CUDA compilers that pointer inputs do not alias, enabling aggressive register caching.
* **Status**: `[QUEUED ⏳]`

---

### Chapter 1.17: Embeddings & Projections
> **Synopsis**: Low-level memory lookup tables: gathering rows from an embedding weight matrix based on token IDs.
* **Hardware Motivation**: Implementing the Token Embedding layer in LLMs.
* **Status**: `[QUEUED ⏳]`

---

### Chapter 1.18: Attention & Quantization Packing
> **Synopsis**: Packing 4-bit and 8-bit quantized weights into 32-bit registers using bit-shifts and masks.
* **Hardware Motivation**: AWQ and GPTQ quantized model inference execution.
* **Status**: `[QUEUED ⏳]`

---

### Chapter 1.19: ML Layer Offsets & Layouts
> **Synopsis**: Laying out Transformer block weights (QKV projections, FFN gate/up/down projections, RMSNorm weights) in flat files.
* **Hardware Motivation**: Deserializing GGUF / SafeTensors model weights directly into device memory.
* **Status**: `[QUEUED ⏳]`

---

### Chapter 1.20: Pooling, Masks & Cyclic Buffers
> **Synopsis**: Applying causal attention triangular masks, token padding masks, and cyclic state management.
* **Hardware Motivation**: The complete memory infrastructure supporting FlashAttention and Autoregressive decoding.
* **Status**: `[QUEUED ⏳]`

---

# Volume 2: CUDA Architecture & Execution Model

### Module 2: CUDA Execution Model
* **Core Topics**: GPU Streaming Multiprocessors (SMs), Warp execution (32 lockstep threads), Thread Blocks, 1D/2D/3D Grid dimensions, Warp Divergence mitigation.
* **Hardware Goal**: Transitioning mental models from 1 CPU thread looping to 100,000 GPU threads executing in parallel.

### Module 3: CUDA Memory Hierarchy
* **Core Topics**: Global VRAM, On-Chip Shared Memory (SRAM), Registers, Constant Memory, Bank Conflicts (32 banks), Coalesced Memory Access.
* **Hardware Goal**: Maximizing arithmetic intensity by staging data from slow VRAM into ultra-fast SRAM.

### Module 4: High-Performance Parallel Primitives
* **Core Topics**: Parallel Tree Reductions, Prefix Sums (Scan), Softmax Online Normalization (Flash-Softmax), Matrix Multiplication (GEMM) Tiling.
* **Hardware Goal**: Building the reusable algorithmic primitives behind deep learning operations.

---

# Volume 3: The 22 Production LLM CUDA Kernels

Your ultimate objective: Authoring and benchmarking the complete CUDA implementation of all 22 kernels from your Metal LLM engine (`/Users/rahulkumar/dev/large-language-model/cpp/src/gpu_kernel/`).

```text
+-----------------------------------------------------------------------------------------+
|                                22 LLM KERNELS ARCHITECTURE                              |
|                                                                                         |
|  [ Input Token IDs ] ---> (1) Embedding Forward                                         |
|                                 |                                                       |
|  +------------------------------v----------------------------------------------------+  |
|  | TRANSFORMER BLOCK (Repeats for L Layers)                                          |  |
|  |                                                                                   |  |
|  |  (2,3) RMSNorm (Fwd/Bwd)                                                          |  |
|  |     |                                                                             |  |
|  |  (8-12) QKV GEMM Projections (BF16, Proj, Trans B, GQA)                            |  |
|  |     |                                                                             |  |
|  |  (4,5) RoPE Rotary Positional Embedding (Fwd/Bwd)                                 |  |
|  |     |                                                                             |  |
|  |  (6,7) FlashAttention (Fused Tiling Fwd & Bwd)                                    |  |
|  |     |                                                                             |  |
|  |  (17-19) Residual Add & Fused Add-Norm                                            |  |
|  |     |                                                                             |  |
|  |  (14-16) SwiGLU FFN Activation & Fused GEMM (Fwd/Bwd)                             |  |
|  +------------------------------+----------------------------------------------------+  |
|                                 |                                                       |
|  (20,21) Cross Entropy & Loss Computation                                               |
|                                 |                                                       |
|  (22) AdamW Optimizer Step (Weight Updates)                                             |
+-----------------------------------------------------------------------------------------+
```

| # | Kernel Operation | Category | Metal Reference File | CUDA Implementation | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 1 | **Embedding Forward** | Token Embedding | `embedding_forward.metal` | `embedding_forward.cu` | Queued |
| 2 | **RMSNorm Forward** | Normalization | `rms_norm_forward.metal` | `rms_norm_forward.cu` | Queued |
| 3 | **RMSNorm Backward** | Normalization (Backprop) | `rms_norm_backward.metal` | `rms_norm_backward.cu` | Queued |
| 4 | **RoPE Forward** | Rotary Position Embedding | `rope_forward.metal` | `rope_forward.cu` | Queued |
| 5 | **RoPE Backward** | Rotary Position Embedding | `rope_backward.metal` | `rope_backward.cu` | Queued |
| 6 | **FlashAttention Forward** | Attention (Fused Tiling) | `flash_attn_fwd.metal` | `flash_attn_fwd.cu` | Queued |
| 7 | **Fused Attention Backward**| Attention (Backprop) | `fused_attn_bwd.metal` | `fused_attn_bwd.cu` | Queued |
| 8 | **GEMM BF16** | Matrix Multiplication | `gemm_bf16.metal` | `gemm_bf16.cu` | Queued |
| 9 | **GEMM Projection** | Attention QKV Projection | `gemm_proj.metal` | `gemm_proj.cu` | Queued |
| 10 | **GEMM Projection Trans B** | Transposed Projection | `gemm_proj_trans_b.metal` | `gemm_proj_trans_b.cu` | Queued |
| 11 | **GEMM GQA** | Grouped Query Attention | `gemm_gqa.metal` | `gemm_gqa.cu` | Queued |
| 12 | **GEMM FFN** | Feed-Forward Network | `gemm_ffn.metal` | `gemm_ffn.cu` | Queued |
| 13 | **GEMM Backward** | Linear Layer Gradients | `gemm_backward.metal` | `gemm_backward.cu` | Queued |
| 14 | **SwiGLU Forward** | Activation Function | `swiglu_forward.metal` | `swiglu_forward.cu` | Queued |
| 15 | **SwiGLU Backward** | Activation Gradients | `swiglu_backward.metal` | `swiglu_backward.cu` | Queued |
| 16 | **Fused SwiGLU GEMM** | Fused Operator | `fused_swiglu_gemm.metal` | `fused_swiglu_gemm.cu` | Queued |
| 17 | **Residual Add** | Skip Connections | `residual_add.metal` | `residual_add.cu` | Queued |
| 18 | **Fused Add Norm** | Fused Residual + RMSNorm | `fused_add_norm.metal` | `fused_add_norm.cu` | Queued |
| 19 | **Fused Backward Add Norm** | Backprop Fused Norm | `fused_backward_add_norm.metal`| `fused_backward_add_norm.cu`| Queued |
| 20 | **Cross Entropy** | Loss Function | `cross_entropy.metal` | `cross_entropy.cu` | Queued |
| 21 | **Compute Loss** | Final Loss Reduction | `compute_loss.metal` | `compute_loss.cu` | Queued |
| 22 | **AdamW Optimizer Step** | Weight Update Engine | `adamw_step.metal` | `adamw_step.cu` | Queued |
