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
Phase 2: CUDA Architecture    [░░░░░░░░░░░░░░░░░░░░]  0% (0/6 Modules, 0/22 Topics)
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
- [Volume 2: CUDA Architecture & LLM Execution Model](#volume-2-cuda-architecture--llm-execution-model)
  - [Module 2: Host-Device Bridge & Modern C++](#module-2-host-device-bridge--modern-c)
    - [Topic 2.1: Lambdas, Closures & Function Objects in CUDA](#topic-21-lambdas-closures--function-objects-in-cuda)
    - [Topic 2.2: Templates, SFINAE & Precision Type Traits](#topic-22-templates-sfinae--precision-type-traits)
    - [Topic 2.3: Smart Pointers, RAII & Resource Wrappers](#topic-23-smart-pointers-raii--resource-wrappers)
    - [Topic 2.4: Move Semantics & Tensor Views](#topic-24-move-semantics--tensor-views)
    - [Topic 2.5: Host Parallel Execution & CPU-GPU Co-Design](#topic-25-host-parallel-execution--cpu-gpu-co-design)
  - [Module 3: CUDA Threading Hierarchy & Hardware Execution](#module-3-cuda-threading-hierarchy--hardware-execution)
    - [Topic 3.1: Threads & Registers](#topic-31-threads--registers)
    - [Topic 3.2: Warps, SIMT Execution & Shuffle Primitives](#topic-32-warps-simt-execution--shuffle-primitives)
    - [Topic 3.3: Thread Blocks, Shared Memory & Bank Conflicts](#topic-33-thread-blocks-shared-memory--bank-conflicts)
    - [Topic 3.4: Grids, SM Occupancy & Grid-Stride Loops](#topic-34-grids-sm-occupancy--grid-stride-loops)
  - [Module 4: Memory Hierarchy, Coalescing & SRAM Tiling](#module-4-memory-hierarchy-coalescing--sram-tiling)
    - [Topic 4.1: Memory Coalescing & Read-Only Cache](#topic-41-memory-coalescing--read-only-cache)
    - [Topic 4.2: Shared Memory Matrix Tiling & Micro-Kernels](#topic-42-shared-memory-matrix-tiling--micro-kernels)
  - [Module 5: High-Performance Parallel Primitives](#module-5-high-performance-parallel-primitives)
    - [Topic 5.1: Two-Pass Block & Warp Reductions](#topic-51-two-pass-block--warp-reductions)
    - [Topic 5.2: Atomic Operations & Contention Reduction](#topic-52-atomic-operations--contention-reduction)
    - [Topic 5.3: Numerically Stable Reductions (Online Softmax)](#topic-53-numerically-stable-reductions-online-softmax)
  - [Module 6: Low-Precision & Tensor Cores](#module-6-low-precision--tensor-cores)
    - [Topic 6.1: BFloat16 Math & Mixed Precision](#topic-61-bfloat16-math--mixed-precision)
    - [Topic 6.2: Tensor Cores & The WMMA API](#topic-62-tensor-cores--the-wmma-api)
  - [Module 7: Production LLM Inference & Training Primitives](#module-7-production-llm-inference--training-primitives)
    - [Topic 7.1: Elementwise Operations & Fused Reshaping](#topic-71-elementwise-operations--fused-reshaping)
    - [Topic 7.2: Token Embedding Lookup & Gather](#topic-72-token-embedding-lookup--gather)
    - [Topic 7.3: RMSNorm & Rotary Position Embeddings (RoPE)](#topic-73-rmsnorm--rotary-position-embeddings-rope)
    - [Topic 7.4: Cross-Entropy Loss & AdamW Optimizer](#topic-74-cross-entropy-loss--adamw-optimizer)
    - [Topic 7.5: GEMM & Linear Projections Suite](#topic-75-gemm--linear-projections-suite)
    - [Topic 7.6: FlashAttention Suite (Forward & Backward)](#topic-76-flashattention-suite-forward--backward)
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

# Volume 2: CUDA Architecture & LLM Execution Model

### Module 2: Host-Device Bridge & Modern C++
> **Synopsis**: Bridge host C++20 orchestration with high-performance GPU execution using extended lambdas, compile-time templates, RAII streams, and zero-copy tensor views.
* **Topics**:
  - **2.1: Lambdas, Closures & Function Objects**: Device closures, capturing rules, inlined higher-order kernels.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.1_lambdas_function_objects/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.1_lambdas_function_objects/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.1_lambdas_function_objects/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.1_lambdas_function_objects/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.1_lambdas_function_objects/exercise/champion_workbook.cu)
  - **2.2: Templates, SFINAE & Precision Type Traits**: Compile-time precision dispatch across FP32, FP16, BF16.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.2_templates_type_traits/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.2_templates_type_traits/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.2_templates_type_traits/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.2_templates_type_traits/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.2_templates_type_traits/exercise/champion_workbook.cu)
  - **2.3: Smart Pointers, RAII & Resource Wrappers**: Deterministic VRAM lifecycles, CUDA stream priorities, async overlap.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.3_raii_smart_pointers_streams/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.3_raii_smart_pointers_streams/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.3_raii_smart_pointers_streams/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.3_raii_smart_pointers_streams/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.3_raii_smart_pointers_streams/exercise/champion_workbook.cu)
  - **2.4: Move Semantics & Tensor Views**: Zero-copy tensor slicing, non-owning strided spans.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.4_move_semantics_views/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.4_move_semantics_views/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.4_move_semantics_views/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.4_move_semantics_views/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.4_move_semantics_views/exercise/champion_workbook.cu)
  - **2.5: Host Parallel Execution & CPU-GPU Co-Design**: Double-buffering ping-pong pipelines and non-blocking stream execution.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.5_stl_parallel_execution/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.5_stl_parallel_execution/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.5_stl_parallel_execution/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.5_stl_parallel_execution/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module2/2.5_stl_parallel_execution/exercise/champion_workbook.cu)

---

### Module 3: CUDA Threading Hierarchy & Hardware Execution
> **Synopsis**: Understand physical GPU hardware architecture: Streaming Multiprocessors (SMs), 32-thread lockstep warps, register files, and shared memory banking.
* **Topics**:
  - **3.1: Threads & Registers**: Thread coordinates, register pressure, local memory spills, 128-bit `float4` loads.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.1_threads_registers/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.1_threads_registers/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.1_threads_registers/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.1_threads_registers/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.1_threads_registers/exercise/champion_workbook.cu)
  - **3.2: Warps, SIMT & Shuffle Intrinsics**: Warp divergence elimination, register exchange via `__shfl_down_sync`.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.2_warps_simt_shuffles/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.2_warps_simt_shuffles/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.2_warps_simt_shuffles/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.2_warps_simt_shuffles/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.2_warps_simt_shuffles/exercise/champion_workbook.cu)
  - **3.3: Thread Blocks & Shared Memory Banking**: 32 SRAM banks, stride-induced conflict serialization, +1 padding techniques.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.3_blocks_shared_banking/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.3_blocks_shared_banking/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.3_blocks_shared_banking/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.3_blocks_shared_banking/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.3_blocks_shared_banking/exercise/champion_workbook.cu)
  - **3.4: Grids, SM Occupancy & Grid-Stride Loops**: Achieving 100% SM occupancy, hardware-agnostic grid-stride looping.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.4_grids_occupancy_strides/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.4_grids_occupancy_strides/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.4_grids_occupancy_strides/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.4_grids_occupancy_strides/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module3/3.4_grids_occupancy_strides/exercise/champion_workbook.cu)

---

### Module 4: Memory Hierarchy, Coalescing & SRAM Tiling
> **Synopsis**: Maximize arithmetic intensity and GPU memory bus saturation by staging data from high-latency DRAM into on-chip shared SRAM without bank conflicts.
* **Topics**:
  - **4.1: Memory Coalescing & Read-Only Cache**: 128-byte DRAM cache-line coalescing, `__ldg` texture cache streaming.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module4/4.1_coalescing_readonly_cache/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module4/4.1_coalescing_readonly_cache/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module4/4.1_coalescing_readonly_cache/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module4/4.1_coalescing_readonly_cache/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module4/4.1_coalescing_readonly_cache/exercise/champion_workbook.cu)
  - **4.2: Shared Memory Matrix Tiling**: 2D tile staging, collaborative DRAM loads, register micro-kernels.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module4/4.2_shared_memory_tiling/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module4/4.2_shared_memory_tiling/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module4/4.2_shared_memory_tiling/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module4/4.2_shared_memory_tiling/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module4/4.2_shared_memory_tiling/exercise/champion_workbook.cu)

---

### Module 5: High-Performance Parallel Primitives
> **Synopsis**: Build the fundamental mathematical primitives of modern deep learning engines: tree reductions, atomics, and numerically stable online reductions.
* **Topics**:
  - **5.1: Two-Pass Block & Warp Reductions**: Hierarchical intra-warp and inter-warp tree reductions.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module5/5.1_warp_block_reductions/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module5/5.1_warp_block_reductions/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module5/5.1_warp_block_reductions/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module5/5.1_warp_block_reductions/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module5/5.1_warp_block_reductions/exercise/champion_workbook.cu)
  - **5.2: Atomic Operations & Contention Reduction**: Hardware atomics, memory fences, hierarchical block aggregation.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module5/5.2_atomic_operations/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module5/5.2_atomic_operations/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module5/5.2_atomic_operations/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module5/5.2_atomic_operations/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module5/5.2_atomic_operations/exercise/champion_workbook.cu)
  - **5.3: Numerically Stable Reductions (Online Softmax)**: Safe 3-pass vs 1-pass online softmax with running correction.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module5/5.3_numerically_stable_reductions/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module5/5.3_numerically_stable_reductions/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module5/5.3_numerically_stable_reductions/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module5/5.3_numerically_stable_reductions/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module5/5.3_numerically_stable_reductions/exercise/champion_workbook.cu)

---

### Module 6: Low-Precision & Tensor Cores
> **Synopsis**: Accelerate compute-bound operations using 16-bit brain float format and hardware Tensor Cores for 10x throughput scaling.
* **Topics**:
  - **6.1: BFloat16 Math & Vectorized Types**: Dynamic range equivalence to FP32, packed arithmetic (`__nv_bfloat162`).
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module6/6.1_bfloat16_math/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module6/6.1_bfloat16_math/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module6/6.1_bfloat16_math/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module6/6.1_bfloat16_math/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module6/6.1_bfloat16_math/exercise/champion_workbook.cu)
  - **6.2: Tensor Cores & The WMMA API**: Warp Matrix Multiply Accumulate (`nvcuda::wmma`), fragment loading, matrix multiply-accumulate.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module6/6.2_wmma_tensor_cores/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module6/6.2_wmma_tensor_cores/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module6/6.2_wmma_tensor_cores/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module6/6.2_wmma_tensor_cores/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module6/6.2_wmma_tensor_cores/exercise/champion_workbook.cu)

---

### Module 7: Production LLM Inference & Training Primitives
> **Synopsis**: Author the complete operator suite powering state-of-the-art LLM architectures (LLaMA 3, Mistral, Gemma 2, DeepSeek): token embeddings, RoPE, RMSNorm, AdamW, GEMM projections, and FlashAttention.
* **Topics**:
  - **7.1: Elementwise Operations & Fused Reshaping**: Fused residual addition, SwiGLU activation, 128-bit streaming.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.1_elementwise_reshape/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.1_elementwise_reshape/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.1_elementwise_reshape/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.1_elementwise_reshape/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.1_elementwise_reshape/exercise/champion_workbook.cu)
  - **7.2: Token Embedding Lookup & Gather**: Translating vocab token IDs into dense hidden vectors with coalesced 128-byte DRAM reads.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.2_embedding_lookup/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.2_embedding_lookup/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.2_embedding_lookup/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.2_embedding_lookup/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.2_embedding_lookup/exercise/champion_workbook.cu)
  - **7.3: RMSNorm & Rotary Position Embeddings (RoPE)**: Root Mean Square normalization via warp shuffles and complex rotary position encoding.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.3_normalization_rope/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.3_normalization_rope/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.3_normalization_rope/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.3_normalization_rope/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.3_normalization_rope/exercise/champion_workbook.cu)
  - **7.4: Cross-Entropy Loss & AdamW Optimizer**: Fused log-sum-exp loss reduction and single-pass GPU AdamW momentum/variance parameter updates.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.4_loss_and_optimizer/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.4_loss_and_optimizer/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.4_loss_and_optimizer/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.4_loss_and_optimizer/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.4_loss_and_optimizer/exercise/champion_workbook.cu)
  - **7.5: GEMM & Linear Projections Suite**: Fused QKV linear projections, SwiGLU FFN projections, and register micro-tiling.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.5_gemm_projections/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.5_gemm_projections/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.5_gemm_projections/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.5_gemm_projections/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.5_gemm_projections/exercise/champion_workbook.cu)
  - **7.6: FlashAttention Suite (Forward & Backward)**: IO-aware tiled online softmax attention without materializing NxN attention matrices in HBM.
    - 📖 [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.6_flash_attention/theory.md) | ⚡ [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.6_flash_attention/cheat_sheet.md) | 📝 Workbooks: [Beginner](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.6_flash_attention/exercise/beginner_workbook.cu), [Intermediate](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.6_flash_attention/exercise/intermediate_workbook.cu), [Champion](file:///Users/rahulkumar/dev/cuda_training/src/module7/7.6_flash_attention/exercise/champion_workbook.cu)

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
