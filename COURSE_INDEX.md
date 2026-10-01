# Master Curriculum Tracker & Journey Index

This document is your single-source-of-truth bookmark and progress tracker for the entire journey: from C++ memory systems, to CUDA kernel foundations, all the way to authoring the **22 production CUDA LLM kernels** matching your Metal implementation in `dev/large-language-model`.

---

## 📍 CURRENT BOOKMARK & ACTIVE STATION

- **Current Phase**: Phase 1: C++ Low-Level Memory Foundations
- **Current Chapter**: **Chapter 1.2: Strides & Pointer Indirection**
- **Last Completed**: Chapter 1.1: Basic Offsets & Pointer Arithmetic (100% Passed 🏆)
- **Active Playground**: `playground.cpp`

---

## 🗺️ The 3-Phase Master Roadmap

```text
+-------------------------------------------------------------------------------+
| PHASE 1: C++ Low-Level Memory Foundations (Module 1: Chapters 1.1 - 1.20)      |
| Goal: Master byte layouts, pointer scaling, alignment, strides, and memory.   |
| Status: [ 1 / 20 Completed ]                                                  |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| PHASE 2: CUDA Architecture & Kernel Creation (Modules 2 - 4)                  |
| Goal: Write GPU device kernels, grids, blocks, warps, and shared memory tiling.|
| Status: [ Queued ]                                                            |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| PHASE 3: The 22 Production LLM CUDA Kernels                                   |
| Goal: Port and write all 22 LLM kernels from your Metal LLM engine into CUDA. |
| Status: [ 0 / 22 Completed ]                                                  |
+-------------------------------------------------------------------------------+
```

---

## Phase 1: C++ Low-Level Memory Systems (Module 1)

Focus: Master physical memory layouts, strides, byte addressing, and allocators from first principles.

| Chapter | Topic | Theory & Cheat Sheet | Beginner | Interm. | Champion | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1.1** | **Basic Memory Offsets & Pointer Arithmetic** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.1_basic_offsets/theory.md) \| [Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.1_basic_offsets/cheat_sheet.md) | [x] | [x] | [x] | **COMPLETED 🏆** |
| **1.2** | **Strides & Pointer Indirection** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.2_strides_indirection/theory.md) | [ ] | [ ] | [ ] | **CURRENT 📍** |
| **1.3** | **Reductions & Swaps** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.3_reductions_swaps/theory.md) | [ ] | [ ] | [ ] | Queued |
| **1.4** | **Multi-Array Strided Traversal** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.4_multi_array_strided/theory.md) | [ ] | [ ] | [ ] | Queued |
| **1.5** | **Row/Column Matrix Indexing** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.5_row_col_indexing/theory.md) | [ ] | [ ] | [ ] | Queued |
| **1.6** | **Permute & Flatten Operations** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.6_permute_flatten/theory.md) | [ ] | [ ] | [ ] | Queued |
| **1.7** | **Subgrids & Padding** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.7_subgrids_padding/theory.md) | [ ] | [ ] | [ ] | Queued |
| **1.8** | **Packing, Wrapping & Stencils** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.8_pack_wrap_stencil/theory.md) | [ ] | [ ] | [ ] | Queued |
| **1.9** | **Single Array Allocation Schemes** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.9_single_array_alloc/theory.md) | [ ] | [ ] | [ ] | Queued |
| **1.10** | **Multidimensional Alignment** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.10_multidim_alignment/theory.md) | [ ] | [ ] | [ ] | Queued |
| **1.11** | **Arenas & Custom Placements** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.11_arenas_placements/theory.md) | [ ] | [ ] | [ ] | Queued |
| **1.12** | **Lifetimes & Ownership Semantics** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.12_lifetimes_ownership/theory.md) | [ ] | [ ] | [ ] | Queued |
| **1.13** | **Null & Bounds Verification** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.13_null_bounds_checks/theory.md) | [ ] | [ ] | [ ] | Queued |
| **1.14** | **Size, Padding & Type Punning** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.14_size_padding_punning/theory.md) | [ ] | [ ] | [ ] | Queued |
| **1.15** | **Casts & Double Pointers** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.15_casts_double_pointers/theory.md) | [ ] | [ ] | [ ] | Queued |
| **1.16** | **Pointer Relations & Copying** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.16_pointer_relations_copying/theory.md) | [ ] | [ ] | [ ] | Queued |
| **1.17** | **Embeddings & Projections** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.17_embeddings_projections/theory.md) | [ ] | [ ] | [ ] | Queued |
| **1.18** | **Attention & Quantization Packing** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.18_attention_quantization/theory.md) | [ ] | [ ] | [ ] | Queued |
| **1.19** | **ML Layer Offsets & Layouts** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.19_ml_layer_offsets/theory.md) | [ ] | [ ] | [ ] | Queued |
| **1.20** | **Pooling, Masks & Cyclic Buffers** | [Theory](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.20_pooling_masks_cycles/theory.md) | [ ] | [ ] | [ ] | Queued |

---

## Phase 2: CUDA Architecture & Execution Foundations (Modules 2 - 4)

Focus: Transitioning from host C++ pointer mechanics to GPU execution.

| Module | Core Concept | Key Focus Areas | Status |
| :--- | :--- | :--- | :---: |
| **Module 2** | **CUDA Execution Model** | GPU Threads, Warps (32 threads), Thread Blocks, 1D/2D/3D Grids, Divergence | Queued |
| **Module 3** | **CUDA Memory Hierarchy** | Global VRAM, Shared Memory (SRAM), Registers, Constant Memory, Bank Conflicts | Queued |
| **Module 4** | **Parallel Primitives** | Parallel Reductions, Prefix Sum (Scan), Softmax, GEMM Tiling | Queued |

---

## Phase 3: The 22 Production LLM CUDA Kernels

Your ultimate objective: Authoring and benchmarking the complete CUDA implementation of all 22 kernels from your Metal LLM engine (`/Users/rahulkumar/dev/large-language-model/cpp/src/gpu_kernel/`).

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

---

## 🏆 Completed Chapters Detail Log

### Chapter 1.1: Basic Memory Offsets & Pointer Arithmetic
- **Status**: Completed (2026-10-01)
- **Workbooks Completed**:
  - `beginner_workbook.cpp` (4/4 passed) — Pointer scaling, weight blob deserialization, in-place reversal, ring buffer wrapping.
  - `intermediate_workbook.cpp` (3/3 passed) — 2D pitched memory layouts (`cudaMallocPitch`), 128-bit vectorized stream copy (`Float4` @ 46+ GB/s), bitwise alignment & padding math.
  - `champion_workbook.cpp` (3/3 passed) — Custom aligned bump allocator (arena memory pool), double-buffered pointer ping-pong pipeline, 16x16 tiled matrix transpose (cache-thrashing fix @ 10.70 GB/s).
- **Artifacts**:
  - [Theory Guide](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.1_basic_offsets/theory.md)
  - [Master Cheat Sheet](file:///Users/rahulkumar/dev/cuda_training/src/module1/1.1_basic_offsets/cheat_sheet.md)
