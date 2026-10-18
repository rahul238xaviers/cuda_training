# How to Read & Master This Curriculum

From physical byte arithmetic and zero-cost C++ abstractions to saturating GPU memory bandwidth, warp scheduling, and production LLM kernels.

---

## 1. The Three-Volume Architectural Progression

GPU kernels do not exist in isolation. Writing optimal CUDA requires deep intuition for pointers, memory alignments, cache line structures, and value semantics. The curriculum is partitioned into 3 progressive volumes:

| Volume | Focus Domain | Curriculum Scope | Core Architectural Takeaways |
| :--- | :--- | :---: | :--- |
| **Volume 1** | **C++ Low-Level Systems** | 20 Modules | Pointer arithmetic, strides, memory alignment, 64-byte cache lines, custom arenas, zero-copy buffers, SIMD, and acquire-release concurrency. |
| **Volume 2** | **CUDA Hardware Execution** | 21 Modules | Streaming Multiprocessors (SMs), warps, divergent branches, coalesced DRAM transactions, bank-conflict-free shared SRAM, tree reductions, and WMMA Tensor Cores. |
| **Volume 3** | **Production GPU Kernels** | 28 Kernels | Dense GEMM, Sparse/MoE, FlashAttention-2, Fused RMSNorm, Multimodal Audio/Video, INT4/FP8 Quantization, and AdamW Optimizers. |

---

## 2. The Three-Tier Mastery Methodology

Every chapter provides three tiers of hands-on practice. You only master a technique when you progress through all three levels:

1. **Tier 1: Beginner (Mechanics & Memory Tracing)**
   - Master basic indexing, bounds checking, and clean memory offset arithmetic.
   - Trace exact byte offsets and cache line boundaries by hand before writing code.
2. **Tier 2: Intermediate (Vectorization & Edge Cases)**
   - Eliminate branching divergence, align buffers to 16-byte SIMD boundaries, and handle non-power-of-2 tail elements.
   - Utilize vectorized load/store instructions (`float4`, AVX-512) to saturate memory controllers.
3. **Tier 3: Champion (Hardware Saturation & Performance)**
   - Maximize hardware saturation, avoid bank conflicts, leverage register double-buffering, and minimize memory bus traffic.
   - Prove correctness against reference implementations and pass strict microbenchmark timing thresholds.

> [!IMPORTANT]
> **Pedagogy Invariant**: Always implement and validate your mental models in `playground.cpp` or `playground.cu` first before attempting workbook problems. Test failure is your best teacher!

---

## 3. The Kernel Verification Sandbox

The lab environment includes dedicated verification tools to ensure your kernels are production-ready:

* **Microbenchmarking Harness**: Evaluate TFLOPS throughput, memory bandwidth saturation, and roofline efficiency.
* **Memory Safety Sanitizers**: Run AddressSanitizer (ASan) and `compute-sanitizer` to detect out-of-bounds memory accesses and race conditions.
* **Automated Sync**: Seamless synchronization between your local code, SQLite progress tracking, and interactive textbook diagrams.

> [!TIP]
> **Roofline Saturation Rule**: In bandwidth-bound kernels (like SAXPY or RMSNorm), adding more compute cores yields zero speedup. Always optimize your memory access pattern to saturate the physical DRAM bus first!
