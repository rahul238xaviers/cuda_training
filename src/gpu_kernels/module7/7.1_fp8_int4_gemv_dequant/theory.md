# FP8 / INT4 Weight-Only GEMV with On-The-Fly Dequant

## Core Mental Model & Hardware Motivation
In modern high-performance AI systems, FP8 / INT4 Weight-Only GEMV with On-The-Fly Dequant (Quantization) serves as a critical computation bottleneck.
The goal of this kernel is:
y = sum_k(dequant(W_int4[row, k], scale[row]) * x[k]) with 75% DRAM traffic reduction

### Hardware Context
* **Arithmetic Intensity**: Packing 2 INT4 nibbles per byte, on-the-fly register unpacking, and scale-factor multiply
* **Memory Access Pattern**: Coalesced DRAM accesses streaming into registers and shared SRAM.
* **Threading Layout**: Cooperative thread blocks minimizing warp divergence and bank serialization.

## Architectural Equations (Plain Text ASCII Math)
Objective:
  y = sum_k(dequant(W_int4[row, k], scale[row]) * x[k]) with 75% DRAM traffic reduction

Roofline Formula:
  DRAM Bandwidth = (Bytes Read + Bytes Written) / (Elapsed Seconds * 1e9) GB/s
  Throughput = Total FLOPs / (Elapsed Seconds * 1e12) TFLOPs

## Common Pitfalls to Avoid
1. **Uncoalesced Memory Transactions**: Strided or unaligned memory access forces multiple 32-byte or 128-byte transactions per warp.
2. **Shared Memory Bank Conflicts**: Simultaneous access to different addresses in the same 32 SRAM banks serializes execution.
3. **Register Spilling**: Allocating too many registers per thread causes local memory spills to slow DRAM.
