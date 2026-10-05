# 2:4 Structured Sparsity Accelerated GEMM

## Core Mental Model & Hardware Motivation
In modern high-performance AI systems, 2:4 Structured Sparsity Accelerated GEMM (Sparse Computations) serves as a critical computation bottleneck.
The goal of this kernel is:
C = A_sparse_24 * B where exactly 2 of every 4 adjacent elements are zero

### Hardware Context
* **Arithmetic Intensity**: NVIDIA Ampere/Hopper 2:4 sparse tensor core acceleration with 2x math throughput
* **Memory Access Pattern**: Coalesced DRAM accesses streaming into registers and shared SRAM.
* **Threading Layout**: Cooperative thread blocks minimizing warp divergence and bank serialization.

## Architectural Equations (Plain Text ASCII Math)
Objective:
  C = A_sparse_24 * B where exactly 2 of every 4 adjacent elements are zero

Roofline Formula:
  DRAM Bandwidth = (Bytes Read + Bytes Written) / (Elapsed Seconds * 1e9) GB/s
  Throughput = Total FLOPs / (Elapsed Seconds * 1e12) TFLOPs

## Common Pitfalls to Avoid
1. **Uncoalesced Memory Transactions**: Strided or unaligned memory access forces multiple 32-byte or 128-byte transactions per warp.
2. **Shared Memory Bank Conflicts**: Simultaneous access to different addresses in the same 32 SRAM banks serializes execution.
3. **Register Spilling**: Allocating too many registers per thread causes local memory spills to slow DRAM.
