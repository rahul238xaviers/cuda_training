# 2D Block-Tiled FP32 Shared Memory GEMM

## Core Mental Model & Hardware Motivation
In modern high-performance AI systems, 2D Block-Tiled FP32 Shared Memory GEMM (Dense Linear) serves as a critical computation bottleneck.
The goal of this kernel is:
C = A * B where A is (M x K), B is (K x N), and C is (M x N)

### Hardware Context
* **Arithmetic Intensity**: Square SRAM tiling, collaborative DRAM staging, register micro-tile accumulation
* **Memory Access Pattern**: Coalesced DRAM accesses streaming into registers and shared SRAM.
* **Threading Layout**: Cooperative thread blocks minimizing warp divergence and bank serialization.

## Architectural Equations (Plain Text ASCII Math)
Objective:
  C = A * B where A is (M x K), B is (K x N), and C is (M x N)

Roofline Formula:
  DRAM Bandwidth = (Bytes Read + Bytes Written) / (Elapsed Seconds * 1e9) GB/s
  Throughput = Total FLOPs / (Elapsed Seconds * 1e12) TFLOPs

## Common Pitfalls to Avoid
1. **Uncoalesced Memory Transactions**: Strided or unaligned memory access forces multiple 32-byte or 128-byte transactions per warp.
2. **Shared Memory Bank Conflicts**: Simultaneous access to different addresses in the same 32 SRAM banks serializes execution.
3. **Register Spilling**: Allocating too many registers per thread causes local memory spills to slow DRAM.
