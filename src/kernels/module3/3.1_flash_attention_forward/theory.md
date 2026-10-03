# FlashAttention-2 Forward Kernel

## Core Mental Model & Hardware Motivation
In modern high-performance AI systems, FlashAttention-2 Forward Kernel (Attention Mechanisms) serves as a critical computation bottleneck.
The goal of this kernel is:
O = softmax(Q * K^T / sqrt(d)) * V using online streaming max and sum-exp in SRAM

### Hardware Context
* **Arithmetic Intensity**: Online softmax, SRAM tiling of Q, K, V blocks, and zero DRAM N x N materialization
* **Memory Access Pattern**: Coalesced DRAM accesses streaming into registers and shared SRAM.
* **Threading Layout**: Cooperative thread blocks minimizing warp divergence and bank serialization.

## Architectural Equations (Plain Text ASCII Math)
Objective:
  O = softmax(Q * K^T / sqrt(d)) * V using online streaming max and sum-exp in SRAM

Roofline Formula:
  DRAM Bandwidth = (Bytes Read + Bytes Written) / (Elapsed Seconds * 1e9) GB/s
  Throughput = Total FLOPs / (Elapsed Seconds * 1e12) TFLOPs

## Common Pitfalls to Avoid
1. **Uncoalesced Memory Transactions**: Strided or unaligned memory access forces multiple 32-byte or 128-byte transactions per warp.
2. **Shared Memory Bank Conflicts**: Simultaneous access to different addresses in the same 32 SRAM banks serializes execution.
3. **Register Spilling**: Allocating too many registers per thread causes local memory spills to slow DRAM.
