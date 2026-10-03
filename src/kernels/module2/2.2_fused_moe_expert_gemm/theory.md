# Fused MoE SwiGLU Expert Matrix Multiply

## Core Mental Model & Hardware Motivation
In modern high-performance AI systems, Fused MoE SwiGLU Expert Matrix Multiply (Mixture of Experts) serves as a critical computation bottleneck.
The goal of this kernel is:
Expert_e(x) = (W_gate * x * sigmoid(W_gate * x)) * (W_up * x)

### Hardware Context
* **Arithmetic Intensity**: Grouped GEMM across dynamic expert tokens without global scatter/gather copying
* **Memory Access Pattern**: Coalesced DRAM accesses streaming into registers and shared SRAM.
* **Threading Layout**: Cooperative thread blocks minimizing warp divergence and bank serialization.

## Architectural Equations (Plain Text ASCII Math)
Objective:
  Expert_e(x) = (W_gate * x * sigmoid(W_gate * x)) * (W_up * x)

Roofline Formula:
  DRAM Bandwidth = (Bytes Read + Bytes Written) / (Elapsed Seconds * 1e9) GB/s
  Throughput = Total FLOPs / (Elapsed Seconds * 1e12) TFLOPs

## Common Pitfalls to Avoid
1. **Uncoalesced Memory Transactions**: Strided or unaligned memory access forces multiple 32-byte or 128-byte transactions per warp.
2. **Shared Memory Bank Conflicts**: Simultaneous access to different addresses in the same 32 SRAM banks serializes execution.
3. **Register Spilling**: Allocating too many registers per thread causes local memory spills to slow DRAM.
