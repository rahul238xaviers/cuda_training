# Distributed Gradient Norm Clipping & Loss Scaling

## Core Mental Model & Hardware Motivation
In modern high-performance AI systems, Distributed Gradient Norm Clipping & Loss Scaling (Optimizers) serves as a critical computation bottleneck.
The goal of this kernel is:
total_norm = sqrt(sum(|g|^2)); if (total_norm > max_norm) g = g * (max_norm / total_norm)

### Hardware Context
* **Arithmetic Intensity**: Two-stage tree reduction of squared gradient norms and in-place scaling
* **Memory Access Pattern**: Coalesced DRAM accesses streaming into registers and shared SRAM.
* **Threading Layout**: Cooperative thread blocks minimizing warp divergence and bank serialization.

## Architectural Equations (Plain Text ASCII Math)
Objective:
  total_norm = sqrt(sum(|g|^2)); if (total_norm > max_norm) g = g * (max_norm / total_norm)

Roofline Formula:
  DRAM Bandwidth = (Bytes Read + Bytes Written) / (Elapsed Seconds * 1e9) GB/s
  Throughput = Total FLOPs / (Elapsed Seconds * 1e12) TFLOPs

## Common Pitfalls to Avoid
1. **Uncoalesced Memory Transactions**: Strided or unaligned memory access forces multiple 32-byte or 128-byte transactions per warp.
2. **Shared Memory Bank Conflicts**: Simultaneous access to different addresses in the same 32 SRAM banks serializes execution.
3. **Register Spilling**: Allocating too many registers per thread causes local memory spills to slow DRAM.
