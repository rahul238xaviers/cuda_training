# Rotary Positional Embedding (RoPE)

## Core Mental Model & Hardware Motivation
In modern high-performance AI systems, Rotary Positional Embedding (RoPE) (Sequence Models) serves as a critical computation bottleneck.
The goal of this kernel is:
R(x, m) = [x0 * cos(m*theta) - x1 * sin(m*theta), x0 * sin(m*theta) + x1 * cos(m*theta)]

### Hardware Context
* **Arithmetic Intensity**: Fused in-place complex rotation on Query and Key head dimensions
* **Memory Access Pattern**: Coalesced DRAM accesses streaming into registers and shared SRAM.
* **Threading Layout**: Cooperative thread blocks minimizing warp divergence and bank serialization.

## Architectural Equations (Plain Text ASCII Math)
Objective:
  R(x, m) = [x0 * cos(m*theta) - x1 * sin(m*theta), x0 * sin(m*theta) + x1 * cos(m*theta)]

Roofline Formula:
  DRAM Bandwidth = (Bytes Read + Bytes Written) / (Elapsed Seconds * 1e9) GB/s
  Throughput = Total FLOPs / (Elapsed Seconds * 1e12) TFLOPs

## Common Pitfalls to Avoid
1. **Uncoalesced Memory Transactions**: Strided or unaligned memory access forces multiple 32-byte or 128-byte transactions per warp.
2. **Shared Memory Bank Conflicts**: Simultaneous access to different addresses in the same 32 SRAM banks serializes execution.
3. **Register Spilling**: Allocating too many registers per thread causes local memory spills to slow DRAM.
