# Fused AdamW Optimizer with FP32 Master Weights

## Core Mental Model & Hardware Motivation
In modern high-performance AI systems, Fused AdamW Optimizer with FP32 Master Weights (Optimizers) serves as a critical computation bottleneck.
The goal of this kernel is:
m = beta1*m + (1-beta1)*g; v = beta2*v + (1-beta2)*g^2; p = p - lr*(m_hat/(sqrt(v_hat)+eps) + wd*p)

### Hardware Context
* **Arithmetic Intensity**: Multi-tensor fused update: gradient, first moment (m), second moment (v), and weight decay
* **Memory Access Pattern**: Coalesced DRAM accesses streaming into registers and shared SRAM.
* **Threading Layout**: Cooperative thread blocks minimizing warp divergence and bank serialization.

## Architectural Equations (Plain Text ASCII Math)
Objective:
  m = beta1*m + (1-beta1)*g; v = beta2*v + (1-beta2)*g^2; p = p - lr*(m_hat/(sqrt(v_hat)+eps) + wd*p)

Roofline Formula:
  DRAM Bandwidth = (Bytes Read + Bytes Written) / (Elapsed Seconds * 1e9) GB/s
  Throughput = Total FLOPs / (Elapsed Seconds * 1e12) TFLOPs

## Common Pitfalls to Avoid
1. **Uncoalesced Memory Transactions**: Strided or unaligned memory access forces multiple 32-byte or 128-byte transactions per warp.
2. **Shared Memory Bank Conflicts**: Simultaneous access to different addresses in the same 32 SRAM banks serializes execution.
3. **Register Spilling**: Allocating too many registers per thread causes local memory spills to slow DRAM.
