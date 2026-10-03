# ViT Patchify & 3D Spatial-Temporal Embedding

## Core Mental Model & Hardware Motivation
In modern high-performance AI systems, ViT Patchify & 3D Spatial-Temporal Embedding (Vision & Video) serves as a critical computation bottleneck.
The goal of this kernel is:
patch_token[p, d] = sum_{c, h, w}(pixel[p, c, h, w] * proj_weight[c, h, w, d])

### Hardware Context
* **Arithmetic Intensity**: Transforming raw (B, C, T, H, W) video tensors into linear token projections
* **Memory Access Pattern**: Coalesced DRAM accesses streaming into registers and shared SRAM.
* **Threading Layout**: Cooperative thread blocks minimizing warp divergence and bank serialization.

## Architectural Equations (Plain Text ASCII Math)
Objective:
  patch_token[p, d] = sum_{c, h, w}(pixel[p, c, h, w] * proj_weight[c, h, w, d])

Roofline Formula:
  DRAM Bandwidth = (Bytes Read + Bytes Written) / (Elapsed Seconds * 1e9) GB/s
  Throughput = Total FLOPs / (Elapsed Seconds * 1e12) TFLOPs

## Common Pitfalls to Avoid
1. **Uncoalesced Memory Transactions**: Strided or unaligned memory access forces multiple 32-byte or 128-byte transactions per warp.
2. **Shared Memory Bank Conflicts**: Simultaneous access to different addresses in the same 32 SRAM banks serializes execution.
3. **Register Spilling**: Allocating too many registers per thread causes local memory spills to slow DRAM.
