# Cheat Sheet: FlashAttention-2 Forward Kernel

## Quick Architecture Summary
* **Category**: Attention Mechanisms
* **Goal**: Online softmax, SRAM tiling of Q, K, V blocks, and zero DRAM N x N materialization
* **Formula**: O = softmax(Q * K^T / sqrt(d)) * V using online streaming max and sum-exp in SRAM

## Key Execution Rules
* **Thread Block Size**: 256 or 512 threads per block recommended for balanced occupancy.
* **Register Limit**: Keep per-thread register count <= 64 to maintain full SM warp residency.
* **SRAM Padding**: Add +1 stride padding (`__shared__ float tile[32][33]`) to eliminate bank conflicts.

## Performance Checklist
- [x] Vectorized 128-bit memory transfers (`float4` or `__nv_bfloat162`).
- [x] Warp shuffle reduction (`__shfl_down_sync`) instead of shared memory for warp-level reductions.
- [x] Zero global DRAM intermediate writes.
