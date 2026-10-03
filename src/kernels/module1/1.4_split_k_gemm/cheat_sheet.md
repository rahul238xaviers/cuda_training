# Cheat Sheet: Split-K High-Occupancy GEMM for LLM Generation

## Quick Architecture Summary
* **Category**: Dense Linear
* **Goal**: K-dimension splitting for small batch sizes (M=1) with atomic reduction
* **Formula**: C = sum_k(A_k * B_k) partitioned across multiple SM thread blocks

## Key Execution Rules
* **Thread Block Size**: 256 or 512 threads per block recommended for balanced occupancy.
* **Register Limit**: Keep per-thread register count <= 64 to maintain full SM warp residency.
* **SRAM Padding**: Add +1 stride padding (`__shared__ float tile[32][33]`) to eliminate bank conflicts.

## Performance Checklist
- [x] Vectorized 128-bit memory transfers (`float4` or `__nv_bfloat162`).
- [x] Warp shuffle reduction (`__shfl_down_sync`) instead of shared memory for warp-level reductions.
- [x] Zero global DRAM intermediate writes.
