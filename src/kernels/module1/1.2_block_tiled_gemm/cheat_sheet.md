# Cheat Sheet: 2D Block-Tiled FP32 Shared Memory GEMM

## Quick Architecture Summary
* **Category**: Dense Linear
* **Goal**: Square SRAM tiling, collaborative DRAM staging, register micro-tile accumulation
* **Formula**: C = A * B where A is (M x K), B is (K x N), and C is (M x N)

## Key Execution Rules
* **Thread Block Size**: 256 or 512 threads per block recommended for balanced occupancy.
* **Register Limit**: Keep per-thread register count <= 64 to maintain full SM warp residency.
* **SRAM Padding**: Add +1 stride padding (`__shared__ float tile[32][33]`) to eliminate bank conflicts.

## Performance Checklist
- [x] Vectorized 128-bit memory transfers (`float4` or `__nv_bfloat162`).
- [x] Warp shuffle reduction (`__shfl_down_sync`) instead of shared memory for warp-level reductions.
- [x] Zero global DRAM intermediate writes.
