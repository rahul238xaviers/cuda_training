# Cheat Sheet: FP8 / INT4 Weight-Only GEMV with On-The-Fly Dequant

## Quick Architecture Summary
* **Category**: Quantization
* **Goal**: Packing 2 INT4 nibbles per byte, on-the-fly register unpacking, and scale-factor multiply
* **Formula**: y = sum_k(dequant(W_int4[row, k], scale[row]) * x[k]) with 75% DRAM traffic reduction

## Key Execution Rules
* **Thread Block Size**: 256 or 512 threads per block recommended for balanced occupancy.
* **Register Limit**: Keep per-thread register count <= 64 to maintain full SM warp residency.
* **SRAM Padding**: Add +1 stride padding (`__shared__ float tile[32][33]`) to eliminate bank conflicts.

## Performance Checklist
- [x] Vectorized 128-bit memory transfers (`float4` or `__nv_bfloat162`).
- [x] Warp shuffle reduction (`__shfl_down_sync`) instead of shared memory for warp-level reductions.
- [x] Zero global DRAM intermediate writes.
