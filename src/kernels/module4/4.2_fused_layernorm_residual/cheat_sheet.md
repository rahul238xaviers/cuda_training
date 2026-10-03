# Cheat Sheet: Fused LayerNorm with Residual Add & Bias

## Quick Architecture Summary
* **Category**: Normalizations
* **Goal**: Three-in-one kernel: residual addition, mean/variance calculation, and normalization
* **Formula**: x_res = x + res; y = ((x_res - mean) / sqrt(var + eps)) * gamma + beta

## Key Execution Rules
* **Thread Block Size**: 256 or 512 threads per block recommended for balanced occupancy.
* **Register Limit**: Keep per-thread register count <= 64 to maintain full SM warp residency.
* **SRAM Padding**: Add +1 stride padding (`__shared__ float tile[32][33]`) to eliminate bank conflicts.

## Performance Checklist
- [x] Vectorized 128-bit memory transfers (`float4` or `__nv_bfloat162`).
- [x] Warp shuffle reduction (`__shfl_down_sync`) instead of shared memory for warp-level reductions.
- [x] Zero global DRAM intermediate writes.
