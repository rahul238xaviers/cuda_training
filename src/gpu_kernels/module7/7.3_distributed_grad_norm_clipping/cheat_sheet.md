# Cheat Sheet: Distributed Gradient Norm Clipping & Loss Scaling

## Quick Architecture Summary
* **Category**: Optimizers
* **Goal**: Two-stage tree reduction of squared gradient norms and in-place scaling
* **Formula**: total_norm = sqrt(sum(|g|^2)); if (total_norm > max_norm) g = g * (max_norm / total_norm)

## Key Execution Rules
* **Thread Block Size**: 256 or 512 threads per block recommended for balanced occupancy.
* **Register Limit**: Keep per-thread register count <= 64 to maintain full SM warp residency.
* **SRAM Padding**: Add +1 stride padding (`__shared__ float tile[32][33]`) to eliminate bank conflicts.

## Performance Checklist
- [x] Vectorized 128-bit memory transfers (`float4` or `__nv_bfloat162`).
- [x] Warp shuffle reduction (`__shfl_down_sync`) instead of shared memory for warp-level reductions.
- [x] Zero global DRAM intermediate writes.
