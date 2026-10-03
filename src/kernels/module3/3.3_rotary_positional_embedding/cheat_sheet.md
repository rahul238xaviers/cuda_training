# Cheat Sheet: Rotary Positional Embedding (RoPE)

## Quick Architecture Summary
* **Category**: Sequence Models
* **Goal**: Fused in-place complex rotation on Query and Key head dimensions
* **Formula**: R(x, m) = [x0 * cos(m*theta) - x1 * sin(m*theta), x0 * sin(m*theta) + x1 * cos(m*theta)]

## Key Execution Rules
* **Thread Block Size**: 256 or 512 threads per block recommended for balanced occupancy.
* **Register Limit**: Keep per-thread register count <= 64 to maintain full SM warp residency.
* **SRAM Padding**: Add +1 stride padding (`__shared__ float tile[32][33]`) to eliminate bank conflicts.

## Performance Checklist
- [x] Vectorized 128-bit memory transfers (`float4` or `__nv_bfloat162`).
- [x] Warp shuffle reduction (`__shfl_down_sync`) instead of shared memory for warp-level reductions.
- [x] Zero global DRAM intermediate writes.
