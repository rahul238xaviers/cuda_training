# Cheat Sheet: Numerically Stable Block-Wide Online Softmax

## Quick Architecture Summary
* **Category**: Activations
* **Goal**: Single-pass streaming online softmax without materializing global maximum
* **Formula**: p_i = exp(x_i - max(x)) / sum(exp(x - max(x))) in 1 unified block pass

## Key Execution Rules
* **Thread Block Size**: 256 or 512 threads per block recommended for balanced occupancy.
* **Register Limit**: Keep per-thread register count <= 64 to maintain full SM warp residency.
* **SRAM Padding**: Add +1 stride padding (`__shared__ float tile[32][33]`) to eliminate bank conflicts.

## Performance Checklist
- [x] Vectorized 128-bit memory transfers (`float4` or `__nv_bfloat162`).
- [x] Warp shuffle reduction (`__shfl_down_sync`) instead of shared memory for warp-level reductions.
- [x] Zero global DRAM intermediate writes.
