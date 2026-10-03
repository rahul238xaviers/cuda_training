# Cheat Sheet: Linear Attention & State-Space Model (SSM) Scan

## Quick Architecture Summary
* **Category**: Sequence Models
* **Goal**: Mamba-style associative parallel scan over recurrent hidden state
* **Formula**: h_t = a_t * h_{t-1} + b_t * x_t using parallel prefix associative scan

## Key Execution Rules
* **Thread Block Size**: 256 or 512 threads per block recommended for balanced occupancy.
* **Register Limit**: Keep per-thread register count <= 64 to maintain full SM warp residency.
* **SRAM Padding**: Add +1 stride padding (`__shared__ float tile[32][33]`) to eliminate bank conflicts.

## Performance Checklist
- [x] Vectorized 128-bit memory transfers (`float4` or `__nv_bfloat162`).
- [x] Warp shuffle reduction (`__shfl_down_sync`) instead of shared memory for warp-level reductions.
- [x] Zero global DRAM intermediate writes.
