# Cheat Sheet: 1D Causal Dilated Convolution for Waveforms

## Quick Architecture Summary
* **Category**: Audio Processing
* **Goal**: WaveNet / AudioGen causal convolutions with exponential dilation and residual connections
* **Formula**: y[t] = sum_k(weight[k] * x[t - k * dilation]) with zero future leakage

## Key Execution Rules
* **Thread Block Size**: 256 or 512 threads per block recommended for balanced occupancy.
* **Register Limit**: Keep per-thread register count <= 64 to maintain full SM warp residency.
* **SRAM Padding**: Add +1 stride padding (`__shared__ float tile[32][33]`) to eliminate bank conflicts.

## Performance Checklist
- [x] Vectorized 128-bit memory transfers (`float4` or `__nv_bfloat162`).
- [x] Warp shuffle reduction (`__shfl_down_sync`) instead of shared memory for warp-level reductions.
- [x] Zero global DRAM intermediate writes.
