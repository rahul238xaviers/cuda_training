# Cheat Sheet: 1D Short-Time Fourier Transform (STFT) & Mel-Filterbank

## Quick Architecture Summary
* **Category**: Audio Processing
* **Goal**: Sliding window Hann tapering, fast Cooley-Tukey FFT in SRAM, and Mel filterbank projection
* **Formula**: Mel_m = sum_k(Filter[m, k] * |FFT(window[t] * Hann)|^2)

## Key Execution Rules
* **Thread Block Size**: 256 or 512 threads per block recommended for balanced occupancy.
* **Register Limit**: Keep per-thread register count <= 64 to maintain full SM warp residency.
* **SRAM Padding**: Add +1 stride padding (`__shared__ float tile[32][33]`) to eliminate bank conflicts.

## Performance Checklist
- [x] Vectorized 128-bit memory transfers (`float4` or `__nv_bfloat162`).
- [x] Warp shuffle reduction (`__shfl_down_sync`) instead of shared memory for warp-level reductions.
- [x] Zero global DRAM intermediate writes.
