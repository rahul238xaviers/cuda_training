# Cheat Sheet: 2D/3D Windowed Video Attention (Swin/Video-LLM)

## Quick Architecture Summary
* **Category**: Vision & Video
* **Goal**: Local 3D spatial-temporal attention windows with cyclic shifting and masking
* **Formula**: Local_Attn = softmax(Q_w * K_w^T / sqrt(d) + Mask_w) * V_w

## Key Execution Rules
* **Thread Block Size**: 256 or 512 threads per block recommended for balanced occupancy.
* **Register Limit**: Keep per-thread register count <= 64 to maintain full SM warp residency.
* **SRAM Padding**: Add +1 stride padding (`__shared__ float tile[32][33]`) to eliminate bank conflicts.

## Performance Checklist
- [x] Vectorized 128-bit memory transfers (`float4` or `__nv_bfloat162`).
- [x] Warp shuffle reduction (`__shfl_down_sync`) instead of shared memory for warp-level reductions.
- [x] Zero global DRAM intermediate writes.
