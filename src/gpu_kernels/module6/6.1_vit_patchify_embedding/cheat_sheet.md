# Cheat Sheet: ViT Patchify & 3D Spatial-Temporal Embedding

## Quick Architecture Summary
* **Category**: Vision & Video
* **Goal**: Transforming raw (B, C, T, H, W) video tensors into linear token projections
* **Formula**: patch_token[p, d] = sum_{c, h, w}(pixel[p, c, h, w] * proj_weight[c, h, w, d])

## Key Execution Rules
* **Thread Block Size**: 256 or 512 threads per block recommended for balanced occupancy.
* **Register Limit**: Keep per-thread register count <= 64 to maintain full SM warp residency.
* **SRAM Padding**: Add +1 stride padding (`__shared__ float tile[32][33]`) to eliminate bank conflicts.

## Performance Checklist
- [x] Vectorized 128-bit memory transfers (`float4` or `__nv_bfloat162`).
- [x] Warp shuffle reduction (`__shfl_down_sync`) instead of shared memory for warp-level reductions.
- [x] Zero global DRAM intermediate writes.
