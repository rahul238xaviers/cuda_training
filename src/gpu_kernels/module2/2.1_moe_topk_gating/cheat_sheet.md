# Cheat Sheet: MoE Top-K Gating & Routing Softmax

## Quick Architecture Summary
* **Category**: Mixture of Experts
* **Goal**: Argmax Top-K selection, gating softmax normalization, and token-to-expert sorting
* **Formula**: top_k_indices, top_k_weights = top_k(softmax(logits), k=2)

## Key Execution Rules
* **Thread Block Size**: 256 or 512 threads per block recommended for balanced occupancy.
* **Register Limit**: Keep per-thread register count <= 64 to maintain full SM warp residency.
* **SRAM Padding**: Add +1 stride padding (`__shared__ float tile[32][33]`) to eliminate bank conflicts.

## Performance Checklist
- [x] Vectorized 128-bit memory transfers (`float4` or `__nv_bfloat162`).
- [x] Warp shuffle reduction (`__shfl_down_sync`) instead of shared memory for warp-level reductions.
- [x] Zero global DRAM intermediate writes.
