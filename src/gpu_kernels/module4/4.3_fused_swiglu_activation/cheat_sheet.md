# Cheat Sheet: Fused SwiGLU & GeLU Activation Projections

## Quick Architecture Summary
* **Category**: Activations
* **Goal**: Elementwise fused gating math: f(x, gate) = (x * sigmoid(beta * x)) * gate
* **Formula**: y = silu(gate) * x = (gate / (1 + exp(-gate))) * x

## Key Execution Rules
* **Thread Block Size**: 256 or 512 threads per block recommended for balanced occupancy.
* **Register Limit**: Keep per-thread register count <= 64 to maintain full SM warp residency.
* **SRAM Padding**: Add +1 stride padding (`__shared__ float tile[32][33]`) to eliminate bank conflicts.

## Performance Checklist
- [x] Vectorized 128-bit memory transfers (`float4` or `__nv_bfloat162`).
- [x] Warp shuffle reduction (`__shfl_down_sync`) instead of shared memory for warp-level reductions.
- [x] Zero global DRAM intermediate writes.
