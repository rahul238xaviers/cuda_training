# Cheat Sheet: Fused AdamW Optimizer with FP32 Master Weights

## Quick Architecture Summary
* **Category**: Optimizers
* **Goal**: Multi-tensor fused update: gradient, first moment (m), second moment (v), and weight decay
* **Formula**: m = beta1*m + (1-beta1)*g; v = beta2*v + (1-beta2)*g^2; p = p - lr*(m_hat/(sqrt(v_hat)+eps) + wd*p)

## Key Execution Rules
* **Thread Block Size**: 256 or 512 threads per block recommended for balanced occupancy.
* **Register Limit**: Keep per-thread register count <= 64 to maintain full SM warp residency.
* **SRAM Padding**: Add +1 stride padding (`__shared__ float tile[32][33]`) to eliminate bank conflicts.

## Performance Checklist
- [x] Vectorized 128-bit memory transfers (`float4` or `__nv_bfloat162`).
- [x] Warp shuffle reduction (`__shfl_down_sync`) instead of shared memory for warp-level reductions.
- [x] Zero global DRAM intermediate writes.
