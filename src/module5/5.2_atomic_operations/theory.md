# 5.2 Atomic Operations & Contention Reduction

## 1. Hardware Atomic Operations in CUDA

An **Atomic Operation** performs a Read-Modify-Write (RMW) cycle on a memory address without interruption from other threads:
- No two threads can access the same address concurrently during an atomic operation.
- In modern NVIDIA GPUs (Pascal sm_60+ and Ampere/Hopper), atomics are executed natively at the **L2 Cache** controller or shared memory controller.

Common atomic functions:
- `atomicAdd(address, val)`
- `atomicSub(address, val)`
- `atomicMin(address, val)` / `atomicMax(address, val)`
- `atomicExch(address, val)`: Unconditionally swaps the value.
- `atomicCAS(address, compare, val)`: Compare-And-Swap, the foundation for lock-free concurrency and custom atomic types.

---

## 2. Contention & The "Atomic Bottleneck"

If 100,000 threads simultaneously execute:
```cpp
atomicAdd(global_loss_ptr, thread_loss);
```
All 100,000 threads are serialized at the L2 cache controller, dropping execution speed by orders of magnitude!

### Contention Mitigation Hierarchy
To achieve peak throughput in loss aggregations and optimizer gradient steps:
1. **Thread-Level Aggregation**: Each thread loops over multiple elements in registers (`local_loss += ...`).
2. **Warp-Level Aggregation**: Use `__shfl_down_sync` to sum within each 32-thread warp (reduces atomic calls by 32x).
3. **Block-Level Aggregation**: Shared memory scratchpad reduces all warps in the block (reduces atomic calls by 256x–1024x).
4. **Single Global Atomic**: Only thread 0 of each block issues an `atomicAdd` to global DRAM.
   - 100,000 elements $\rightarrow$ only $\approx 100$ global atomic transactions!
