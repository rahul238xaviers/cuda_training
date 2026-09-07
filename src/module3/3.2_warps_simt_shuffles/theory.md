# 3.2 Warps, SIMT Execution & Shuffle Primitives

## 1. Warp Architecture & SIMT Execution

On NVIDIA GPUs, the hardware schedules threads in groups of 32 known as a **Warp**.
- A warp is executed by 32 SIMD execution units on a Streaming Multiprocessor (SM) in lockstep.
- All 32 threads in a warp execute the same instruction at the same clock cycle.
- The thread index within a warp (0 to 31) is called the **lane ID**:
  ```cpp
  int lane_id = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;
  ```

---

## 2. Warp Divergence

When threads within the same warp follow different execution paths due to data-dependent branching (`if/else`):
```cpp
if (threadIdx.x % 2 == 0) {
    do_even();  // Only even lanes active; odd lanes idle (masked off)
} else {
    do_odd();   // Only odd lanes active; even lanes idle (masked off)
}
```
The GPU serializes both paths! The total execution time becomes the sum of both branches. To maximize throughput:
- Ensure all 32 threads in a warp take the same branch whenever possible.
- Use warp-aligned tiling and branch granularity.

---

## 3. Intra-Warp Shuffle Primitives (Register-to-Register)

Historically, threads had to communicate via Shared Memory (`__shared__`) followed by `__syncthreads()`. Since Kepler (sm_30) and Volta (sm_70), CUDA provides **Warp Shuffle Intrinsics**: direct register-to-register data exchange between lanes in the same warp in a single clock cycle, bypassing shared memory entirely.

All shuffle primitives require an **active thread mask** (typically `0xffffffff` for all 32 threads active):

### 3.1 `__shfl_sync`
Broadcasts a value from a specific source lane `srcLane` (0–31) to all threads in the warp:
```cpp
float broadcasted = __shfl_sync(0xffffffff, my_val, 0); // Lane 0 broadcasts its value
```

### 3.2 `__shfl_down_sync`
Shifts a value down from a lane with higher index: lane $i$ receives the value from lane $i + \text{delta}$:
```cpp
float neighbor = __shfl_down_sync(0xffffffff, my_val, delta);
```
This is the fundamental building block of tree-based warp reductions.

### 3.3 `__shfl_xor_sync`
Performs a butterfly exchange based on bitwise XOR with `laneMask`:
```cpp
float exchanged = __shfl_xor_sync(0xffffffff, my_val, 16);
```

---

## 4. Canonical 32-Thread Warp Reduction

A warp of 32 threads can compute the sum (or max/min) of all 32 elements in exactly $\log_2(32) = 5$ instructions:
```cpp
__device__ inline float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val; // Lane 0 contains the sum of all 32 lanes
}
```
Lane 0 can then broadcast this value or write it to global/shared memory.
