# 3.3 Thread Blocks, Shared Memory & Bank Conflicts

## 1. Block Execution & Shared Memory Architecture

A **Thread Block** (or Threadgroup in Apple Metal) is a cooperative group of up to 1024 threads scheduled on a single Streaming Multiprocessor (SM).

Threads in the same block share high-speed on-chip SRAM called **Shared Memory (`__shared__`)**:
- Latency: ~20–30 clock cycles (compared to ~400–600 cycles for global DRAM).
- Size: Configurable up to 48KB–164KB per SM depending on architecture (Ampere/Hopper).

---

## 2. Shared Memory Banking & 32 Banks

Shared memory is split into **32 equally sized memory banks** organized in 4-byte (32-bit) words:
$$\text{Bank ID} = \left( \frac{\text{Byte Address}}{4} \right) \pmod{32}$$

### 2.1 Conflict-Free Access
- **Linear / Stride 1 Access**: Thread $i$ in a warp reads word $i$.
  - Thread 0 $\rightarrow$ Bank 0
  - Thread 1 $\rightarrow$ Bank 1
  - ...
  - Thread 31 $\rightarrow$ Bank 31
  All 32 requests are serviced simultaneously in **1 cycle**.
- **Broadcast**: Multiple threads reading the exact same address in the same bank are serviced in 1 broadcast cycle.

### 2.2 Bank Conflicts & The Stride Disaster
A **Bank Conflict** occurs when multiple threads in a warp access *different addresses* that map to the *same bank*. The memory controller serializes the conflicting requests!

Example: Column access of a $32 \times 32$ matrix in shared memory:
```cpp
__shared__ float tile[32][32]; // row-major
float val = tile[threadIdx.x][0]; // Column access!
```
- Thread 0 accesses index `0 * 32 + 0 = 0` $\rightarrow$ Bank 0
- Thread 1 accesses index `1 * 32 + 0 = 32` $\rightarrow$ Bank $(32 \pmod{32}) = 0$
- Thread 2 accesses index `2 * 32 + 0 = 64` $\rightarrow$ Bank $(64 \pmod{32}) = 0$
All 32 threads target Bank 0! This is a **32-way bank conflict** causing a 32x throughput slowdown.

### 2.3 The Padding Solution
By adding 1 dummy column (`+1` padding), successive rows are offset by 1 bank:
```cpp
__shared__ float tile[32][33]; // Padded column
```
- Thread 0 accesses index $0 \times 33 = 0 \rightarrow$ Bank 0
- Thread 1 accesses index $1 \times 33 = 33 \rightarrow$ Bank 1
- Thread 2 accesses index $2 \times 33 = 66 \rightarrow$ Bank 2
Zero bank conflicts! All 32 accesses occur in parallel.

---

## 3. Dynamic Shared Memory & Barriers

### 3.1 Dynamic Shared Memory Allocation
When the shared memory size is determined at runtime:
```cpp
extern __shared__ float s_mem[];
// Launched with: my_kernel<<<grid, block, shared_bytes>>>(...);
```

### 3.2 Block Barrier Synchronization
`__syncthreads()` guarantees that all threads in the block have reached this point and all shared memory writes are visible.
**Warning**: If `__syncthreads()` is placed inside a conditional branch where not all threads enter, the GPU will deadlock!
