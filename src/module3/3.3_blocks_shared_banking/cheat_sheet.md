# Topic 3.3: Thread Blocks, Shared SRAM & Bank Conflicts — Cheat Sheet

A concise reference for thread block limits, shared memory allocation, and 32-bank conflict rules.

---

### 1. Thread Block Physical Limits
- Max threads per block: 1024 (e.g. 32x32, 256x1, 1024x1).
- Recommended block size: Multiple of 32 (128, 256, 512) to ensure 100% warp utilization.

---

### 2. Shared Memory Banking (32 Banks)
- On-chip Shared Memory is divided into 32 independent banks, each 4 bytes wide.
- Bank assignment: `Bank_ID = (Byte_Address / 4) % 32 = Word_Index % 32`.
- **Zero Conflict (Peak Speed)**: All 32 threads access different banks simultaneously, or all threads read the exact same address (Broadcast).
- **N-Way Conflict (Slowdown)**: When N threads in a warp access different words mapped to the same bank, the accesses are serialized by N times.

---

### 3. Eliminating Bank Conflicts (Stride Padding)
- **Problem**: 32x32 float array in shared memory: `__shared__ float tile[32][32];`
  - Accessing a column `tile[threadIdx.x][0]` causes a **32-way bank conflict** because every row stride is 32 (all column elements land in Bank 0!).
- **Solution (Padding by +1)**:
  ```cpp
  __shared__ float tile[32][33]; // Pad inner dimension by 1
  ```
  - Stride becomes 33: `(row * 33) % 32 = row % 32`. Every thread lands in a unique bank!

---

### 4. Barrier Synchronization
```cpp
__syncthreads(); // Synchronizes all threads within the thread block
```

---

### 5. LLM Systems Context
- **FlashAttention & Tiled GEMM**: Tiling Query, Key, and Value blocks into padded shared memory with zero bank conflicts.
