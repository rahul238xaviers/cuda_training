# Topic 4.1: Memory Coalescing & Read-Only Cache — Cheat Sheet

A concise reference for 128-byte cache line transactions and `__ldg()` read-only caching.

---

### 1. Global Memory Coalescing Rules
- GPU memory controllers fetch data from off-chip DRAM in aligned **32-byte or 128-byte transactions**.
- **Coalesced Access (100% Efficiency)**:
  - Thread 0 reads byte 0..3, Thread 1 reads byte 4..7, ..., Thread 31 reads byte 124..127.
  - The entire warp request is satisfied in exactly **one 128-byte transaction**.
- **Uncoalesced Stride Access (3% Efficiency)**:
  - If threads read with a large stride, each thread lands in a different cache line, requiring 32 separate 128-byte memory transactions to load only 128 bytes of useful data!

---

### 2. Compiler Restrict & Read-Only Cache (`__ldg`)
- When pointers are marked `const T* __restrict__`, the compiler routes DRAM reads through the **Read-Only Data Cache** (Texture Cache):
  ```cpp
  float val = __ldg(&in[idx]);
  ```
- **Benefits**:
  - Bypasses L1 write-back cache, eliminating cache pollution.
  - Provides dedicated bandwidth for read-only model weights.

---

### 3. LLM Systems Context
- **Weight Streaming in GEMM**: LLM weight matrices (billions of parameters) are read-only and benefit directly from `__restrict__` and `__ldg()` caching.

---

### 4. Common Pitfalls
- Accessing struct-of-arrays (SoA) vs array-of-structs (AoS):
  - AoS (`struct { float x, y, z; } arr[N];`): Causes strided, uncoalesced memory reads.
  - SoA (`float x[N], y[N], z[N];`): Guarantees coalesced streaming reads for each feature.
