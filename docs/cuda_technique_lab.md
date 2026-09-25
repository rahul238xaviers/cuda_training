# CUDA Technique Lab — Practice Guide

**Purpose**: learn each CUDA technique in isolation. Every lab is self-contained:
**Given** the inputs, **Task** is what you write, **Verify** is the pass bar.
Write each kernel by hand — the point is the pattern lands in your hands.

**Lab discipline** (from `docs/cuda_kernel_rules.md`, always in force):
- `(x, y)` — x = cols first, y = rows second.
- block rules, checked in order:
  1. **R3 legal** — block area ≤ `maxThreadsPerBlock` (a device property, read
     with `cudaDeviceProp.maxThreadsPerBlock` or `gpu_check.cu`). Ex: GTX 1650 card
     reports 1024, so block (32, 32) = 1024 is legal, (32, 33) = 1056 is not.
  2. **R1 warp-clean** — block area ÷ `warpSize` (32) is a whole number of warps.
     Ex: 128 ÷ 32 = 4 ✓ | 160 ÷ 32 = 5 ✓ | 100 ÷ 32 = 3.125 ✗.
  3. **R2 occupancy** — `maxThreadsPerMultiProcessor` (1024 on this card) ÷ block
     area = integer blocks/SM. Ex: 1024 ÷ 128 = 8 ✓ | 1024 ÷ 320 = 3.2 ✗.
- grid = `ceil(cols/blockDim.x)` × `ceil(rows/blockDim.y)`.
- guard: `if (row < rows && col < cols)`.
- host memory family `h_*` (malloc), device family `d_*` (cudaMalloc), `cudaMemcpy` is the only bridge, kernel args are always `d_*`.

**Compile** for every lab:
```
nvcc -O2 -arch=sm_75 --ptxas-options=-v src/cuda/<file>.cu -o output/<bin>
```
Read the ptxas register/spill line after every build.

---

# Lab 1 — Grid-Stride Loop

**Concept**: one thread serves many elements. A fixed grid walks the whole
array, each thread stepping by `stride = blockDim.x * gridDim.x`, touching
elements `{ start, start+stride, start+2*stride, ... }`.

**Why fixed grid**: total threads per launch is capped, so an array of 2^28
elements cannot have one thread per element. Covering any n with a modest fixed
grid is the technique.

**Given**:
```
x[ n ],  n = 271,000,000  (1 << 28)
block  = 256 threads,  grid = 512 blocks   → total threads = 131,072
y[ n ] = output, same size
```

**Task**: write `addHalfKernel`:
```
y[i] = x[i] + 0.5f      for all i in [0, n), strided
```

**Rules**: exactly the loop
```
stride = blockDim.x * gridDim.x
for (i = blockIdx.x*blockDim.x + threadIdx.x ; i < n ; i += stride)
```

**Verify** (host, after `cudaMemcpy` DeviceToHost):
```
fabs( y[i] - (x[i] + 0.5f) ) <= 1e-4f   for all i    →  PASS
```
Time it with `cudaEvent` (GPU) and `std::chrono` (CPU add loop).

**Compute these by hand before running** (self-check):
1. `stride` = ?
2. elements served by one thread on average = n ÷ total threads = ?
3. how many threads of the 131,072 do NO work? (hint: n % total-threads)

**Answer** (check after you solve): stride = 131,072; avg 2048 elements/thread;
0 idle threads — 2^28 = 4 × 131,072 exactly.

| API | Signature | What it does |
|---|---|---|
| `blockDim` | built-in `dim3` | threads per block |
| `gridDim` | built-in `dim3` | blocks per grid |
| `threadIdx` | built-in `dim3` | index inside the block |
| `blockIdx` | built-in `dim3` | index of the block |

---

# Lab 2 — Block Reduction

**Concept**: threads accumulate their own totals, then cooperate inside one
block to combine them. Requires `__shared__` (on-chip per-block array) and
`__syncthreads()` (block-wide barrier — every thread must reach it) to avoid
races, and `atomicAdd` so block totals land in one output safely.

**Given**:
```
x[ n ],  n = 2^20 (1,048,576)
block  = 256 threads,  grid = 256 blocks
out[1] = single float, pre-zeroed with cudaMemset
```

**Task**: write `reduceKernel` producing `out = Σ x[i]`:
```
1. stride-accumulate:  threadLocal += x[i]   (grid-stride over n)
2. blockSum[threadIdx.x] = threadLocal ; __syncthreads()
3. halving tree:  for s = blockDim.x/2; s > 0; s >>= 1
                     if (threadIdx.x < s) blockSum[i] += blockSum[i+s]
                     __syncthreads()    (inside the loop, every rung)
4. if (threadIdx.x == 0) atomicAdd(out, blockSum[0])
```

**Verify**:
```
sum over i of x[i] (CPU, computed in double)  vs  *out
diff ÷ |cpu|  <=  1e-4f   →  PASS
```

| API | Signature | What it does |
|---|---|---|
| `__shared__` | `__shared__ T name[C]` | on-chip per-block array, visible to the whole block |
| `__syncthreads()` | `void` | block barrier — all threads must arrive |
| `atomicAdd` | `T atomicAdd(T* addr, T val)` | `*addr += val`, race-free |
| `cudaMemset` | `cudaMemset(ptr, 0, bytes)` | zero `out` before launch |

---

# Lab 3 — Two-Stage Reduction

**Concept**: one block can't hold a huge array's total — it needs many blocks,
and blocks can't share results. So: **kernel 1** turns each block's slice into a
partial written to its own slot; **kernel 2** (one block) sums the few partials.
No atomics between blocks — each writes a different slot.

**Given**:
```
x[ n ],  n = 2^23 (8,388,608)
blocks = 512,  threads = 256
scratch[512]        pre-zeroed
out[1]
```

**Task**:
1. `partialKernel` = Lab 2's loop, but final step is
   ```
   if (threadIdx.x == 0)  scratch[blockIdx.x] = blockSum[0];
   ```
2. `finalKernel` = ONE block, 256 threads: each thread pulls one `scratch[i]`,
   same halving tree on 256 values, thread 0 writes `*out`.

**Verify**: same as Lab 2 — CPU double sum vs `*out`, relative diff ≤ 1e-4.

**Why two kernels instead of one atomic per block**: 512 workers writing 512
*different* slots is parallel; 512 atomics on ONE slot would serialize.

| API | Signature | What it does |
|---|---|---|
| `cudaMalloc` | `cudaMalloc(&ptr, bytes)` | allocate device VRAM |
| `cudaMemcpy` | `cudaMemcpy(dst, src, bytes, kind)` | transfer host ⇄ device |
| `cudaMemcpyDeviceToHost` | enum | device → host |
| `cudaMemcpyHostToDevice` | enum | host → device |
| `cudaDeviceSynchronize` | `void` | block CPU until GPU work done |

---

# Lab 4 — Shared Memory Tiling

**Concept**: stage a chunk of global into shared ONCE, sync, then read shared
many times (~20 cycle reads vs ~500 from global). The reusable shape:
**Stage → Sync → Consume**. This is a GEMM tile in miniature.

**Given**:
```
x[ n ],  n = 4096
TILE  = 128,  threads = 128,  blocks = 32   (4096 ÷ 128)
y[ n ]
```

**Task**: write `tileReverseKernel`:
```
1. base = blockIdx.x * TILE
2. STAGE:   for (i = threadIdx.x; i < TILE; i += threads)  tile[i] = x[base+i]
3. SYNC:    __syncthreads()
4. CONSUME: for (i = threadIdx.x; i < TILE; i += threads)
              y[ base + (TILE-1-i) ] = tile[i]
```

**Verify**:
```
y[ base + (TILE-1-i) ] == x[ base + i ]   for all tiles, all i   →  PASS
```

**Rule-check** (do by hand): threads 128 → R1: 128÷32 = 4 ✓ warp-clean;
R2: 1024÷128 = 8 ✓.

| API | Signature | What it does |
|---|---|---|
| `__shared__` | keyword | the hero — fast on-chip window |
| `__syncthreads()` | `void` | MANDATORY between stage and consume |

---

# Lab 5 — Warp Shuffle Reduction

**Concept**: reduce within a warp (32 lanes) using register exchange — no shared
memory, no barriers. Each lane sends its value to lane−delta (`__shfl_down_sync`
with `offset` = 16, 8, 4, 2, 1); lane 0 ends with the warp sum. Then reduce the
few warp-sums with shared memory.

**Given**:
```
x[ n ],  n = 2^20
block   = 256 threads  →  8 warps per block
grid    = 256 blocks
out[1], pre-zeroed
```

**Task**: write `shuffleReduceKernel`:
```
1. lane = threadIdx.x & 31     wid = threadIdx.x >> 5
2. stride-accumulate v += x[i]
3. for (off = 16; off > 0; off >>= 1)  v += __shfl_down_sync(0xffffffff, v, off)
4. if (lane == 0) sResult[wid] = v          // shared[8]
5. __syncthreads()
6. threads 0..7 sum sResult[0..7];  thread 0 atomicAdd(out, total)
```

**Verify**: Lab 3's bar — CPU double sum vs `*out`, relative diff ≤ 1e-4.

**Why better than shared for ≤32 lanes**: 6 register hops, zero memory, zero
barriers.

| API | Signature | What it does |
|---|---|---|
| `__shfl_down_sync` | `T __shfl_down_sync(unsigned mask, T v, int delta, int width=32)` | lane receives `v` from lane+delta |
| `threadIdx.x & 31` | bitmask | lane id |
| `threadIdx.x >> 5` | right shift | warp id |

---

# Lab 6 — Atomic Scatter to 2D

**Concept**: many threads → few destinations. Plain `R[idx] += 1` is a
read-modify-write; two threads can interleave and lose an update. `atomicAdd`
serializes it. Real use: gradient accumulation in `fused_attn_bwd`.

**Given**:
```
R[ rows*cols ],  rows = cols = 64        (4096 cells)
1 block, 1024 threads                     (max block)
```

**Task**: write `scatterKernel`:
```
r = ( threadIdx.x * 2654435761u ) % rows
c = ( threadIdx.x * 1597334677u ) % cols
atomicAdd( &R[ r*cols + c ], 1.0f )
```

**Verify**:
```
Σ over all cells of R  ==  1024   (exactly)   →  PASS
```

**Why the two constants**: large odd multipliers spread threads across the
4096 cells so the test isn't "everyone hit the same cell". The test only checks
total.

| API | Signature | What it does |
|---|---|---|
| `atomicAdd` | `T atomicAdd(T* addr, T v)` | `*addr += v`, race-free |
| `threadIdx.x` | built-in | per-thread identity source |

---

# After the Lab — mapping to real kernels

```
grid-stride loop      →  adamw_step, residual_add
block reduce + barrier →  rms_norm, cross_entropy
two-stage reduce      →  cross_entropy sum, softmax normalization
shared tile + sync    →  gemm family
warp shuffle          →  sub-32 reductions, tails
atomic scatter        →  fused_attn_bwd gradient accumulation
```

Every lab verified against a CPU reference with a tolerance, per the
`docs/cuda_kernel_rules.md` discipline.