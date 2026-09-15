# CUDA Kernel Design — Rules & Workflow

**Author**: Rahul Kumar
**Purpose**: One place to revise the kernel-design workflow. It follows the order you actually design in: you are GIVEN an array, you pick a block, you compute the grid, and only then do you write index formulas.

**Core philosophy**: You start with real data. Everything else flows from that.

---

## Check the Graphics Card details

**Source**: [`src/cuda/gpu_check.cu`](../src/cuda/gpu_check.cu)

Compile & run:
```bash
nvcc -O2 -arch=sm_75 src/cuda/gpu_check.cu -o output/gpu_check
./output/gpu_check
```

## 0. Notation Conventions

- CUDA notation is **`(x, y)`** — **x first = columns / width**, **y second = rows / height**.
  Applies to `grid`, `block`, `blockIdx`, `threadIdx`.
- **`Idx` = which position (offset)** | **`Dim` = how many (count)**.
- Array shape is stated as **rows × cols**.
- Exercises either state the array shape **or** say "assume perfect tiling."

> **Device reference**: [gpu_check.cu](../src/cuda/gpu_check.cu) prints every
> hardware limit on this machine (registers/SM, clocks, bandwidth, budgets).

---

## 1. STEP 1 — You are given an array

Everything starts with real data. Example: **array rows 128 × cols 256**.

```
rows = 128      (how many rows / height, the y direction)
cols = 256      (how many columns / width, the x direction)
```

Before anything else: **know your data's shape**, because the block will be chosen to fit it.

---

## 2. STEP 2 — Choose the block `(x, y)`

A block is a **logical grouping of threads**, `blockDim.x` wide × `blockDim.y` tall.
`block(16, 4)` = 16 columns × 4 rows = **64 threads**.

**Block area = blockDim.x × blockDim.y** — and ONLY that. Never fabricate the area
from the block count, array width, or anything else.

Choose the block with the **R1–R4 rules, in order:**

### R3 — Legal size (check FIRST)
**Is this block allowed at all?**

```
R3 legal ⟺ block area ≤ maxThreadsPerBlock (1024)
```

- Everything ≤ 1024 is legal. Above it: illegal.
- R3 comes FIRST. It's a bool (yes/no). If it fails, no other rule matters.

### R1 — Warp-clean (thread count is a whole number of warps)
**Does the block divide evenly into 32-thread warps?**

```
warp-clean ⟺ block area ÷ 32 = integer
```

- A warp is **EXACTLY 32 threads** — fixed, not "max 32". Every warp = full 32 lanes.
- 64 → 2 ✓ | 128 → 4 ✓ | 256 → 8 ✓
- 16 → 0.5 ✗ | 40 → 1.25 ✗ | 1000 → 31.25 ✗
- **Anything below 32 is automatically R1-fail.**

### R2 — Occupancy (whole blocks per SM)
**How many of these blocks fit per SM, evenly?**

```
R2 clean ⟺ 1024 ÷ block area = integer
```

- 1024 ÷ 128 = 8 ✓ | 1024 ÷ 320 = 3.2 ✗ (idle thread-slots in some SM)
- Direction: `capacity ÷ block`, NEVER `block ÷ capacity`.

### R4 — Aspect (block matches data shape)
**Does the block's width-to-height ratio track the data?**

```
R4 pass ⟺ blockDim.x ≥ blockDim.y    (for wide data)
```

- Wide data (width ≥ height) → wide block, x ≥ y.
- Square data → (16,16), (8,8) acceptable.
- Skip/weigh for ~square arrays.

### Worked block choice — array 128 rows × 256 cols (wide)

| Block | Area | R3 (≤1024) | R1 (÷32) | R2 (1024÷area) | R4 (x≥y) | Verdict |
|---|---|---|---|---|---|---|
| (16, 8) | 128 | ✓ | 4 ✓ | 8 ✓ | ✓ | valid |
| (4, 4) | 16 | ✓ | **0.5 ✗** | 64 ✓ | ✓ | **R1-fail** |
| (32, 16) | 512 | ✓ | 16 ✓ | 2 ✓ | ✓ | strong wide choice |
| (8, 16) | 128 | ✓ | 4 ✓ | 8 ✓ | **✗ tall** | R4-fail |

**Shortlist**: (32,16) strong, (16,8) valid. Avoid degenerate strips like (64,1).

**Memory hooks**
- "R2 = occupies SM, R3 = legal size ceiling."
- Sweet spot: blocks of 128–256 threads on modern GPUs.

---

## 3. STEP 3 — Compute the grid (ceil, always)

The grid is the number of blocks you launch per axis. It **must cover the array**,
so round UP:

```
gridDim.x = ceil(cols ÷ blockDim.x)
gridDim.y = ceil(rows ÷ blockDim.y)
```

Worked: array 128×256, block (32,16):

```
gridDim.x = ceil(256 ÷ 32) = 8
gridDim.y = ceil(128 ÷ 16) = 8
grid = (8, 8)
```

- **ALWAYS ceil, never round.** Rounding down would leave real data uncovered.

### Block count & threads & phantoms

```
block count   = gridDim.x × gridDim.y
total threads = block count × block area
array cells   = rows × cols
phantoms      = total threads − array cells   (must be ≥ 0)
```

Worked: grid (8,8), block (32,16):

```
blocks      = 8 × 8 = 64
threads     = 64 × 512 = 32,768
cells       = 128 × 256 = 32,768
phantoms    = 0
```

### Phantom rule (the ONLY source of phantoms)
- **Phantoms exist ONLY when the grid overshoots the array** — i.e. when an axis
  had to round UP (an axis where `cols` or `rows` is not divisible by the block).
- Perfect division on both axes ⟹ zero phantoms.
- Ceiling present ⟹ phantoms. **"Ceiling ⟹ phantoms; perfect fit ⟹ zero."**
- Phantom = `grid coverage − array` where coverage = `gridDim.x·blockDim.x × gridDim.y·blockDim.y`.

---

## 4. STEP 4 — Index formulas (NOW the flat index makes sense)

Only after array + block + grid exist do you compute where each thread lands.
The thread lives in this space:

```
A thread's coordinate:
  blockIdx.(x,y)  → which block (its offset in the grid)
  threadIdx.(x,y) → which thread inside that block
```

There are exactly three companion formulas — the index pipeline. Each thread
computes its own; its values live in its own registers:

### a) Row & column (global thread position)

```
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
```

- `row` ranges 0 .. rows−1 across the whole grid.
- `col` ranges 0 .. cols−1 across the whole grid.

### b) Linear (flat) index — the memory address

```
flat = row * cols + col      # USES data width `cols`, NEVER blockDim.x
```

- `cols` is the ARRAY width (number of columns). Using `blockDim.x` here is a bug.
- This converts the 2D coordinate into the 1D position inside the array's storage —
  exactly the address the load/store will use.

### c) Guard for phantoms

```
if (row < rows && col < cols) {
    out[flat] = ...            # phantom threads skip the work
}
```

- Blocked silences the overshoot: phantom threads were launched by the ceil but
  have no real cell, so the guard must reject them.

---

## 5. STEP 5 — Register safety & occupancy (the three budgets)

A block is **resident on an SM** only if the SM has enough of EVERY resource.
Count blocks/SM under each budget, then **take the smallest**:

```
blocksResident = min of:

  Thread budget:   1024 threads/SM ÷ threads-per-block
  Register budget: 65536 registers/SM ÷ (regs-per-thread × threads-per-block)
  Shared budget:   49152 B/SM ÷ shared-per-block
  Block budget:    maxBlocksPerMultiProcessor (16 on GTX 1650)
```

> The 4th budget came from `gpu_check.cu`'s output: `max blocks/SM: 16`.
> It only binds for very small blocks (e.g. < 64 threads).

**Conventions:**
- Every division rounds **DOWN** (fit INTO a budget, never exceed it).
- `registers/block = threads/block × registers-per-thread`.
- Register count is **kernel-specific** — get it from:
  ```
  nvcc -O2 -arch=sm_75 --ptxas-options=-v <file>.cu -o <out>   → "Used N registers"
  ```

**Per-SM hardware (GTX 1650, queried):**

| Resource | Capacity |
|---|---|
| SMs | 16 |
| Threads/SM (max) | 1,024 |
| Threads/block (max) | 1,024 |
| Registers/SM | 65,536 (64K) |
| Shared mem/block | 49,152 B (48 KB) |
| Warp size | 32 |

**Register arithmetic:**
- 1 register = **32 bits = 4 bytes = 1 float**.
- Full occupancy: 65,536 ÷ 1,024 threads = **64 registers/thread = 256 B = 64 floats**.
- `int`/`float`/`bool` → 1 register. `double`/`long long` → 2.
- **Liveness**: two values alive at the same time need separate registers; dead
  registers get reused. The compiler does this; you read `--ptxas-options=-v`.

**Worked example** — `vectorDotKernel` (24 regs, 1,024 B shared, 256-thread blocks):

| Budget | Formula | Blocks |
|---|---|---|
| Thread | 1,024 ÷ 256 | **4** ← limiter |
| Register | 65,536 ÷ (24 × 256) | 10 |
| Shared | 49,152 ÷ 1,024 | 48 |
| **Resident** | min(4, 10, 48) | **4** ⟹ 32 warps/SM = full occupancy |

**Why it matters:** occupancy = warps resident per SM. More warps = more latency
hiding (other warps run while one waits on VRAM). Kernels using > 64 regs/thread
cannot host 1,024 threads → occupancy drops → slower. Occupancy is about WARPS,
not block count.

---

## 6. Memory Physics (the flow under the index)

### 6.1 SIMT execution
- A warp executes **ONE instruction at a time on 32 lanes**, each lane with private
  registers/data. The SM issues whole-warp instructions, not per-thread ops.
- Divergent branches run sequentially (half a warp idle while other half runs).

### 6.2 Memory hierarchy (GTX 1650)
| Layer | Location | Size | Speed |
|---|---|---|---|
| Registers | on-chip, inside each SM | 256 KB/SM | ~1 cycle |
| L1 + shared | on-chip, inside each SM | ~64 KB/SM | ~20–30 cycles |
| L2 | on-chip, one for all SMs | ~1 MB | ~200 cycles |
| VRAM (global) | off-chip, on card | 4 GB | ~500+ cycles |
| System RAM | motherboard (CPU side) | 16–32 GB | + PCIe hop |

### 6.3 Coalescing law
- Memory fits in **32-byte sectors** grouped into **128-byte cache lines**.
- 32 consecutive floats = 128 B = 4 sectors = 1 line = **one shot, 100% efficiency**.
- **Whatever you touch in a sector, you pay for the whole sector.**
- Efficiency = `bytes used ÷ bytes fetched`. Strided lanes scatter across sectors → up to ~32× waste.
- `row = blockIdx.y*blockDim.y + threadIdx.y`, `col = blockIdx.x*blockDim.x + threadIdx.x`
  with consecutive `threadIdx.x` → consecutive `col` → **the pattern the hardware loves**.

### 6.4 Data flow timing
- Loads happen at **execution time** (on-demand), not at launch.
- The CPU "push" (`cudaMemcpy`) fills VRAM; the GPU "pull" (`x[index]`) fetches L2→L1→register.
- Managed memory (`cudaMallocManaged`) migrates pages between RAM and VRAM on demand.

### 6.5 cudaMalloc vs cudaMallocManaged
| | `cudaMalloc` | `cudaMallocManaged` |
|---|---|---|
| Home at rest | VRAM, fixed | wherever last touched |
| CPU reads it? | must `cudaMemcpy` first | allowed directly |
| Who moves data | you, explicitly | driver, automatically |
| Cost | predictable | hidden migration stalls |

---

## 7. Toolchain Rules

- **Compute Sanitizer** (successor to `cuda-memcheck`):
  ```
  compute-sanitizer --tool memcheck ./<binary>
  ```
- Multi-line macros: EVERY line except the last needs a trailing `\`.
- Register report: `--ptxas-options=-v`.

---

## 8. Full Workflow Checklist

Given array `rows × cols`:

1. **Know the data** — rows, cols, wide or square.
2. **Pick block** `(x,y)` — R3 legal → R1 warp-clean → R2 integer occupancy → R4 aspect.
3. **grid** — `ceil(cols/blockDim.x)`, `ceil(rows/blockDim.y)`.
4. **Count** — blocks, total threads, phantoms (only from ceil overshoot).
5. **Index** — `row`, `col`, `flat = row*cols + col`, guard `if (row<rows && col<cols)`.
6. **Occupancy** — `--ptxas-options=-v` → three budgets → resident blocks & warps.
7. **Coalescing** — confirm consecutive threadIdx → consecutive memory addresses.