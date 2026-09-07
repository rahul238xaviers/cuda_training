# Champion-Level Curriculum: C++ to CUDA C++ for Machine Learning & GPU Acceleration

**Author**: Rahul Kumar (rahul238xaviers@gmail.com)  
**Goal**: Master CUDA kernel programming to build high-performance LLM training and inference engines from scratch on NVIDIA GPUs.  
**Philosophy**: Deeply **exercise-driven**. For every concept and kernel primitive, progress from C++ host implementations to custom low-level CUDA kernels. Learn the exact minute execution model (threads, registers, warps, blocks, shared memory, grids) required to write the 24 LLM training kernels powering custom C++ Transformer engines.

**Tracking**: Maintain one `.cu` / `.cpp` workbook per major topic. Use Git for version control. Time commitment: 8–20 weeks (2–4 hours/day + projects).

---

## Environment Setup (Day 0–1)

### 1. WSL 2 + Ubuntu (Recommended for Development)
1. On Windows (PowerShell as Admin):
   ```powershell
   wsl --install
   wsl --install -d Ubuntu
   ```

2. Install NVIDIA Drivers **on Windows** (latest from NVIDIA website).

3. Inside WSL Ubuntu:
   ```bash
   sudo apt update && sudo apt upgrade -y
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt update
   sudo apt install build-essential cuda-toolkit-12-8 -y   # Install GCC/G++ and CUDA Toolkit
   ```

4. Add to `~/.bashrc`:
   ```bash
   export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   source ~/.bashrc
   ```

5. Verify:
   ```bash
   nvcc --version
   nvidia-smi
   ```

### 2. VS Code / Development Setup
```bash
# Verify nvcc compiler
nvcc --version
```

#### Compile CUDA Source (`.cu`) with `nvcc`:
```bash
mkdir -p output
nvcc -O3 -std=c++17 -arch=sm_75 --extended-lambda src/module2/2.1_lambdas_function_objects/exercise/beginner_workbook.cu -o output/wb && ./output/wb
```

#### Compile C++ Source (`.cpp`) with `g++`:
```bash
mkdir -p output
g++ -std=c++20 -O3 src/module1/1.1_basic_offsets/exercise/beginner_workbook.cpp -o output/wb && ./output/wb
```

---

## GPU Architecture Translation Guide

For engineers transitioning from graphics shaders or other GPU shading languages to NVIDIA CUDA C++:

| Concept / Primitives | Abstract Compute Paradigm | NVIDIA CUDA C++ |
| :--- | :--- | :--- |
| **Global Thread Index** | Global compute worker index | `int gid = blockIdx.x * blockDim.x + threadIdx.x;` |
| **Thread Index in Block** | Workgroup local thread index | `threadIdx.x` |
| **Block Index in Grid** | Workgroup index | `blockIdx.x` |
| **Threads per Block** | Workgroup dimensions | `blockDim.x` |
| **SIMD / Warp Index** | Wavefront / Subgroup ID | `int warp_id = threadIdx.x / 32;` |
| **Lane Index in Warp** | Lane ID in Wavefront | `int lane_id = threadIdx.x % 32;` |
| **Warp Shuffle Down** | Subgroup shuffle down | `__shfl_down_sync(0xffffffff, val, delta)` |
| **Warp Shuffle XOR** | Subgroup butterfly exchange | `__shfl_xor_sync(0xffffffff, val, mask)` |
| **Shared Memory** | Local / Threadgroup memory | `__shared__ float scratch[256];` |
| **Block Barrier** | Local barrier synchronization | `__syncthreads();` |
| **Data Types** | Half / Bfloat formats | `__nv_bfloat16`, `half`, `float` |
| **Vectorized Types** | 128-bit memory packed load | `float4`, `__nv_bfloat162` |
| **Read-Only Cache** | Constant cache route | `const __restrict__` / `__ldg()` |

---

## Module 1: C++ Fundamentals Refresh (3–5 days)
Focus on low-level memory layout, pointer arithmetic, and striding for GPU readiness.

### 📂 C++ Fundamentals Subtopics (100 Problems)

Every subtopic is structured inside its own directory containing `theory.md`, `exercise/`, and `solution/`.

#### 1.1 Variables, Types, Pointers & Memory
* **Section 1: Pointer Arithmetic & Offsets**
  - [[1.1] Basic Offset & Increment](../src/module1/1.1_basic_offsets/theory.md)
  - [[1.2] Strides & Indirection](../src/module1/1.2_strides_indirection/theory.md)
  - [[1.3] Array Reductions & Swaps](../src/module1/1.3_reductions_swaps/theory.md)
  - [[1.4] Multi-array & Strided Ops](../src/module1/1.4_multi_array_strided/theory.md)

* **Section 3: Heap vs Stack Memory Management**
  - [[1.9] Single & Array Allocations](../src/module1/1.9_single_array_alloc/theory.md)
  - [[1.10] Multi-Dim & Alignment](../src/module1/1.10_multidim_alignment/theory.md)
  - [[1.11] Arenas & Placements](../src/module1/1.11_arenas_placements/theory.md)
  - [[1.12] Lifetimes, Ownership & Resize](../src/module1/1.12_lifetimes_ownership/theory.md)

#### 1.2 Arrays, Strings & Tensor Flattening
* **Section 2: 2D & 3D Array/Tensor Flattening & Strides**
  - [[1.5] Row/Col-Major Indexing](../src/module1/1.5_row_col_indexing/theory.md)
  - [[1.6] 3D/4D Permute & Flatten](../src/module1/1.6_permute_flatten/theory.md)
  - [[1.7] Subgrids, Padding & Diagonals](../src/module1/1.7_subgrids_padding/theory.md)
  - [[1.8] Pack, Wrap, Stencil & Crops](../src/module1/1.8_pack_wrap_stencil/theory.md)

* **Section 5: Machine Learning GPU Prep & Simulations**
  - [[1.17] Embeddings & Projections](../src/module1/1.17_embeddings_projections/theory.md)
  - [[1.18] Attention & Quantization](../src/module1/1.18_attention_quantization/theory.md)
  - [[1.19] ML Layer Offsets](../src/module1/1.19_ml_layer_offsets/theory.md)
  - [[1.20] Pooling, Masks & Cycles](../src/module1/1.20_pooling_masks_cycles/theory.md)

---

## Module 2: Modern C++ for GPU Readiness (5–7 days)
Master modern C++17/C++20 idioms necessary to architect production-grade GPU runtime engines, device memory managers, and generic tensor abstraction layers.

### 📂 Modern C++ Subtopics

Every subtopic is structured inside its own directory containing `theory.md`, `exercise/` (Beginner, Intermediate, Champion), and `solution/`.

#### 2.1 Lambdas, Closures & Function Objects
- [[2.1] Theory Guide](../src/module2/2.1_lambdas_function_objects/theory.md)
* **Minute Subtopics to Master**:
  1. **Lambda Syntax & Capture Modes**: Value `[=]`, reference `[&]`, selective captures `[x, &y]`, move captures `[ptr = std::move(p)]`, `mutable` keyword.
  2. **Stateless vs Stateful Lambdas**: Conversion of stateless lambdas to function pointers; storing stateful closures in `std::function` vs inlined template invocations.
  3. **CUDA Extended Lambdas**: Compiling device lambdas with `nvcc --extended-lambda`, decorating closures with `__device__` and `__host__ __device__`.
  4. **Generic Higher-Order Kernel Callbacks**: Passing lambda predicates and mathematical transforms directly into templated elementwise GPU kernels.
* **ML Application**: Custom elementwise activations (ReLU, GeLU, SiLU, Swish, SwiGLU) composable at compile-time without kernel launch overhead.

#### 2.2 Templates, SFINAE & Precision Type Traits
- [[2.2] Theory Guide](../src/module2/2.2_templates_type_traits/theory.md)
* **Minute Subtopics to Master**:
  1. **Function & Class Templates**: Parameter packs, variadic templates (`typename... Args`), explicit and partial specialization.
  2. **Compile-Time Dispatch**: SFINAE (`std::enable_if_t`), compile-time branching with `if constexpr`, and type categorization with `<type_traits>`.
  3. **ML Precision Traits**: Custom compile-time trait classes mapping storage types (`__nv_bfloat16`, `half`, `float`) to math accumulation types (`AccumulatorType<T>::type`).
  4. **Numerical Bounds & Epsilon Traits**: Precision-safe epsilon and machine min/max constants for numerical stability in normalization and loss functions.
* **ML Application**: Writing generic GPU kernels supporting FP32, FP16, and BF16 with zero runtime overhead.

#### 2.3 Smart Pointers, RAII & Resource Wrappers
- [[2.3] Theory Guide](../src/module2/2.3_raii_smart_pointers_streams/theory.md)
* **Minute Subtopics to Master**:
  1. **RAII Resource Management**: Eliminating manual `cudaFree`, `cudaStreamDestroy`, and `cudaEventDestroy` memory leaks.
  2. **Custom Smart Pointer Deleters**: Pairing `std::unique_ptr` with custom device deleters for `cudaMalloc`, `cudaMallocHost`, and pinned memory.
  3. **StreamGuard & Scoped Synchronization**: RAII wrappers that bind an execution stream to a thread scope and automatically synchronize or reset upon exit.
  4. **EventTimer Utility**: RAII-based CUDA event pair wrapper for microsecond-accurate kernel execution timing and benchmark logging.
* **ML Application**: Leak-free device memory managers, multi-stream pipelines, and profiler harnesses.

#### 2.4 Move Semantics, Perfect Forwarding & Tensor Views
- [[2.4] Theory Guide](../src/module2/2.4_move_semantics_views/theory.md)
* **Minute Subtopics to Master**:
  1. **Rvalue References & Move Semantics**: `std::move`, move constructors, and move assignment operators for heavy GPU resource holders (Rule of 5).
  2. **Perfect Forwarding**: Universal references (`T&&`) and `std::forward` in kernel launch wrappers to preserve value category.
  3. **Non-Owning Tensor Views**: Designing `TensorView<T>` (raw pointer, shape tuple, stride tuple) that decouples tensor metadata from memory ownership.
  4. **Zero-Copy Sub-Tensor Slicing**: Computing slice offsets and strided views without copying GPU DRAM buffers.
* **ML Application**: Efficient tensor passing through multi-layer neural network graphs without DRAM allocation churn.

#### 2.5 Host Parallel Execution & Asynchronous Pipelines
- [[2.5] Theory Guide](../src/module2/2.5_stl_parallel_execution/theory.md)
* **Minute Subtopics to Master**:
  1. **Host Parallel Work-Sharing**: Chunked range partitioning (`parallel_for`) and parallel tree reductions (`parallel_sum`).
  2. **Multi-Threaded CPU Baselines**: Implementing parallel CPU tensor operations as analytical ground-truth baselines to test GPU kernels against.
  3. **Concurrent Host Pipelines**: Using `std::async` and worker thread pools to asynchronously launch concurrent operations into independent CUDA streams.
  4. **Host-Device Overlap**: Concurrently executing CPU preprocessing / data loading on pinned buffers while the GPU executes transformer attention layers.
* **ML Application**: High-throughput data loading, multi-stream model parallelism, and rigorous CPU test harness baselines.

---

## Module 3: CUDA Architecture & Execution Model (Deep Minute Breakdown)

To program GPU kernels effectively, an engineer must understand the exact hardware mapping from high-level code down to individual SM registers.

### 3.1 Thread Level & Register Allocation
- [[3.1] Theory Guide](../src/module3/3.1_threads_registers/theory.md)
* **Minute Subtopics to Master**:
  1. **Thread Registers**: Fast on-chip 32-bit registers allocated per thread. Max 255 registers per thread on Modern Ampere/Hopper architecture.
  2. **Register Pressure & Spilling**: When register usage exceeds capacity, compiler spills variables to high-latency Local Memory (HBM). Inspecting compilation output with `nvcc --ptxas-options=-v`.
  3. **Thread State & Control Flow**: Independent program counter per thread, thread-local stack, and scalar data ownership.
  4. **Vectorized Thread Access**: Using 128-bit SIMD loads per thread (`float4`, `bfloat2`, `ulonglong2`) to maximize global memory bandwidth.

### 3.2 Warp Level (32 Threads) & SIMT Execution
- [[3.2] Theory Guide](../src/module3/3.2_warps_simt_shuffles/theory.md)
* **Minute Subtopics to Master**:
  1. **Warp Scheduling**: Group of 32 threads executing instructions in lockstep (SIMT — Single Instruction, Multiple Threads).
  2. **Warp Divergence**: Conditional branching (`if/else`) where threads in a warp take different execution paths, causing serialized execution.
  3. **Active Thread Mask**: 32-bit bitmask (`0xffffffff`) representing active, un-diverged threads in a warp.
  4. **Intra-Warp Shuffle Primitives**: Hardware-level register-to-register data exchange between warp lanes without shared memory:
     - `__shfl_sync(mask, val, srcLane)`: Broadcast from specific lane.
     - `__shfl_down_sync(mask, val, delta)`: Shift values down by delta lanes.
     - `__shfl_xor_sync(mask, val, laneMask)`: Butterfly exchange for reductions.
  5. **Warp-level Reductions**: Summing/max-finding across 32 threads in 5 clock cycles ($O(\log_2 32)$) using shuffles.

### 3.3 Block Level (Threadgroups) & Shared Memory Banking
- [[3.3] Theory Guide](../src/module3/3.3_blocks_shared_banking/theory.md)
* **Minute Subtopics to Master**:
  1. **Block Architecture**: Cooperative group of threads (up to 1024 threads per block) executing on a single SM (Streaming Multiprocessor).
  2. **Block Indexing & Shapes**: `threadIdx.x/y/z`, `blockDim.x/y/z`, calculating 1D/2D/3D flat linear thread offsets inside a block.
  3. **Shared Memory (`__shared__`)**: High-speed, low-latency on-chip scratchpad memory shared across all threads in a block.
  4. **Shared Memory Banking & Conflicts**: Shared memory is divided into 32 banks (4 bytes wide). If 2 threads in a warp access different addresses within the same bank, a **Bank Conflict** occurs, serializing access.
  5. **Dynamic vs Static Shared Memory**: Allocating `extern __shared__ float s_mem[]` at kernel launch time (`<<<grid, block, shared_bytes>>>`).
  6. **Block Synchronization (`__syncthreads()`)**: Execution barrier enforcing all threads in the block to reach the barrier before any proceed.

### 3.4 Grid Level & SM (Streaming Multiprocessor) Scheduling
- [[3.4] Theory Guide](../src/module3/3.4_grids_occupancy_strides/theory.md)
* **Minute Subtopics to Master**:
  1. **Grid Layout**: Collection of thread blocks (`gridDim.x/y/z`, `blockIdx.x/y/z`).
  2. **SM Occupancy**: Ratio of active warps per SM to maximum supported warps (limited by register count, shared memory size, and block size).
  3. **Grid-Stride Loops**: Writing scalable kernels that handle arbitrary input sizes $N$ regardless of launched grid size:
     ```cpp
     for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
         // Process element i
     }
     ```
  4. **Cooperative Groups & Grid Sync**: `cooperative_groups::this_grid().sync()` for grid-wide synchronization (requires `cudaLaunchCooperativeKernel`).

---

## Module 4: CUDA Memory Hierarchy & Tiling Techniques

### 4.1 Global Memory, Read-Only Cache & Coalescing
- [[4.1] Theory Guide](../src/module4/4.1_coalescing_readonly_cache/theory.md)
* **Memory Coalescing Rules**: Global memory reads/writes are serviced in 32-byte, 64-byte, or 128-byte transactions. Threads in a warp must access contiguous addresses to achieve 100% memory bandwidth utilization.
* **Read-Only Data Cache**: Using `const __restrict__` keyword or `__ldg(ptr)` intrinsic to route immutable data (e.g. weights, token IDs) through the texture/read-only L1 cache.

### 4.2 Shared Memory Matrix Tiling
- [[4.2] Theory Guide](../src/module4/4.2_shared_memory_tiling/theory.md)
* **Tile Loading Pattern**: Loading sub-blocks of $A$ ($BM \times BK$) and $B$ ($BK \times BN$) into `__shared__` memory cooperatively.
* **Register Accumulation**: Storing intermediate dot-product accumulators ($TM \times TN$) in registers per thread before writing back to global memory.

---

## Module 5: Parallel Primitives & Numerical Stability for LLMs

### 5.1 Reductions & Prefix Scans
- [[5.1] Theory Guide](../src/module5/5.1_warp_block_reductions/theory.md)
* **Warp Shuffle + Shared Memory Two-Pass Reduction**:
  1. Each thread computes local reduction over grid-stride loop.
  2. Perform intra-warp reduction using `__shfl_down_sync`.
  3. Lane 0 of each warp writes warp sum to shared memory.
  4. First warp performs final reduction over shared memory sums.

### 5.2 Global Atomic Operations
- [[5.2] Theory Guide](../src/module5/5.2_atomic_operations/theory.md)
* **Atomic Addition**: `atomicAdd(float* address, float val)` for safe concurrent updates across different thread blocks.

### 5.3 Numerically Stable Reductions (Softmax & Loss)
- [[5.3] Theory Guide](../src/module5/5.3_numerically_stable_reductions/theory.md)
* **Log-Sum-Exp Trick**: Subtracting $\max_i(x_i)$ before exponentiation:
  $$\text{Softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$
  Prevents overflow ($e^{x} \rightarrow \infty$) and underflow in floating-point arithmetic.

---

## Module 6: Precision, Intrinsics & Tensor Cores

### 6.1 BFloat16 (`__nv_bfloat16`) Math
- [[6.1] Theory Guide](../src/module6/6.1_bfloat16_math/theory.md)
* **Format**: 1 sign bit, 8 exponent bits, 7 mantissa bits (same dynamic range as FP32, lower precision).
* **Vectorized BFloat16 Ops**: `__nv_bfloat162` packs two 16-bit values into a 32-bit register for dual-issue SIMD instructions (`__hadd2`, `__hmul2`, `__fma2`).

### 6.2 Tensor Cores & WMMA API
- [[6.2] Theory Guide](../src/module6/6.2_wmma_tensor_cores/theory.md)
* **Warp Matrix Multiply and Accumulate (WMMA)**: Dedicated hardware execution units on SM for $16 \times 16 \times 16$ matrix multiplication in a single warp cycle.
  ```cpp
  #include <mma.h>
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  wmma::load_matrix_sync(a_frag, a_ptr, lda);
  wmma::load_matrix_sync(b_frag, b_ptr, ldb);
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  ```

---

## Module 7: Master Roadmap — The 24 LLM Training Kernels

This roadmap guides an engineer step-by-step through implementing the 24 fundamental CUDA kernels essential for building an end-to-end LLM training and inference pipeline.

```
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 1: Elementwise, Indexing & Reshaping (5 Kernels)   │
 └────────────────────────────┬────────────────────────────┘
                              │
 ┌────────────────────────────▼────────────────────────────┐
 │ Stage 2: Token Embedding Lookup (1 Kernel)               │
 └────────────────────────────┬────────────────────────────┘
                              │
 ┌────────────────────────────▼────────────────────────────┐
 │ Stage 3: Normalization & RoPE Embeddings (6 Kernels)    │
 └────────────────────────────┬────────────────────────────┘
                              │
 ┌────────────────────────────▼────────────────────────────┐
 │ Stage 4: Loss Functions & AdamW Optimizer (3 Kernels)   │
 └────────────────────────────┬────────────────────────────┘
                              │
 ┌────────────────────────────▼────────────────────────────┐
 │ Stage 5: Matrix Multiplications & GEMMs (7 Kernels)     │
 └────────────────────────────┬────────────────────────────┘
                              │
 ┌────────────────────────────▼────────────────────────────┐
 │ Stage 6: Tiled FlashAttention-2 Fwd & Bwd (2 Kernels)   │
 └─────────────────────────────────────────────────────────┘
```

---

### Stage 1: Elementwise Operations, Indexing & Reshaping (5 Kernels)
- [[7.1] Theory Guide](../src/module7/7.1_elementwise_reshape/theory.md)

#### Kernel 1: `reshape_3d.cu`
* **What to Learn**: 3D index decomposition ($idx \rightarrow (i, j, k)$), strided multi-dimensional array mapping.
* **CUDA Concepts**: 1D Grid indexing, modulo & integer division arithmetic (`gid / (D2 * D3)`), memory bounds check.

#### Kernel 2: `reshape_4d.cu`
* **What to Learn**: 4D index decomposition for multi-head attention shapes `[Batch, Seq, Heads, Dim]`.
* **CUDA Concepts**: Strided 4D coordinate extraction, flat memory indexing.

#### Kernel 3: `residual_add.cu`
* **What to Learn**: Residual connections ($x_{out} = x_{in} + \text{residual}$).
* **CUDA Concepts**: Vectorized 128-bit memory loads (`float4` / `bfloat2`), memory coalescing, grid-stride loop pattern.

#### Kernel 4: `swiglu_forward.cu`
* **What to Learn**: Gated Feed-Forward Network activation: $\text{SwiGLU}(x, y) = \text{SiLU}(x) \cdot y = \frac{x}{1 + e^{-x}} \cdot y$.
* **CUDA Concepts**: Fused math instructions, fast math intrinsic `__expf()`.

#### Kernel 5: `swiglu_backward.cu`
* **What to Learn**: Backpropagating gradients through SwiGLU: $\frac{\partial L}{\partial x}$ and $\frac{\partial L}{\partial y}$.
* **CUDA Concepts**: Derivations of sigmoid and SiLU, chain rule in elementwise kernels.

---

### Stage 2: Token Embedding Lookup (1 Kernel)
- [[7.2] Theory Guide](../src/module7/7.2_embedding_lookup/theory.md)

#### Kernel 6: `embedding_forward.cu`
* **What to Learn**: Token embedding gather lookup replacing host-side CPU loops.
* **CUDA Concepts**: 2D coordinate extraction (`token_pos = gid / hidden_dim`, `d = gid % hidden_dim`), indirect memory indexing into weight matrix, read-only cache (`const __restrict__`).

---

### Stage 3: Normalization & Rotary Position Embeddings (6 Kernels)
- [[7.3] Theory Guide](../src/module7/7.3_normalization_rope/theory.md)

#### Kernel 7: `rms_norm_forward.cu`
* **What to Learn**: Root Mean Square Layer Normalization:
  $$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}, \quad y_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i$$
* **CUDA Concepts**: One block per row, grid-stride loop for hidden dim, intra-warp reduction (`__shfl_down_sync`), shared memory scratchpad across warps, block barrier (`__syncthreads()`), reciprocal square root `rsqrtf()`.

#### Kernel 8: `rms_norm_backward.cu`
* **What to Learn**: RMSNorm gradient computation with respect to input $x$ and scale parameter $\gamma$.
* **CUDA Concepts**: Multi-pass reduction over row dimension, accumulating $\sum (dY_i \cdot x_i)$ and $\sum (dY_i \cdot \gamma_i \cdot x_i)$.

#### Kernel 9: `rope_forward.cu`
* **What to Learn**: Rotary Position Embedding (RoPE) applying 2D rotation matrices to pairs of hidden dimensions:
  $$\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$
* **CUDA Concepts**: Strided pair indexing, computing inverse frequency $\theta = b^{-2i/d}$, trigonometric operations.

#### Kernel 10: `rope_backward.cu`
* **What to Learn**: Inverse rotary transformation for backpropagation through RoPE.
* **CUDA Concepts**: Applying transpose rotation matrix ($\theta \rightarrow -\theta$).

#### Kernel 11: `fused_add_norm.cu`
* **What to Learn**: Fusing residual addition ($x \leftarrow x + \text{residual}$) and RMSNorm into a single kernel launch.
* **CUDA Concepts**: Reducing High Bandwidth Memory (HBM) roundtrips, performing in-place residual updates in registers prior to norm reduction.

#### Kernel 12: `fused_backward_add_norm.cu`
* **What to Learn**: Fused backward pass for residual connection and RMSNorm.
* **CUDA Concepts**: Gradient passing to both residual branch and norm input simultaneously.

---

### Stage 4: Loss Functions & AdamW Optimizer (3 Kernels)
- [[7.4] Theory Guide](../src/module7/7.4_loss_and_optimizer/theory.md)

#### Kernel 13: `cross_entropy.cu`
* **What to Learn**: Softmax cross-entropy loss calculation:
  $$L = -\log \left( \frac{e^{z_{target} - \max(z)}}{\sum_j e^{z_j - \max(z)}} \right)$$
* **CUDA Concepts**: Two-stage block reduction (Stage 1: Find max value; Stage 2: Sum exponentials), Log-Sum-Exp numerical stability.

#### Kernel 14: `compute_loss.cu`
* **What to Learn**: Aggregating token losses across batch and sequence dimensions.
* **CUDA Concepts**: Global atomic addition (`atomicAdd`), grid reduction to single scalar.

#### Kernel 15: `adamw_step.cu`
* **What to Learn**: AdamW optimizer parameter update step:
  $$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
  $$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
  $$\theta_t = \theta_{t-1} - \eta \cdot \lambda \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
* **CUDA Concepts**: Vectorized load/store of 4 arrays simultaneously ($\theta, g, m, v$), fused math operations.

---

### Stage 5: Matrix Multiplications & Projection GEMMs (7 Kernels)
- [[7.5] Theory Guide](../src/module7/7.5_gemm_projections/theory.md)

#### Kernel 16: `gemm_proj.cu`
* **What to Learn**: General Matrix Multiplication ($C = A \cdot B$) for linear projections ($Q, K, V$, Output Projections).
* **CUDA Concepts**: 2D Shared memory block tiling ($BM \times BN$), inner loop K-dimension accumulator ($BK$), register micro-tiling ($TM \times TN$).

#### Kernel 17: `gemm_proj_trans_b.cu`
* **What to Learn**: GEMM with transposed $B$ matrix ($C = A \cdot B^T$).
* **CUDA Concepts**: Coalesced memory loading for transposed matrices, avoiding shared memory bank conflicts on transposed writes.

#### Kernel 18: `gemm_bf16.cu`
* **What to Learn**: BFloat16 GEMM execution using Tensor Cores.
* **CUDA Concepts**: WMMA API (`nvcuda::wmma`), loading BFloat16 fragments, accumulator precision conversion.

#### Kernel 19: `gemm_gqa.cu`
* **What to Learn**: Grouped Query Attention (GQA) GEMM broadcasting key/value heads across query head groups.
* **CUDA Concepts**: Strided batch GEMM, repeating KV head indexing ($head_{kv} = head_q / \text{gqa\_group\_size}$).

#### Kernel 20: `gemm_ffn.cu`
* **What to Learn**: Feed-Forward Network GEMM with large intermediate hidden dimensions (e.g. $4 \times d_{model}$ or $8/3 \times d_{model}$).
* **CUDA Concepts**: High-throughput SM grid scheduling, large $N$-dimension block tiling.

#### Kernel 21: `gemm_backward.cu`
* **What to Learn**: Backward pass matrix multiplications ($dX = dO \cdot W^T$ and $dW = X^T \cdot dO$).
* **CUDA Concepts**: Transpose-A and Transpose-B GEMM variants, accumulating gradients into weight buffers.

#### Kernel 22: `fused_swiglu_gemm.cu`
* **What to Learn**: Fusing dual matrix multiplications ($W_{gate} \cdot X$ and $W_{up} \cdot X$) and SwiGLU activation in a single pass.
* **CUDA Concepts**: Dual fragment loading, computing activation directly in registers before global memory store.

---

### Stage 6: Attention Architecture — FlashAttention-2 (2 Kernels)
- [[7.6] Theory Guide](../src/module7/7.6_flash_attention/theory.md)

#### Kernel 23: `flash_attn_fwd.cu`
* **What to Learn**: FlashAttention-2 forward pass (Tri Dao et al.):
  - SRAM Tiling of $Q, K, V$ matrices without materializing the $N \times N$ attention matrix in HBM.
  - Online Softmax tracking running max $m^{(i)}$ and unnormalized sum $l^{(i)}$.
  - Iterative output update:
    $$O^{(j)} = \text{diag}\left(e^{m^{(j-1)} - m^{(j)}}\right) O^{(j-1)} + P^{(j)} V_j$$
* **CUDA Concepts**: SRAM block tiling ($B_r \times d$, $B_c \times d$), causal masking logic (`if (col > row) score = -INF`), online rescale factors in registers, dynamic shared memory allocation.

#### Kernel 24: `fused_attn_bwd.cu`
* **What to Learn**: FlashAttention-2 backward pass:
  - Recomputing attention matrix $S_{ij} = Q_i K_j^T$ on-the-fly in SRAM to avoid storing activation matrices.
  - Pre-summing $D_i = \sum_{k} (dO_{ik} \odot O_{ik})$.
  - Gradient computations: $dP_{ij} = dO_i V_j^T$, $dS_{ij} = P_{ij} \odot (dP_{ij} - D_i)$, $dQ_i += dS_{ij} K_j$, $dK_j += dS_{ij}^T Q_i$, $dV_j += P_{ij}^T dO_i$.
* **CUDA Concepts**: Complex backward SRAM tiling, multi-buffer shared memory, atomic gradient accumulation into $dK$ and $dV$.

---

## Module 8: Systems Integration, Profiling & PyTorch Extensions (10–14 days)

### 8.1 Nsight Systems & Nsight Compute Profiling
* **Nsight Systems (`nsys`)**: Trace CUDA kernel execution timeline, host-to-device memory transfers, stream concurrency.
* **Nsight Compute (`ncu`)**: Analyze kernel SM efficiency, memory bandwidth utilization (SOL — Speed of Light), warp instruction stalls, register pressure, and bank conflicts.

### 8.2 Custom PyTorch CUDA C++ Extensions
* **Binding CUDA Kernels to PyTorch**: Using `torch/extension.h` and pybind11 to expose custom CUDA kernels to PyTorch autograd engine.
  ```cpp
  #include <torch/extension.h>
  torch::Tensor rms_norm_cuda_forward(torch::Tensor input, torch::Tensor weight, float eps);
  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("rms_norm_forward", &rms_norm_cuda_forward, "RMSNorm Forward (CUDA)");
  }
  ```

---

## Daily Practice Protocol

1. **Daily Minimum**: Write & test 1 new kernel or major optimization daily.
2. **Verification Requirement**: Every `.cu` workbook must include a C++ CPU reference verification function comparing GPU results with absolute error tolerance ($< 10^{-4}$ for FP32, $< 10^{-2}$ for BF16).
3. **Profiling Requirement**: Run `ncu` on every custom kernel to measure Memory Bandwidth % and Compute SOL.