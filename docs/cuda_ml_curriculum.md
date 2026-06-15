# Champion-Level Curriculum: C++ to CUDA C++ for Machine Learning & GPU Acceleration

**Author**: Rahul Kumar (rahul238xaviers@gmail.com)  
**Goal**: Master CUDA programming for champion-level performance in ML workloads (LLM SFT, inference, agentic RAG, custom kernels, FinTech simulations).  
**Philosophy**: Deeply **exercise-driven**. For every major concept/function, complete **50–100+ varied problems** (mathematics, linear algebra, tensors, ML primitives, simulations). Build workbook-style `.cpp` files with verification, timing, and colored PASS/FAIL reporting — modeled after the provided `std::transform` coordinate mapping workbook.

**Tracking**: Maintain one `.cpp` workbook per major topic. Use Git for version control. Time commitment: 8–20 weeks (2–4 hours/day + projects).

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
   sudo apt install cuda-toolkit-12-8 -y   # Check latest version
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

### 2. Install VS Code + Antigravity (from WSL Ubuntu)
```bash
# Install VS Code
sudo apt install code -y   # or use Microsoft repository for latest

# Install Antigravity (lightweight, fast VS Code alternative with excellent C++/CUDA support)
sudo apt install antigravity -y   # If available, or build from source / use flatpak
```

**Alternative**: Install VS Code on Windows and use Remote - WSL extension (preferred).

### 3. IDE Extensions Setup

#### VS Code (Highly Recommended)
- **Remote - WSL** (connect to Ubuntu)
- **C/C++** (by Microsoft) — IntelliSense, debugging
- **CUDA** (search "CUDA" by NVIDIA or "C++ CUDA Support")
- **CMake Tools**
- **Python** (for mixed projects)
- **GitLens**, **Markdown All in One**
- Settings: Enable C++ IntelliSense for CUDA files (`.cu`)

#### Antigravity / Other Editors
- Install C++ language server (clangd)
- CUDA syntax highlighting plugins
- Configure build tasks for `nvcc`

**Test**: Create `vector_add.cu`, compile with `nvcc`, and debug in VS Code.

### 4. macOS / Apple Silicon Setup (For C++ Workbooks)
**Important:** On Apple Silicon (M1 through M5 chips), the behavior of parallel C++ code depends entirely on which compiler you use.

#### ✅ RECOMMENDED: Clang++ with Grand Central Dispatch (GCD)
Apple's Clang implements `std::execution::par` using GCD - Apple's native parallel framework optimized for Apple Silicon.  
**No extra libraries needed. Compile with:**
```bash
clang++ -std=c++20 -O3 -D_LIBCPP_ENABLE_EXPERIMENTAL program.cpp -o program
./program
```

---

## Module 1: C++ Fundamentals Refresh (3–5 days)
Focus on low-level mastery for CUDA compatibility.

### 1.1 Variables, Types, Pointers & Memory
**Core Functions/Concepts**:
- `sizeof`, primitive types, pointer arithmetic (`*`, `+`, `&`), `new`/`delete`, `malloc`/`free`
- References, `nullptr`

**Exercises** (Aim for 80–100 problems):
- 20 pointer arithmetic on 1D arrays (math: vector addition, dot products)
- 20 on 2D/3D flattening (matrix ops, tensor indexing)
- 20 memory allocation patterns (heap vs stack for large ML batches)
- 20 edge cases: null deref, out-of-bounds, alignment
- ML: Simulate embedding lookups, attention score storage

**Workbook Style**: Create `pointer_arithmetic_workbook.cpp` with verification loop like your example.

### 1.2 Arrays, Strings & Flattening
**Core**:
- C-arrays, `std::array`, `std::vector`
- Row/column-major, striding formulas

**Exercises** (100+):
- 30 matrix transpose/flatten (various shapes: 32x32, 784x512 for MNIST-like)
- 30 3D/4D tensor indexing (batch x channel x height x width)
- 20 strided access patterns (convolutions, sliding windows)
- 20 math: matrix multiplication manual implementation
- ML: Image padding, voxel data in medical imaging simulations

**Practice**: Extend your existing notes with code variants.

---

## Module 2: Modern C++ for GPU Readiness (5–7 days)

### 2.1 Lambdas & Captures
**Core**: Lambda syntax, captures `[=]`, `[&]`, `[ptr]`, mutable

**Exercises** (80–100):
- 25 simple transforms (math functions: sin, exp on vectors)
- 25 coordinate mapping (build on your std::transform workbook)
- 20 capture pointer vs value for mutable tensors
- 20 ML: Element-wise activation (ReLU, GELU, softmax components)
- Variety: Financial time-series transformations, risk factor computations

**Workbook**: Expand your provided `16.1_level_stdpar_coord_mapping_workbook.cpp` to 50+ problems.

### 2.2 STL Algorithms with `<execution>`
**Core**: `std::transform`, `std::for_each`, `std::reduce`, `std::sort`, `std::copy`, `std::fill`
**Policies**: `par`, `par_unseq`, `unseq`

**Exercises per Algorithm** (Aim 80–100 each):
- **std::transform** (as in your example): 100 problems covering all your coordinate scenarios + extensions (rotations, normalizations, fused ops like add + scale)
- **std::reduce**: Summations, max/min, custom ops for loss calculations, attention weights
- **std::for_each**: In-place mutations for gradient updates, quantization
- Math/ML Variety: Linear algebra (BLAS level-1), statistics, Monte Carlo sampling for FinTech risk

**Workbook**: One file per algorithm or mega-workbook with sections.

### 2.3 Templates & Smart Pointers
**Exercises**: Generic tensor class, RAII for GPU resources (preview).

---

## Module 3: CUDA Basics (5–8 days)

### 3.1 Setup, Kernels & Launch
**Core**:
- `nvcc`, `__global__`, `<<<>>>`, `threadIdx`, `blockIdx`, proper global indexing

**Exercises** (80+):
- 30 vector ops (add, scale, saxpy)
- 20 matrix element-wise
- 20 simple reductions
- ML: Forward pass primitives for embeddings, layer norms

**Workbook**: `basic_cuda_kernels_workbook.cpp` with host/device verification.

### 3.2 Memory Transfers
**Core**: `cudaMalloc`, `cudaMemcpy`, `cudaFree`, error handling

**Exercises** (70+):
- Pinned memory for faster transfers in training loops
- Batch data movement patterns

---

## Module 4: CUDA Memory Hierarchy (7–10 days) — High Priority for ML Perf

### 4.1 Global & Unified Memory
**Core**: `cudaMallocManaged`, prefetch

**Exercises** (100):
- 40 tensor allocation patterns (various shapes for LLMs)
- 30 unified memory page fault analysis (profile with Nsight)
- 30 vs explicit copy benchmarks (training throughput)

### 4.2 Shared Memory & Tiling
**Core**: `__shared__`, `__syncthreads()`

**Exercises** (100+):
- Tiled matrix multiplication (various tile sizes)
- Convolution tiling
- Attention score computation with shared caches
- ML: Optimized GEMM for transformer FFN layers

**Workbook**: Multiple variants with performance timing.

### 4.3 Advanced: Constant, Texture, Coalescing
**Exercises**: Cache optimization for lookup tables in embeddings, image processing.

---

## Module 5: Optimization & Parallel Primitives (10–14 days)

### 5.1 Reductions, Scans, Atomics
**Core**: Warp primitives, cooperative groups, `atomicAdd`

**Exercises** (100 per major primitive):
- Softmax, layer norm, attention reductions
- Histogram/binning for quantization
- FinTech: Portfolio risk aggregations

### 5.2 Mixed Precision & Tensor Cores
**Exercises**: FP16/BF16 matmul, fused kernels

---

## Module 6: CUDA Libraries for ML (7–10 days)

### 6.1 cuBLAS / CUTLASS
**Exercises** (80+):
- 50 batched GEMM for multi-head attention
- Custom GEMM kernels with CUTLASS templates

### 6.2 cuDNN
**Exercises**: Build conv nets, RNNs, transformer blocks from primitives.

### 6.3 Thrust, NCCL
**Exercises**: High-level GPU STL for prototyping, multi-GPU all-reduce for distributed SFT.

---

## Module 7: Advanced Systems & Profiling (10–14 days)

**Topics**:
- Streams, Graphs, Multi-GPU
- Nsight profiling (bottleneck identification)
- Custom PyTorch CUDA extensions

**Exercises**: Full mini-transformer training loop in CUDA, compare to PyTorch, optimize KV cache for inference, agentic RAG vector search acceleration.

**Capstone**:
- 1. CUDA tensor library with 100+ ops
- 2. LLM SFT kernel optimizations
- 3. FinTech Monte Carlo on GPU

---

## Daily/Weekly Practice Rules
- **Minimum 10–20 new problems/day** per active topic.
- Always include verification + timing.
- Profile every kernel (Nsight).
- Vary dimensions: powers of 2, ML common shapes (4096, 8192, etc.).
- Math variety: Linear algebra, calculus (derivatives for backprop), probability (sampling).
- ML usage: Embeddings, attention, FFN, normalization, loss, optimizers.

**Resources**:
- NVIDIA DLI courses (do all labs + extend with 50 extra exercises each)
- CUDA Programming Guide + Best Practices
- GitHub: llm.c, cutlass, tiny-cuda-nn
- Your own workbooks repo

Update this file as you progress. Add new sections for completed workbooks.