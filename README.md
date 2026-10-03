# CUDA C++ for Machine Learning & LLM GPU Kernel Engineering

[![CUDA 12+](https://img.shields.io/badge/CUDA-12.x%20%7C%2011.8+-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![C++17/C++20](https://img.shields.io/badge/C%2B%2B-17%20%2F%2020-00599C?logo=c%2B%2B)](https://en.cppreference.com/)
[![Docker Ready](https://img.shields.io/badge/Docker-Containerized%20Lab-2496ED?logo=docker)](https://www.docker.com/)
[![Interactive Textbook](https://img.shields.io/badge/Next.js-Digital%20Lab%20%26%20Book-black?logo=next.js)](http://localhost:3000)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A self-contained, exercise-driven curriculum and interactive digital lab engineered to take software engineers and ML researchers from C++ low-level memory systems (pointer arithmetic, cache lines, strided layouts) to authoring production-grade, memory-bandwidth-saturated CUDA kernels for Large Language Model (LLM) training and inference.

---

## 🌟 What This Repository Contains

1. **Interactive Digital Textbook & Lab Web App (`book/`)**: A Next.js application featuring data-driven hardware schematics, memory ladders, interactive code runners, note-taking, and progress tracking.
2. **126 Hands-On Workbooks across 3 Tiers**:
   - **Beginner**: Guided exercises focusing on core APIs and mental models.
   - **Intermediate**: Optimization techniques, boundary conditions, and cache behavior.
   - **Champion**: Production-grade implementations, warp shuffles, and memory-bandwidth saturation.
3. **28 Production GPU Kernels (`src/kernels/`)**: FlashAttention-2, SwiGLU, RMSNorm, RoPE, AdamW, GEMM projections, and MoE Top-K gating.
4. **Containerized Environment (`Dockerfile` & `docker-compose.yml`)**: NVIDIA CUDA 12.4 + Clang + Node.js 20 LTS for zero-configuration, reproducible development.

---

## 📖 Curriculum Architecture

The curriculum is structured across three primary tracks:

```text
├── Volume 1: C++ Low-Level Systems Foundations (src/module1/)
│   ├── Part 1: Memory Layout & Value Semantics (1.1 - 1.5)
│   ├── Part 2: Resource Ownership, RAII & Allocators (1.6 - 1.8)
│   ├── Part 3: Virtual Memory & Cache Hierarchy (1.9 - 1.12)
│   ├── Part 4: Hardware SIMD & Bit Manipulation (1.13 - 1.15)
│   └── Part 5: Concurrency, NUMA & High-Throughput I/O (1.16 - 1.20)
│
├── Volume 2: CUDA Architecture & Execution Model (src/module2/ to src/module7/)
│   ├── Module 2: Host-Device Bridge & Modern C++ (2.1 - 2.5)
│   ├── Module 3: CUDA Threading Hierarchy & Hardware Execution (3.1 - 3.4)
│   ├── Module 4: Memory Hierarchy, Coalescing & SRAM Tiling (4.1 - 4.2)
│   ├── Module 5: High-Performance Parallel Primitives (5.1 - 5.3)
│   ├── Module 6: Low-Precision & Tensor Cores (6.1 - 6.2)
│   └── Module 7: Production LLM Inference & Training Primitives (7.1 - 7.6)
│
└── Volume 3: Production GPU Kernels Suite (src/kernels/)
    ├── Kernel Module 1: Dense Linear (GEMV, Tiled GEMM, WMMA, Split-K)
    ├── Kernel Module 2: Sparse & MoE (Top-K Gating, Fused MoE, CSR SpMV, 2:4 Sparsity)
    ├── Kernel Module 3: Attention (FlashAttention-2 Fwd/Bwd, PagedAttention, Sliding Window)
    ├── Kernel Module 4: Normalizations (RMSNorm, LayerNorm, Fused Add-Norm)
    ├── Kernel Module 5: Audio (STFT, Mel-Filterbank, Griffin-Lim)
    ├── Kernel Module 6: Vision & Video (Patches, RoPE-2D, Bicubic Downsample)
    └── Kernel Module 7: Quantization & Optimizers (FP8 Gemm, INT4 W4A16, AdamW, Lion)
```

---

## 🛠️ Prerequisites

### Hardware Requirements
- **NVIDIA GPU** (For CUDA Kernels & Benchmarks):
  - Supported Architectures: Turing (`sm_75`), Ampere (`sm_80`, `sm_86`), Ada Lovelace (`sm_89`), Hopper (`sm_90`), or Blackwell (`sm_100`).
  - Recommended VRAM: 8 GB or higher (RTX 3070+, RTX 4080+, A10/A100/H100).
- **CPU-Only Mode** (macOS / Systems without NVIDIA GPUs):
  - You can fully compile and run all **Volume 1 (C++ Systems, Module 1)** workbooks and run the **Digital Textbook Web App** locally.

### Software Prerequisites (Host System)
| Software / Package | Minimum Version | Recommended | Purpose |
| :--- | :--- | :--- | :--- |
| **NVIDIA Driver** | >= 525.60.13 | >= 535 or 550+ | GPU kernel execution |
| **CUDA Toolkit** | 11.8+ | 12.4+ | `nvcc` GPU compiler and runtime |
| **C++ Compiler** | GCC 9+ / Clang 14+ | Clang 16+ / GCC 12+ | C++17 / C++20 language features |
| **Node.js** | 18.x LTS | 20.x LTS | Digital Textbook web frontend & server |
| **npm** | 9.x+ | 10.x+ | Package manager for book application |
| **Docker Engine** | 20.10+ | 24.0+ | (Optional) Containerized execution |
| **Docker Compose** | 2.0+ | 2.20+ | (Optional) Multi-service lab runner |
| **NVIDIA Container Toolkit** | 1.13+ | Latest | (Optional) GPU passthrough inside Docker |

---

## 🐳 Quickstart with Docker (Recommended)

Docker provides an isolated, pre-configured environment containing **Ubuntu 22.04**, **NVIDIA CUDA 12.4.1 Devel**, **Clang**, **Build-Essential**, and **Node.js 20 LTS**.

### 1. Install Docker & NVIDIA Container Toolkit (Ubuntu/Debian)

If you haven't installed the NVIDIA Container Toolkit yet:

```bash
# 1. Configure NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3. Configure Docker runtime for NVIDIA GPUs and restart daemon
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU availability in Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

### 2. Launch the Lab with Docker Compose

Run the following command from the repository root:

```bash
# Build and run containerized lab (Digital Book + CUDA development environment)
docker compose up --build
```

- **Open the Digital Textbook & Web Lab**: Open `http://localhost:3000` in your browser.
- **Persistent Progress**: Your local repository files are mounted into `/workspace`. Any code you write in `exercise/`, `solution/`, or `playground.cpp` is saved immediately to your host machine.

To run in the background (detached mode):
```bash
docker compose up -d
```

To stop the container:
```bash
docker compose down
```

> [!NOTE]
> **Running on macOS or Non-GPU Machines:**
> If running Docker on macOS or a machine without an NVIDIA GPU, edit `docker-compose.yml` to comment out lines 20-26 (the `deploy.resources.reservations.devices` block), or use the CPU-only command below.

---

### 3. Alternative: Running Directly with Docker CLI

You can also build and run the image directly without Docker Compose:

```bash
# Build image
docker build -t cuda-systems-lab .

# Run with full GPU passthrough (Linux with NVIDIA GPU):
docker run --gpus all -it --rm \
  -p 3000:3000 \
  -v $(pwd):/workspace \
  --name cuda_lab \
  cuda-systems-lab

# Run on CPU-only machines (macOS / Intel / Apple Silicon):
docker run -it --rm \
  -p 3000:3000 \
  -v $(pwd):/workspace \
  --name cuda_lab_cpu \
  cuda-systems-lab
```

---

### 4. Opening an Interactive Shell Inside Docker

To compile and test workbooks inside the running container:

```bash
# Open an interactive bash session
docker exec -it cuda_systems_lab bash

# Inside the container, compile any workbook with nvcc or clang:
nvcc -O3 -std=c++17 -arch=sm_80 src/module3/3.1_threads_registers/exercise/beginner_workbook.cu -o /tmp/test && /tmp/test

# Run automated batch evaluation:
bash src/compile_and_run_cuda.sh all
```

---

## 💻 Native Host Installation (Without Docker)

If you prefer running directly on your host machine:

### 1. Ubuntu / Debian / WSL 2 Setup

```bash
# 1. Update and install essential build tools
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    clang \
    clang-format \
    llvm \
    git \
    curl \
    pkg-config \
    python3

# 2. Install Node.js 20 LTS
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# 3. Verify compilers and runtimes
clang++ --version
g++ --version
node -v
npm -v
nvidia-smi   # Requires NVIDIA GPU
nvcc --version
```

### 2. macOS Setup (Apple Silicon / Intel)

For running C++ Systems (Module 1), theory studies, and the web book:

```bash
# 1. Install Xcode Command Line Tools
xcode-select --install

# 2. Install Node.js via Homebrew
brew install node clang-format

# 3. Verify
clang++ --version
node -v
```

---

## 🌐 Launching the Digital Textbook & Web Lab

The interactive textbook gives you visual architecture schematics, theory walkthroughs, live kernel evaluation, and multi-playground sandboxes:

```bash
# 1. Navigate to the book application directory
cd book

# 2. Install dependencies
npm install

# 3. Start the Next.js development server
npm run dev
```

Open `http://localhost:3000` in your browser.

Key Features:
- **Categorized Navigation**: Smoothly switch between **C++ Systems**, **CUDA Architecture**, and **GPU Kernels**.
- **Interactive Schematics**: 64-bit byte ladders, stride-folding diagrams, warp shuffle diagrams, and cache line transfers.
- **Integrated Sandboxes**:
  - `playground.cpp` (Clang++ -O3 CPU systems benchmarking)
  - `playground.cu` (NVCC -O3 GPU kernel scratchpad)
  - `playground_kernel.cu` (Harness for benchmarking production LLM kernels)

---

## 🎯 Step-by-Step Learning Workflow

Every topic in `src/` follows a standardized structure:
```text
src/moduleX/<topic_name>/
├── theory.md               # Hardware mental models, formulas, and CUDA kernel APIs
├── exercise/               # Workbooks with // TODO gaps and integrated unit tests
│   ├── beginner_workbook.cu (or .cpp)
│   ├── intermediate_workbook.cu
│   └── champion_workbook.cu
└── solution/               # Solutions directory (your personal workspace)
```

1. **Read Theory**: Open `theory.md` to understand memory constraints, cache line rules, and hardware requirements.
2. **Open Exercise**: Open `exercise/beginner_workbook.cu` (or `.cpp`).
3. **Implement**: Fill in the `// TODO` gaps.
4. **Compile & Verify**:
   ```bash
   # Compile and test C++ workbook:
   clang++ -std=c++20 -O3 src/module1/1.1_basic_offsets/exercise/beginner_workbook.cpp -o /tmp/test && /tmp/test

   # Compile and test CUDA workbook:
   nvcc -O3 -std=c++17 -arch=sm_80 src/module3/3.1_threads_registers/exercise/beginner_workbook.cu -o /tmp/test && /tmp/test
   ```
5. **Scorecard**: Workbooks feature built-in test suites verifying edge cases, memory bounds, and correctness:
   ```text
   [Test 1: Output Correctness] PASSED
   [Test 2: Memory Bounds]      PASSED
   [Test 3: Numerical Limits]   PASSED
   Passed: 3 / 3 tests.
   ```
6. **Level Up**: Advance to `intermediate_workbook.cu` and `champion_workbook.cu`.

---

## ⚡ Automated Test Runner Scripts

We provide automated test runner scripts to evaluate your work across the repository:

```bash
# Run all C++ Systems workbooks (Module 1):
bash src/module1/compile_and_run_all.sh

# Run all CUDA workbooks (Modules 2 to 7):
bash src/compile_and_run_cuda.sh all

# Run a specific module:
bash src/compile_and_run_cuda.sh module2
bash src/compile_and_run_cuda.sh module3

# Run a specific kernel or topic:
bash src/compile_and_run_cuda.sh 7.6_flash_attention
```

---

## 📊 Complete 126-Exercise Progress Tracker

Copy and paste this checklist into your personal GitHub Issue, pull request, or Markdown notes to track your progress:

```markdown
# 🚀 CUDA ML Training Curriculum Progress Tracker

## Volume 1: C++ Low-Level Systems Foundations (60 Exercises)
### Part 1: Memory Layout & Value Semantics
- [ ] 1.1 Basic Offset & Increment — Beginner
- [ ] 1.1 Basic Offset & Increment — Intermediate
- [ ] 1.1 Basic Offset & Increment — Champion
- [ ] 1.2 Strides & Indirection — Beginner
- [ ] 1.2 Strides & Indirection — Intermediate
- [ ] 1.2 Strides & Indirection — Champion
- [ ] 1.3 Array Reductions & Swaps — Beginner
- [ ] 1.3 Array Reductions & Swaps — Intermediate
- [ ] 1.3 Array Reductions & Swaps — Champion
- [ ] 1.4 Multi-Array & Strided Ops — Beginner
- [ ] 1.4 Multi-Array & Strided Ops — Intermediate
- [ ] 1.4 Multi-Array & Strided Ops — Champion
- [ ] 1.5 Row/Col-Major Indexing — Beginner
- [ ] 1.5 Row/Col-Major Indexing — Intermediate
- [ ] 1.5 Row/Col-Major Indexing — Champion

### Part 2: Resource Ownership, RAII & Allocators
- [ ] 1.6 3D/4D Permute & Flatten — Beginner
- [ ] 1.6 3D/4D Permute & Flatten — Intermediate
- [ ] 1.6 3D/4D Permute & Flatten — Champion
- [ ] 1.7 Subgrids, Padding & Diagonals — Beginner
- [ ] 1.7 Subgrids, Padding & Diagonals — Intermediate
- [ ] 1.7 Subgrids, Padding & Diagonals — Champion
- [ ] 1.8 Pack, Wrap, Stencil & Crops — Beginner
- [ ] 1.8 Pack, Wrap, Stencil & Crops — Intermediate
- [ ] 1.8 Pack, Wrap, Stencil & Crops — Champion

### Part 3: Virtual Memory & Cache Hierarchy
- [ ] 1.9 Single & Array Allocations — Beginner
- [ ] 1.9 Single & Array Allocations — Intermediate
- [ ] 1.9 Single & Array Allocations — Champion
- [ ] 1.10 Multi-Dim & Alignment — Beginner
- [ ] 1.10 Multi-Dim & Alignment — Intermediate
- [ ] 1.10 Multi-Dim & Alignment — Champion
- [ ] 1.11 Arenas & Placements — Beginner
- [ ] 1.11 Arenas & Placements — Intermediate
- [ ] 1.11 Arenas & Placements — Champion
- [ ] 1.12 Lifetimes, Ownership & Resize — Beginner
- [ ] 1.12 Lifetimes, Ownership & Resize — Intermediate
- [ ] 1.12 Lifetimes, Ownership & Resize — Champion

### Part 4: Hardware SIMD & Bit Manipulation
- [ ] 1.13 Bitwise Flags & Masks — Beginner
- [ ] 1.13 Bitwise Flags & Masks — Intermediate
- [ ] 1.13 Bitwise Flags & Masks — Champion
- [ ] 1.14 Circular Buffers & Ring Queues — Beginner
- [ ] 1.14 Circular Buffers & Ring Queues — Intermediate
- [ ] 1.14 Circular Buffers & Ring Queues — Champion
- [ ] 1.15 Sparse Representations — Beginner
- [ ] 1.15 Sparse Representations — Intermediate
- [ ] 1.15 Sparse Representations — Champion

### Part 5: Concurrency, NUMA & High-Throughput I/O
- [ ] 1.16 Custom Allocators — Beginner
- [ ] 1.16 Custom Allocators — Intermediate
- [ ] 1.16 Custom Allocators — Champion
- [ ] 1.17 Embeddings & Projections — Beginner
- [ ] 1.17 Embeddings & Projections — Intermediate
- [ ] 1.17 Embeddings & Projections — Champion
- [ ] 1.18 Attention & Quantization — Beginner
- [ ] 1.18 Attention & Quantization — Intermediate
- [ ] 1.18 Attention & Quantization — Champion
- [ ] 1.19 ML Layer Offsets — Beginner
- [ ] 1.19 ML Layer Offsets — Intermediate
- [ ] 1.19 ML Layer Offsets — Champion
- [ ] 1.20 Pooling, Masks & Cycles — Beginner
- [ ] 1.20 Pooling, Masks & Cycles — Intermediate
- [ ] 1.20 Pooling, Masks & Cycles — Champion

---

## Volume 2: CUDA Architecture & Execution Model (66 Exercises)
### Module 2: Host-Device Bridge & Modern C++
- [ ] 2.1 Lambdas & Function Objects — Beginner
- [ ] 2.1 Lambdas & Function Objects — Intermediate
- [ ] 2.1 Lambdas & Function Objects — Champion
- [ ] 2.2 Templates & Type Traits — Beginner
- [ ] 2.2 Templates & Type Traits — Intermediate
- [ ] 2.2 Templates & Type Traits — Champion
- [ ] 2.3 RAII & Stream Managers — Beginner
- [ ] 2.3 RAII & Stream Managers — Intermediate
- [ ] 2.3 RAII & Stream Managers — Champion
- [ ] 2.4 Move Semantics & Tensor Views — Beginner
- [ ] 2.4 Move Semantics & Tensor Views — Intermediate
- [ ] 2.4 Move Semantics & Tensor Views — Champion
- [ ] 2.5 Host Parallel Execution — Beginner
- [ ] 2.5 Host Parallel Execution — Intermediate
- [ ] 2.5 Host Parallel Execution — Champion

### Module 3: CUDA Threading Hierarchy & Hardware Execution
- [ ] 3.1 Thread Registers & Local Spilling — Beginner
- [ ] 3.1 Thread Registers & Local Spilling — Intermediate
- [ ] 3.1 Thread Registers & Local Spilling — Champion
- [ ] 3.2 Warps, Divergence & Shuffles — Beginner
- [ ] 3.2 Warps, Divergence & Shuffles — Intermediate
- [ ] 3.2 Warps, Divergence & Shuffles — Champion
- [ ] 3.3 Blocks, Shared Memory & Bank Conflicts — Beginner
- [ ] 3.3 Blocks, Shared Memory & Bank Conflicts — Intermediate
- [ ] 3.3 Blocks, Shared Memory & Bank Conflicts — Champion
- [ ] 3.4 Grids, Occupancy & Grid-Stride Loops — Beginner
- [ ] 3.4 Grids, Occupancy & Grid-Stride Loops — Intermediate
- [ ] 3.4 Grids, Occupancy & Grid-Stride Loops — Champion

### Module 4: Memory Hierarchy, Coalescing & SRAM Tiling
- [ ] 4.1 Global Memory Coalescing & Read-Only Cache — Beginner
- [ ] 4.1 Global Memory Coalescing & Read-Only Cache — Intermediate
- [ ] 4.1 Global Memory Coalescing & Read-Only Cache — Champion
- [ ] 4.2 Shared Memory Matrix Tiling — Beginner
- [ ] 4.2 Shared Memory Matrix Tiling — Intermediate
- [ ] 4.2 Shared Memory Matrix Tiling — Champion

### Module 5: High-Performance Parallel Primitives
- [ ] 5.1 Warp & Block Reductions — Beginner
- [ ] 5.1 Warp & Block Reductions — Intermediate
- [ ] 5.1 Warp & Block Reductions — Champion
- [ ] 5.2 Global Atomic Operations — Beginner
- [ ] 5.2 Global Atomic Operations — Intermediate
- [ ] 5.2 Global Atomic Operations — Champion
- [ ] 5.3 Numerically Stable Reductions (Softmax & Loss) — Beginner
- [ ] 5.3 Numerically Stable Reductions (Softmax & Loss) — Intermediate
- [ ] 5.3 Numerically Stable Reductions (Softmax & Loss) — Champion

### Module 6: Low-Precision & Tensor Cores
- [ ] 6.1 BFloat16 (`__nv_bfloat16`) Arithmetic & Intrinsics — Beginner
- [ ] 6.1 BFloat16 (`__nv_bfloat16`) Arithmetic & Intrinsics — Intermediate
- [ ] 6.1 BFloat16 (`__nv_bfloat16`) Arithmetic & Intrinsics — Champion
- [ ] 6.2 Tensor Cores & WMMA API — Beginner
- [ ] 6.2 Tensor Cores & WMMA API — Intermediate
- [ ] 6.2 Tensor Cores & WMMA API — Champion

### Module 7: Production LLM Inference & Training Primitives
- [ ] 7.1 Elementwise & Reshaping (`reshape_3d`, `reshape_4d`, `residual_add`, `swiglu_fwd/bwd`) — Beginner
- [ ] 7.1 Elementwise & Reshaping (`reshape_3d`, `reshape_4d`, `residual_add`, `swiglu_fwd/bwd`) — Intermediate
- [ ] 7.1 Elementwise & Reshaping (`reshape_3d`, `reshape_4d`, `residual_add`, `swiglu_fwd/bwd`) — Champion
- [ ] 7.2 Token Embedding Lookup (`embedding_forward`) — Beginner
- [ ] 7.2 Token Embedding Lookup (`embedding_forward`) — Intermediate
- [ ] 7.2 Token Embedding Lookup (`embedding_forward`) — Champion
- [ ] 7.3 Normalization & RoPE (`rms_norm_fwd/bwd`, `rope_fwd/bwd`, `fused_add_norm_fwd/bwd`) — Beginner
- [ ] 7.3 Normalization & RoPE (`rms_norm_fwd/bwd`, `rope_fwd/bwd`, `fused_add_norm_fwd/bwd`) — Intermediate
- [ ] 7.3 Normalization & RoPE (`rms_norm_fwd/bwd`, `rope_fwd/bwd`, `fused_add_norm_fwd/bwd`) — Champion
- [ ] 7.4 Loss Functions & AdamW Optimizer (`cross_entropy`, `compute_loss`, `adamw_step`) — Beginner
- [ ] 7.4 Loss Functions & AdamW Optimizer (`cross_entropy`, `compute_loss`, `adamw_step`) — Intermediate
- [ ] 7.4 Loss Functions & AdamW Optimizer (`cross_entropy`, `compute_loss`, `adamw_step`) — Champion
- [ ] 7.5 GEMM Projections (`gemm_proj`, `gemm_bf16`, `gemm_gqa`, `gemm_ffn`, `fused_swiglu_gemm`) — Beginner
- [ ] 7.5 GEMM Projections (`gemm_proj`, `gemm_bf16`, `gemm_gqa`, `gemm_ffn`, `fused_swiglu_gemm`) — Intermediate
- [ ] 7.5 GEMM Projections (`gemm_proj`, `gemm_bf16`, `gemm_gqa`, `gemm_ffn`, `fused_swiglu_gemm`) — Champion
- [ ] 7.6 Tiled FlashAttention-2 Suite (`flash_attn_fwd`, `fused_attn_bwd`) — Beginner
- [ ] 7.6 Tiled FlashAttention-2 Suite (`flash_attn_fwd`, `fused_attn_bwd`) — Intermediate
- [ ] 7.6 Tiled FlashAttention-2 Suite (`flash_attn_fwd`, `fused_attn_bwd`) — Champion
```

---

## 🛡️ License & Contributing

Distributed under the MIT License. Contributions, additional architecture optimizations, and bug reports are welcome via Pull Requests.
