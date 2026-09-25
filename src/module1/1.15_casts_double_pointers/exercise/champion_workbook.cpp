#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <cstdint>

// =========================================================================
// CHAMPION WORKBOOK: Type Casts & Double Pointers
//
// Module: 1.15 - Hardware Alignment, Struct Padding & Type Punning
// Level:  Champion
//
// Focus: Multi-level GPU virtual memory page table walk (uint64_t** MMU),
//        PyTorch-style 2D function pointer dispatch table, and intrusive
//        multi-tensor buffer deserialization with 64-byte alignment checks.
//
// Compilation:
//   clang++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.15_champion
//   ../../../output/1.15_champion
// =========================================================================

void reportStatus(const std::string& name, bool passed) {
    std::cout << "  " << std::left << std::setw(60) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

// -------------------------------------------------------------------------
// Helper Types for Problem 2
// -------------------------------------------------------------------------
enum class DeviceBackend { CPU = 0, CUDA = 1, NUM_BACKENDS = 2 };
enum class TensorScalarType { FP32 = 0, INT32 = 1, NUM_DTYPES = 2 };

using KernelFn = void(*)(const void*, const void*, void*, size_t);

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: Type Casts & Double Pointers (Champion Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Multi-Level Virtual Memory Page Table Translation (uint64_t** MMU)
    //
    // Context: GPU Memory Management Units (MMUs) translate Virtual Addresses (VAs)
    //          to Physical Addresses (PAs) using multi-level page tables.
    //          Architecture:
    //            - Page size: 4096 bytes (12-bit offset).
    //            - VA structure (30 bits total):
    //              [Dir Index: 9 bits (bits 21..29)]
    //              [Table Index: 9 bits (bits 12..20)]
    //              [Offset: 12 bits (bits 0..11)]
    //            - Page Directory: `uint64_t** dir` (512 entries).
    //              Each entry points to a Page Table `uint64_t*` (512 entries).
    //
    // Task: Implement:
    //       1. `mapPage(uint64_t** dir, uint64_t vaddr, uint64_t physical_frame)`:
    //          Extracts dir_idx and tbl_idx.
    //          If `dir[dir_idx] == nullptr`, allocates `new uint64_t[512]` initialized to 0.
    //          Sets `dir[dir_idx][tbl_idx] = physical_frame | 0x1` (bit 0 = Present bit).
    //       2. `translateVA(uint64_t** dir, uint64_t vaddr, uint64_t& out_paddr)`:
    //          Returns true if validly mapped (Present bit set), extracting physical_frame
    //          and appending offset: `out_paddr = (entry & ~0xFFFULL) | offset`.
    //          Returns false on page fault.
    // -------------------------------------------------------------------------
    {
        auto mapPage = [](uint64_t** dir, uint64_t vaddr, uint64_t physical_frame) {
            // --- YOUR CODE STARTS HERE ---
            uint64_t dir_idx = (vaddr >> 21) & 0x1FF;
            uint64_t tbl_idx = (vaddr >> 12) & 0x1FF;

            if (dir[dir_idx] == nullptr) {
                dir[dir_idx] = new uint64_t[512]();
            }
            dir[dir_idx][tbl_idx] = (physical_frame & ~0xFFFULL) | 0x1;
            // --- YOUR CODE ENDS HERE ---
        };

        auto translateVA = [](uint64_t** dir, uint64_t vaddr, uint64_t& out_paddr) -> bool {
            // --- YOUR CODE STARTS HERE ---
            uint64_t dir_idx = (vaddr >> 21) & 0x1FF;
            uint64_t tbl_idx = (vaddr >> 12) & 0x1FF;
            uint64_t offset  = vaddr & 0xFFF;

            if (dir[dir_idx] == nullptr) return false;
            uint64_t entry = dir[dir_idx][tbl_idx];
            if ((entry & 0x1) == 0) return false; // not present

            out_paddr = (entry & ~0xFFFULL) | offset;
            return true;
            // --- YOUR CODE ENDS HERE ---
        };

        uint64_t* directory[512] = {nullptr};

        uint64_t va1 = 0x00401050; // dir_idx=2, tbl_idx=1, offset=0x050
        uint64_t frame1 = 0x10000000;
        mapPage(directory, va1, frame1);

        uint64_t va2 = 0x00802100; // dir_idx=4, tbl_idx=2, offset=0x100
        uint64_t frame2 = 0x20000000;
        mapPage(directory, va2, frame2);

        bool ok = true;
        uint64_t pa = 0;
        if (!translateVA(directory, va1, pa) || pa != (frame1 | 0x050)) ok = false;
        if (!translateVA(directory, va2, pa) || pa != (frame2 | 0x100)) ok = false;

        // Unmapped address fault test
        uint64_t unmapped_va = 0x01000000;
        if (translateVA(directory, unmapped_va, pa)) ok = false;

        // Cleanup
        for (int i = 0; i < 512; ++i) {
            delete[] directory[i];
        }

        reportStatus("Problem 1: Multi-Level Page Table (uint64_t** MMU)", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: PyTorch ATen-Style 2D Function Pointer Dispatch Table
    //
    // Context: Deep learning frameworks dispatch kernel execution through a
    //          2D lookup table indexed by `[DeviceBackend][TensorScalarType]`.
    //          The kernel signature is type-erased `void(*)(const void*, const void*, void*, size_t)`.
    //
    // Task: Implement `DispatchEngine`:
    //       - Stores `KernelFn table[NUM_BACKENDS][NUM_DTYPES];`
    //       - `registerOp(DeviceBackend b, TensorScalarType t, KernelFn fn)`
    //       - `invokeOp(DeviceBackend b, TensorScalarType t, const void* a, const void* b, void* out, size_t n)`
    //       Define concrete kernels:
    //       - cpu_fp32_add: adds float arrays.
    //       - cpu_int32_add: adds int32_t arrays.
    // -------------------------------------------------------------------------
    {
        auto cpu_fp32_add = [](const void* a, const void* b, void* out, size_t n) {
            const float* fa = reinterpret_cast<const float*>(a);
            const float* fb = reinterpret_cast<const float*>(b);
            float* fo = reinterpret_cast<float*>(out);
            for (size_t i = 0; i < n; ++i) fo[i] = fa[i] + fb[i];
        };

        auto cpu_int32_add = [](const void* a, const void* b, void* out, size_t n) {
            const int32_t* ia = reinterpret_cast<const int32_t*>(a);
            const int32_t* ib = reinterpret_cast<const int32_t*>(b);
            int32_t* io = reinterpret_cast<int32_t*>(out);
            for (size_t i = 0; i < n; ++i) io[i] = ia[i] + ib[i];
        };

        class DispatchEngine {
            KernelFn table[static_cast<size_t>(DeviceBackend::NUM_BACKENDS)]
                          [static_cast<size_t>(TensorScalarType::NUM_DTYPES)] = {{nullptr}};
        public:
            // --- YOUR CODE STARTS HERE ---
            void registerOp(DeviceBackend b, TensorScalarType t, KernelFn fn) {
                table[static_cast<size_t>(b)][static_cast<size_t>(t)] = fn;
            }

            void invokeOp(DeviceBackend b, TensorScalarType t,
                          const void* a, const void* b_buf, void* out, size_t n) {
                KernelFn fn = table[static_cast<size_t>(b)][static_cast<size_t>(t)];
                if (fn) {
                    fn(a, b_buf, out, n);
                }
            }
            // --- YOUR CODE ENDS HERE ---
        };

        DispatchEngine engine;
        engine.registerOp(DeviceBackend::CPU, TensorScalarType::FP32, cpu_fp32_add);
        engine.registerOp(DeviceBackend::CPU, TensorScalarType::INT32, cpu_int32_add);

        const size_t N = 8;
        std::vector<float> fa = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        std::vector<float> fb = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
        std::vector<float> fo(N, 0.0f);

        std::vector<int32_t> ia = {10, 20, 30, 40, 50, 60, 70, 80};
        std::vector<int32_t> ib = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<int32_t> io(N, 0);

        engine.invokeOp(DeviceBackend::CPU, TensorScalarType::FP32, fa.data(), fb.data(), fo.data(), N);
        engine.invokeOp(DeviceBackend::CPU, TensorScalarType::INT32, ia.data(), ib.data(), io.data(), N);

        bool ok = true;
        for (size_t i = 0; i < N; ++i) {
            if (fo[i] != fa[i] + fb[i]) ok = false;
            if (io[i] != ia[i] + ib[i]) ok = false;
        }

        reportStatus("Problem 2: 2D Type-Erased Function Pointer Dispatch Table", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Intrusive Multi-Tensor Buffer Deserializer
    //
    // Context: In binary model formats (Safetensors / GGUF), weight tensors are
    //          stored with 64-byte alignment in a shared memory-mapped blob.
    //          A header table specifies `{ name, offset, size_bytes }`.
    //
    // Task: Implement `deserializeTensors(const uint8_t* blob, size_t blob_size,
    //                                    const std::vector<TensorDesc>& descs,
    //                                    std::vector<const float*>& out_ptrs)`:
    //       For each descriptor:
    //         - Checks `offset + size_bytes <= blob_size`.
    //         - Checks that the effective address `(blob + offset)` is 64-byte aligned.
    //         - If aligned and valid, `out_ptrs[i] = reinterpret_cast<const float*>(blob + offset)`.
    //         - If invalid or misaligned, set `out_ptrs[i] = nullptr`.
    // -------------------------------------------------------------------------
    {
        struct TensorDesc {
            char name[16];
            size_t byte_offset;
            size_t byte_size;
        };

        auto deserializeTensors = [](const uint8_t* blob, size_t blob_size,
                                     const std::vector<TensorDesc>& descs,
                                     std::vector<const float*>& out_ptrs) {
            // --- YOUR CODE STARTS HERE ---
            out_ptrs.resize(descs.size(), nullptr);
            for (size_t i = 0; i < descs.size(); ++i) {
                const auto& d = descs[i];
                if (d.byte_offset + d.byte_size > blob_size) {
                    out_ptrs[i] = nullptr;
                    continue;
                }
                const uint8_t* addr = blob + d.byte_offset;
                if ((reinterpret_cast<uintptr_t>(addr) & 63) != 0) {
                    out_ptrs[i] = nullptr; // Misaligned
                    continue;
                }
                out_ptrs[i] = reinterpret_cast<const float*>(addr);
            }
            // --- YOUR CODE ENDS HERE ---
        };

        const size_t BLOB_SIZE = 16384;
        std::vector<uint8_t> storage(BLOB_SIZE, 0);
        uint8_t* raw_blob = storage.data();
        // Compute 64-byte aligned base inside storage
        uintptr_t raw_addr = reinterpret_cast<uintptr_t>(raw_blob);
        size_t pad = (64 - (raw_addr % 64)) % 64;
        uint8_t* aligned_blob = raw_blob + pad;
        size_t effective_blob_size = BLOB_SIZE - pad;

        std::vector<TensorDesc> descs = {
            {"q_weight", 0, 1024},         // aligned at 0
            {"k_weight", 1024, 1024},      // aligned at 1024 (1024 % 64 == 0)
            {"misaligned_v", 1050, 1024},  // misaligned (1050 % 64 != 0)
            {"overflow_w", effective_blob_size - 100, 1024} // exceeds blob size
        };

        std::vector<const float*> out_ptrs;
        deserializeTensors(aligned_blob, effective_blob_size, descs, out_ptrs);

        bool ok = true;
        if (out_ptrs.size() != 4) ok = false;
        if (out_ptrs[0] != reinterpret_cast<const float*>(aligned_blob)) ok = false;
        if (out_ptrs[1] != reinterpret_cast<const float*>(aligned_blob + 1024)) ok = false;
        if (out_ptrs[2] != nullptr) ok = false; // must reject misaligned
        if (out_ptrs[3] != nullptr) ok = false; // must reject overflow

        reportStatus("Problem 3: Intrusive Tensor Buffer Deserializer", ok);
        if (ok) passed++;
        total++;
    }

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "--- SCORECARD ---" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  Passed: " << passed << " / " << total << " tests." << std::endl;
    if (passed == total) {
        std::cout << "\033[1;32m  [STATUS] ALL " << total << " TESTS PASSED! \033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m  [STATUS] INCOMPLETE (" << (total - passed) << " tests failed) \033[0m" << std::endl;
    }
    std::cout << "=================================================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
