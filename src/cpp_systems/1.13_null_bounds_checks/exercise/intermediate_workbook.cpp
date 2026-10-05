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
// INTERMEDIATE WORKBOOK: Null, Bounds & Alignment Checks
//
// Module: 1.13 - Hardware Alignment, Struct Padding & Type Punning
// Level:  Intermediate
//
// Focus: 3D bounding box slicing with zero-padding, circular ring buffer bounds
//        guards, and power-of-two aligned memory pool sub-allocation.
//
// Compilation:
//   clang++ -std=c++20 -O3 exercise/intermediate_workbook.cpp -o ../../../output/1.13_intermediate
//   ../../../output/1.13_intermediate
// =========================================================================

void reportStatus(const std::string& name, bool passed) {
    std::cout << "  " << std::left << std::setw(60) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

// Template class for Problem 2
template <typename T, size_t Capacity>
class BoundedRingQueue {
    T buffer[Capacity];
    size_t head = 0;
    size_t tail = 0;
    size_t count = 0;
public:
    // --- YOUR CODE STARTS HERE ---
    bool push(const T& val) {
        if (count == Capacity) return false;
        buffer[head] = val;
        head = (head + 1) % Capacity;
        count++;
        return true;
    }

    bool pop(T& val) {
        if (count == 0) return false;
        val = buffer[tail];
        tail = (tail + 1) % Capacity;
        count--;
        return true;
    }

    size_t size() const { return count; }
    bool isFull() const { return count == Capacity; }
    bool isEmpty() const { return count == 0; }
    // --- YOUR CODE ENDS HERE ---
};

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: Null, Bounds & Alignment Checks (Intermediate) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: 3D Strided Tensor Safe Bounding Box Slicer with Zero Padding
    //
    // Context: In 3D computer vision (NeRFs, medical voxel grids, LiDAR 3D convolutions),
    //          kernels extract 3D subvolumes around points of interest. Often, the
    //          bounding box extends beyond the tensor boundaries. Safe runtimes must
    //          copy valid voxels and fill out-of-bounds voxels with zero without
    //          causing illegal out-of-bounds memory accesses.
    //
    // Task: Implement `sliceBox3D(const float* src, int D, int H, int W,
    //                            float* dst, int d0, int d1, int h0, int h1, int w0, int w1)`:
    //       Destination dimensions:
    //         out_D = d1 - d0, out_H = h1 - h0, out_W = w1 - w0.
    //       For each local index (od, oh, ow) in destination:
    //         global coordinate: g_d = d0 + od, g_h = h0 + oh, g_w = w0 + ow.
    //         If 0 <= g_d < D and 0 <= g_h < H and 0 <= g_w < W:
    //           dst[od * (out_H * out_W) + oh * out_W + ow] = src[g_d * (H * W) + g_h * W + g_w];
    //         Else:
    //           dst[od * (out_H * out_W) + oh * out_W + ow] = 0.0f;
    // -------------------------------------------------------------------------
    {
        auto sliceBox3D = [](const float* src, int D, int H, int W,
                             float* dst, int d0, int d1, int h0, int h1, int w0, int w1) {
            // --- YOUR CODE STARTS HERE ---
            int out_D = d1 - d0;
            int out_H = h1 - h0;
            int out_W = w1 - w0;
            if (!src || !dst || out_D <= 0 || out_H <= 0 || out_W <= 0) return;

            for (int od = 0; od < out_D; ++od) {
                int gd = d0 + od;
                for (int oh = 0; oh < out_H; ++oh) {
                    int gh = h0 + oh;
                    for (int ow = 0; ow < out_W; ++ow) {
                        int gw = w0 + ow;
                        size_t dst_idx = static_cast<size_t>(od) * (out_H * out_W) +
                                         static_cast<size_t>(oh) * out_W + ow;
                        if (gd >= 0 && gd < D && gh >= 0 && gh < H && gw >= 0 && gw < W) {
                            size_t src_idx = static_cast<size_t>(gd) * (H * W) +
                                             static_cast<size_t>(gh) * W + gw;
                            dst[dst_idx] = src[src_idx];
                        } else {
                            dst[dst_idx] = 0.0f;
                        }
                    }
                }
            }
            // --- YOUR CODE ENDS HERE ---
        };

        const int D = 8, H = 16, W = 32;
        std::vector<float> src(D * H * W);
        for (int i = 0; i < D * H * W; ++i) src[i] = static_cast<float>(i + 1);

        // Subgrid extending beyond boundaries
        int d0 = -2, d1 = 6;  // out_D = 8
        int h0 = 10, h1 = 20; // out_H = 10 (exceeds H=16)
        int w0 = -4, w1 = 12; // out_W = 16 (starts negative)

        int out_D = d1 - d0;
        int out_H = h1 - h0;
        int out_W = w1 - w0;
        std::vector<float> dst(out_D * out_H * out_W, -999.0f);

        sliceBox3D(src.data(), D, H, W, dst.data(), d0, d1, h0, h1, w0, w1);

        bool ok = true;
        for (int od = 0; od < out_D; ++od) {
            int gd = d0 + od;
            for (int oh = 0; oh < out_H; ++oh) {
                int gh = h0 + oh;
                for (int ow = 0; ow < out_W; ++ow) {
                    int gw = w0 + ow;
                    size_t dst_idx = static_cast<size_t>(od) * (out_H * out_W) +
                                     static_cast<size_t>(oh) * out_W + ow;
                    float val = dst[dst_idx];
                    if (gd >= 0 && gd < D && gh >= 0 && gh < H && gw >= 0 && gw < W) {
                        float expected = src[gd * (H * W) + gh * W + gw];
                        if (val != expected) ok = false;
                    } else {
                        if (val != 0.0f) ok = false;
                    }
                }
            }
        }

        reportStatus("Problem 1: 3D Strided Safe Bounding Box Slicer", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Circular Bounded Ring Buffer Guard with Wrap Arithmetic
    //
    // Context: Asynchronous streaming of training batches between Host (CPU)
    //          and Device (GPU) uses lock-free circular ring queues. Correct
    //          index wrapping and overflow/underflow bounds checks are critical
    //          to prevent dropping or corrupting batches.
    //
    // Task: Implement `BoundedRingQueue<T, Capacity>`:
    //       - `bool push(const T& val)`:
    //           If count == Capacity, return false.
    //           Otherwise store at `head`, update `head = (head + 1) % Capacity`,
    //           increment `count`, return true.
    //       - `bool pop(T& val)`:
    //           If count == 0, return false.
    //           Otherwise retrieve from `tail`, update `tail = (tail + 1) % Capacity`,
    //           decrement `count`, return true.
    //       - `size_t size() const`, `bool isFull() const`, `bool isEmpty() const`.
    // -------------------------------------------------------------------------
    {
        struct BatchTask {
            int batch_id;
            uint64_t timestamp;
        };

        BoundedRingQueue<BatchTask, 8> queue;
        bool ok = true;

        if (!queue.isEmpty() || queue.isFull() || queue.size() != 0) ok = false;

        // Fill capacity
        for (int i = 0; i < 8; ++i) {
            if (!queue.push({i, static_cast<uint64_t>(1000 + i)})) ok = false;
        }
        if (!queue.isFull() || queue.size() != 8) ok = false;
        // Overflow attempt
        if (queue.push({99, 9999})) ok = false;

        // Pop 4 items
        for (int i = 0; i < 4; ++i) {
            BatchTask item;
            if (!queue.pop(item)) ok = false;
            if (item.batch_id != i || item.timestamp != static_cast<uint64_t>(1000 + i)) ok = false;
        }
        if (queue.size() != 4 || queue.isFull() || queue.isEmpty()) ok = false;

        // Push 4 more items (wrapping around the ring buffer array)
        for (int i = 8; i < 12; ++i) {
            if (!queue.push({i, static_cast<uint64_t>(1000 + i)})) ok = false;
        }
        if (!queue.isFull() || queue.size() != 8) ok = false;

        // Drain all items
        for (int i = 4; i < 12; ++i) {
            BatchTask item;
            if (!queue.pop(item)) ok = false;
            if (item.batch_id != i) ok = false;
        }
        if (!queue.isEmpty() || queue.size() != 0) ok = false;
        BatchTask dummy;
        if (queue.pop(dummy)) ok = false; // underflow

        reportStatus("Problem 2: Circular Bounded Ring Queue Guard", ok);
        if (ok) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Aligned Memory Pool Segmenter & Validator
    //
    // Context: Tensor allocators pre-reserve a large GPU/CPU memory arena. When
    //          sub-allocating tensors of varying precisions (e.g. FP32, FP16, INT8),
    //          each block must satisfy strict hardware alignment (16, 32, 64, or 128 bytes).
    //
    // Task: Implement `AlignedArena`:
    //       Constructed with raw pointer `buffer` and `capacity_bytes`.
    //       `void* allocate(size_t size, size_t alignment)`:
    //         - Advances the current offset to the next multiple of `alignment`.
    //         - Checks if `aligned_offset + size <= capacity`.
    //         - If valid, advances internal offset to `aligned_offset + size`,
    //           and returns pointer to the aligned start address.
    //         - If out-of-memory, returns `nullptr`.
    // -------------------------------------------------------------------------
    {
        class AlignedArena {
            uint8_t* base_ptr;
            size_t capacity;
            size_t current_offset;
        public:
            AlignedArena(uint8_t* buffer, size_t cap)
                : base_ptr(buffer), capacity(cap), current_offset(0) {}

            // --- YOUR CODE STARTS HERE ---
            void* allocate(size_t size, size_t alignment) {
                if (!base_ptr || size == 0 || alignment == 0) return nullptr;
                uintptr_t current_addr = reinterpret_cast<uintptr_t>(base_ptr + current_offset);
                uintptr_t mask = alignment - 1;
                uintptr_t aligned_addr = (current_addr + mask) & ~mask;
                size_t padding = aligned_addr - current_addr;

                if (current_offset + padding + size > capacity) {
                    return nullptr; // Out of memory
                }

                current_offset += padding + size;
                return reinterpret_cast<void*>(aligned_addr);
            }

            size_t getUsedBytes() const { return current_offset; }
            size_t getRemainingBytes() const { return capacity - current_offset; }
            // --- YOUR CODE ENDS HERE ---
        };

        const size_t ARENA_SIZE = 4096;
        std::vector<uint8_t> pool(ARENA_SIZE, 0);
        AlignedArena arena(pool.data(), ARENA_SIZE);

        bool ok = true;
        // Alloc 1: 128 bytes, 64-byte alignment
        void* p1 = arena.allocate(128, 64);
        if (!p1 || reinterpret_cast<uintptr_t>(p1) % 64 != 0) ok = false;

        // Alloc 2: 7 bytes, 16-byte alignment
        void* p2 = arena.allocate(7, 16);
        if (!p2 || reinterpret_cast<uintptr_t>(p2) % 16 != 0) ok = false;
        if (p2 <= p1) ok = false;

        // Alloc 3: 500 bytes, 128-byte alignment
        void* p3 = arena.allocate(500, 128);
        if (!p3 || reinterpret_cast<uintptr_t>(p3) % 128 != 0) ok = false;

        // Attempt allocation that exceeds remaining capacity
        void* p_fail = arena.allocate(ARENA_SIZE, 64);
        if (p_fail != nullptr) ok = false;

        reportStatus("Problem 3: Aligned Memory Pool Segmenter", ok);
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
