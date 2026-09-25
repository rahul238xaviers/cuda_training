#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <cstdlib>

// =========================================================================
// CHAMPION WORKBOOK: Single & Array Allocations
//
// Module: 1.9 - Dynamic Heap Allocation & Memory Lifecycle
// Level:  Champion / High-Performance Systems Engineer
//
// Focus: O(1) Fixed-size slab allocators, PyTorch-style block caching
//        allocators, and 2MB huge-page staging buffer emulation.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.9_champion
//   ../../../output/1.9_champion
// =========================================================================

void reportStatus(const std::string& name, bool passed, double throughputGBs = -1.0) {
    std::cout << "  " << std::left << std::setw(60) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m";
        if (throughputGBs > 0.0) {
            std::cout << " (" << std::fixed << std::setprecision(2) << throughputGBs << " GB/s)";
        }
        std::cout << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: Single & Array Allocations (Champion Masterclass) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: O(1) Fixed-Size Slab/Block Allocator
    //
    // Context: In low-latency LLM serving, OS malloc incurs locks and system call overhead.
    //          A Slab Allocator pre-allocates a chunk of memory partitioned into fixed-size
    //          blocks (e.g. BLOCK_SIZE=256 bytes) linked via an embedded free list.
    //          Allocating and freeing takes strictly O(1) time without system calls.
    //
    // Task: Implement a Slab Allocator for NUM_BLOCKS=16 blocks of BLOCK_SIZE=256 bytes:
    //         - `void* allocate()`: pops a block from the free list
    //         - `void deallocate(void* ptr)`: pushes the block back onto the free list
    // -------------------------------------------------------------------------
    {
        const size_t BLOCK_SIZE = 256;
        const size_t NUM_BLOCKS = 16;
        std::vector<uint8_t> pool(NUM_BLOCKS * BLOCK_SIZE);

        struct FreeNode {
            FreeNode* next;
        };

        FreeNode* free_head = nullptr;

        // Initialize free list: chain blocks together
        for (size_t i = 0; i < NUM_BLOCKS; ++i) {
            FreeNode* node = reinterpret_cast<FreeNode*>(pool.data() + i * BLOCK_SIZE);
            node->next = free_head;
            free_head = node;
        }

        // Allocate 10 blocks
        std::vector<void*> allocated;
        for (size_t i = 0; i < 10; ++i) {
            if (free_head) {
                void* p = free_head;
                free_head = free_head->next;
                allocated.push_back(p);
            }
        }

        // Return 5 blocks
        for (size_t i = 0; i < 5; ++i) {
            FreeNode* node = reinterpret_cast<FreeNode*>(allocated.back());
            allocated.pop_back();
            node->next = free_head;
            free_head = node;
        }

        // Verify remaining allocations
        bool p1_passed = (allocated.size() == 5);
        // Count remaining free blocks (should be 16 - 5 = 11)
        int free_count = 0;
        FreeNode* curr = free_head;
        while (curr) {
            free_count++;
            curr = curr->next;
        }
        if (free_count != 11) p1_passed = false;

        reportStatus("Problem 1: O(1) Fixed-Size Slab/Block Allocator", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: PyTorch-Style Caching Allocator (Block Reuse)
    //
    // Context: In `c10::cuda::CUDACachingAllocator`, deallocated memory blocks are
    //          cached in a free pool by size. Subsequent allocations reuse cached
    //          buffers to avoid GPU synchronization overhead.
    //
    // Task: Implement a block cache that reuses an existing block if its capacity >= requested_size.
    //       Verify that allocating size 100, freeing it, and allocating size 80 returns
    //       the EXACT same pointer without invoking std::malloc a second time!
    // -------------------------------------------------------------------------
    {
        struct CachedBlock {
            void* ptr;
            size_t capacity;
            bool in_use;
        };

        std::vector<CachedBlock> block_pool;
        int os_malloc_calls = 0;

        auto alloc_fn = [&](size_t bytes) -> void* {
            // Check for reusable block
            for (auto& b : block_pool) {
                if (!b.in_use && b.capacity >= bytes) {
                    b.in_use = true;
                    return b.ptr;
                }
            }
            // Allocate new from OS
            void* p = std::malloc(bytes);
            os_malloc_calls++;
            block_pool.push_back({p, bytes, true});
            return p;
        };

        auto free_fn = [&](void* ptr) {
            for (auto& b : block_pool) {
                if (b.ptr == ptr) {
                    b.in_use = false;
                    return;
                }
            }
        };

        void* ptr1 = alloc_fn(1024); // OS call 1
        free_fn(ptr1);
        void* ptr2 = alloc_fn(512);  // Should reuse ptr1! OS calls must stay at 1!

        bool p2_passed = (ptr1 == ptr2) && (os_malloc_calls == 1);

        for (auto& b : block_pool) std::free(b.ptr);

        reportStatus("Problem 2: PyTorch-Style Caching Allocator Block Reuse", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: 2MB Huge-Page Alignment & Staging Buffer Rounding
    //
    // Context: High-performance NICs (InfiniBand / RoCE) and GPU Direct RDMA
    //          achieve peak bandwidth only when staging buffers are aligned to
    //          2 MB Huge-Page boundaries (2,097,152 bytes).
    //
    // Task: Given raw address `base_addr` = 0x7FFF00102040ULL:
    //       Calculate:
    //         1. `aligned_huge_page`: rounded up to nearest 2 MB boundary
    //         2. `padding_offset`: distance from base_addr to aligned_huge_page
    // -------------------------------------------------------------------------
    {
        const uint64_t HUGE_PAGE_SIZE = 2 * 1024 * 1024; // 2097152 bytes
        uint64_t base_addr = 0x7FFF00102040ULL;

        uint64_t aligned_huge_page = 0;
        uint64_t padding_offset = 0;

        // TODO: Compute aligned_huge_page and padding_offset using bitwise arithmetic.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        uint64_t expected_aligned = (base_addr + (HUGE_PAGE_SIZE - 1)) & ~(HUGE_PAGE_SIZE - 1);
        uint64_t expected_padding = expected_aligned - base_addr;

        bool p3_passed = (aligned_huge_page == expected_aligned) &&
                         (padding_offset == expected_padding) &&
                         (aligned_huge_page % HUGE_PAGE_SIZE == 0);

        reportStatus("Problem 3: 2MB Huge-Page Alignment & Bitwise Rounding", p3_passed);
        if (p3_passed) passed++;
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
