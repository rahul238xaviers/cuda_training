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
// CHAMPION WORKBOOK: Arenas & Placements
//
// Module: 1.11 - Dynamic Heap Allocation & Memory Lifecycle
// Level:  Champion / High-Performance Systems Engineer
//
// Focus: RAII Scoped Arena Guards, Ring Buffer Memory Arenas, and
//        Arena vs std::malloc Allocation Throughput Benchmarking.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/champion_workbook.cpp -o ../../../output/1.11_champion
//   ../../../output/1.11_champion
// =========================================================================

void reportStatus(const std::string& name, bool passed, double throughputGBs = -1.0) {
    std::cout << "  " << std::left << std::setw(60) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m";
        if (throughputGBs > 0.0) {
            std::cout << " (" << std::fixed << std::setprecision(2) << throughputGBs << " Mops/s)";
        }
        std::cout << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: Arenas & Placements (Champion Masterclass) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Scoped Arena Lifetime Guard (RAII ArenaScope)
    //
    // Context: In deep learning execution engines, sub-operations (like attention
    //          or LayerNorm) allocate scratch space inside an `ArenaScope`.
    //          When the scope exits, its destructor rewinds the arena automatically,
    //          completely eliminating memory leaks without explicit frees.
    //
    // Task: Implement `Arena` and `ArenaScope` RAII helper:
    //       Test nested scopes:
    //         Arena starts at 0
    //         Alloc 100 bytes (offset = 100)
    //         Enter Scope A -> Alloc 200 bytes (offset = 300)
    //           Enter Scope B -> Alloc 400 bytes (offset = 700)
    //           Exit Scope B -> offset must rewind to 300!
    //         Exit Scope A -> offset must rewind to 100!
    // -------------------------------------------------------------------------
    {
        class Arena {
        public:
            std::vector<uint8_t> mem;
            size_t offset = 0;
            Arena(size_t cap) : mem(cap) {}

            void* alloc(size_t sz) {
                if (offset + sz > mem.size()) return nullptr;
                void* p = &mem[offset];
                offset += sz;
                return p;
            }
        };

        struct ArenaScope {
            Arena& arena;
            size_t saved_marker;

            ArenaScope(Arena& a) : arena(a), saved_marker(a.offset) {}
            ~ArenaScope() { arena.offset = saved_marker; }

            ArenaScope(const ArenaScope&) = delete;
            ArenaScope& operator=(const ArenaScope&) = delete;
        };

        Arena arena(2048);
        arena.alloc(100);

        bool p1_passed = true;
        if (arena.offset != 100) p1_passed = false;

        {
            ArenaScope scopeA(arena);
            arena.alloc(200);
            if (arena.offset != 300) p1_passed = false;

            {
                ArenaScope scopeB(arena);
                arena.alloc(400);
                if (arena.offset != 700) p1_passed = false;
            } // scopeB destroyed -> rewinds to 300
            if (arena.offset != 300) p1_passed = false;
        } // scopeA destroyed -> rewinds to 100
        if (arena.offset != 100) p1_passed = false;

        reportStatus("Problem 1: Scoped Arena Lifetime Guard (RAII ArenaScope)", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Circular Ring Buffer Arena for Continuous Inference
    //
    // Context: In continuous streaming token generation, requests continuously allocate
    //          and free temporary scratchpads. A ring arena wraps around its buffer.
    //
    // Task: Implement a ring arena of capacity 1024 bytes:
    //       Allocate 4 chunks of 300 bytes sequentially (total 1200 bytes requested):
    //       Chunk 0: [0..300)
    //       Chunk 1: [300..600)
    //       Chunk 2: [600..900)
    //       Chunk 3: 300 bytes cannot fit in [900..1024), so it wraps to [0..300)!
    // -------------------------------------------------------------------------
    {
        const size_t CAPACITY = 1024;
        uint8_t ring_storage[CAPACITY];
        size_t head = 0;

        auto ring_alloc = [&](size_t bytes) -> void* {
            if (head + bytes <= CAPACITY) {
                void* p = &ring_storage[head];
                head += bytes;
                return p;
            } else {
                // Wrap around to beginning
                head = 0;
                void* p = &ring_storage[head];
                head += bytes;
                return p;
            }
        };

        void* c0 = ring_alloc(300);
        void* c1 = ring_alloc(300);
        void* c2 = ring_alloc(300);
        void* c3 = ring_alloc(300); // Must wrap to ring_storage[0]!

        bool p2_passed = (c0 == &ring_storage[0]) &&
                         (c1 == &ring_storage[300]) &&
                         (c2 == &ring_storage[600]) &&
                         (c3 == &ring_storage[0]) &&
                         (head == 300);

        reportStatus("Problem 2: Circular Ring Buffer Memory Arena", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: Arena vs std::malloc Allocation Throughput Benchmark
    //
    // Context: Demonstrates why low-latency GPU runtimes never use std::malloc
    //          in inner loops. A bump arena executes millions of allocations per second.
    //
    // Task: Perform 100,000 allocations of 64 bytes using an Arena.
    //       Measure throughput in Millions of Operations per Second (Mops/s).
    // -------------------------------------------------------------------------
    {
        const int NUM_ALLOCS = 100000;
        const size_t POOL_SIZE = 16 * 1024 * 1024; // 16 MB
        std::vector<uint8_t> pool(POOL_SIZE);
        size_t offset = 0;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < NUM_ALLOCS; ++i) {
            // Bump allocation
            void* p = &pool[offset];
            offset = (offset + 64) % (POOL_SIZE - 64);
            // Write a dummy float to simulate usage
            *reinterpret_cast<float*>(p) = static_cast<float>(i);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();
        double mops = (NUM_ALLOCS / elapsed_sec) / 1e6;

        bool p3_passed = (mops > 10.0); // Arena should easily exceed 10 Mops/s
        reportStatus("Problem 3: Arena Allocation Throughput Benchmark", p3_passed, mops);
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
