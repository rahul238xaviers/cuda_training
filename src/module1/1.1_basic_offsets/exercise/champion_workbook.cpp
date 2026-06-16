#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

// Global status reporter
void reportStatus(const std::string& name, bool passed) {
    std::cout << "  " << std::left << std::setw(65) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: Basic Memory Offsets (Champion Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // PROBLEM 12: Complex Precedence Combinations
    // Instruction: You are given an array of integers and a pointer 'ptr' pointing to index 0.
    //              Perform the four specified precedence checks (A, B, C, D) using pointer math.
    //              Predict the outcomes and ensure they match the assertions.
    {
        int arr[5] = {10, 20, 30, 40, 50};
        int* ptr = &arr[0];
        
        int valA = 0;
        int valB = 0;
        int valC = 0;
        int valD = 0;

        // TODO: Perform Step A: Use *ptr++ in one line to assign value to valA and increment ptr.
        
        // TODO: Perform Step B: Use *++ptr in one line to increment ptr and assign value to valB.
        
        // TODO: Perform Step C: Use ++*ptr in one line to increment the value under ptr and assign to valC.
        
        // TODO: Perform Step D: Use (*ptr)++ in one line to assign value to valD and then increment the value in memory.

        // Verify correct pointer advancement and array/value mutations
        bool ok = (valA == 10 && ptr == &arr[1]) && 
                  (valB == 30 && ptr == &arr[2]) && 
                  (valC == 31 && arr[2] == 31) && 
                  (valD == 31 && arr[2] == 32 && ptr == &arr[2]);
                  
        reportStatus("Problem 12: Precedence combinations (*ptr++, *++ptr, etc.)", ok);
        if (ok) passed++; total++;
    }

    // PROBLEM 13: Buffer-to-Buffer Stream Copy Loop
    // Instruction: Stream data from 'src' to 'dst' using a low-level copy loop with the
    //              dereference-postfix-increment idiom: *dst++ = *src++.
    //              Both pointer positions must advance to the end of their respective buffers.
    {
        float src[5] = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
        float dst[5] = {0.0f};
        
        float* src_ptr = src;
        float* dst_ptr = dst;
        
        // TODO: Write a while/for loop using *dst_ptr++ = *src_ptr++ to copy all 5 elements
        
        bool ok = (src_ptr == src + 5 && dst_ptr == dst + 5);
        for (int i = 0; i < 5; i++) {
            if (dst[i] != src[i]) ok = false;
        }
        
        reportStatus("Problem 13: Buffer stream copy (*dst++ = *src++)", ok);
        if (ok) passed++; total++;
    }

    // PROBLEM 14: Manual Byte Alignment (Bitwise Rounding)
    // Instruction: In CUDA memory layouts, memory addresses must be aligned to boundaries
    //              (e.g., 64-byte cache lines or 8-byte double floats) to ensure coalesced access.
    //              Write a bitwise alignment expression that rounds a raw address (uintptr_t)
    //              upwards to the nearest 64-byte boundary.
    {
        uintptr_t addr1 = 0x1003;
        uintptr_t addr2 = 0x1040;
        
        uintptr_t aligned_addr1 = 0;
        uintptr_t aligned_addr2 = 0;
        
        // TODO: Align addr1 and addr2 upwards to the nearest 64-byte boundary using bitwise ops
        
        bool ok = (aligned_addr1 == 0x1040 && aligned_addr2 == 0x1040);
        reportStatus("Problem 14: Bitwise manual memory alignment", ok);
        if (ok) passed++; total++;
    }

    // PROBLEM 15: Deep Indirection Write Chain (Double/Triple Pointers)
    // Instruction: Navigate a triple pointer hierarchy. Point a triple pointer 'p3' to
    //              double pointer 'p2', which points to pointer 'p1', which points to 'val'.
    //              Then, modify the value of 'val' to 999 using only 'p3' dereferencing.
    {
        int val = 42;
        int* p1 = &val;
        int** p2 = &p1;
        int*** p3 = nullptr;
        
        // TODO: Complete the pointer chain: Point p3 to p2, then modify 'val' to 999 through p3
        
        bool ok = (p3 == &p2 && ***p3 == 999 && val == 999);
        reportStatus("Problem 15: Deep pointer indirection (triple pointer write)", ok);
        if (ok) passed++; total++;
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
