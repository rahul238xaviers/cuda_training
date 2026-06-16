#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cmath>

// Global status reporter
void reportStatus(const std::string& name, bool passed) {
    std::cout << "  " << std::left << std::setw(65) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

// Struct for Problem 11
struct PaddedStruct {
    char x;
    double y;
};

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: Basic Memory Offsets (Intermediate Level) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // PROBLEM 7: Pointer Distance in Elements
    // Instruction: Given ptr_a and ptr_b pointing to different locations in 'arr',
    //              calculate the distance (in elements) between ptr_b and ptr_a.
    {
        double arr[10] = {0};
        double* ptr_a = &arr[2];
        double* ptr_b = &arr[8];
        ptrdiff_t diff = 0;
        
        // TODO: Compute distance (in elements) ptr_b - ptr_a
        
        bool ok = (diff == 6);
        reportStatus("Problem 7: Pointer distance in elements", ok);
        if (ok) passed++; total++;
    }

    // PROBLEM 8: Pointer Distance in Bytes
    // Instruction: Calculate the physical distance in bytes between ptr_b and ptr_a
    //              by casting both pointers to char* before performing subtraction.
    {
        int arr[10] = {0};
        int* ptr_a = &arr[1];
        int* ptr_b = &arr[6];
        ptrdiff_t byte_diff = 0;
        
        // TODO: Calculate distance in bytes between ptr_b and ptr_a
        
        bool ok = (byte_diff == 5 * sizeof(int));
        reportStatus("Problem 8: Pointer distance in bytes", ok);
        if (ok) passed++; total++;
    }

    // PROBLEM 9: Multi-type Step Sizes
    // Instruction: Observe how shifting pointers of different types behaves differently.
    //              Add an offset of 3 to double* ptrD, float* ptrF, and char* ptrC.
    //              Store the resulting addresses (as uintptr_t values) in addrD, addrF, and addrC.
    {
        double arrD[5] = {0};
        float arrF[5] = {0};
        char arrC[5] = {0};
        
        double* ptrD = arrD;
        float* ptrF = arrF;
        char* ptrC = arrC;
        
        uintptr_t addrD = 0;
        uintptr_t addrF = 0;
        uintptr_t addrC = 0;
        
        // TODO: Shift ptrD, ptrF, and ptrC by 3 elements and store their new address values
        
        bool ok = (addrD == reinterpret_cast<uintptr_t>(arrD + 3) &&
                   addrF == reinterpret_cast<uintptr_t>(arrF + 3) &&
                   addrC == reinterpret_cast<uintptr_t>(arrC + 3));
        reportStatus("Problem 9: Multi-type pointer stride check", ok);
        if (ok) passed++; total++;
    }

    // PROBLEM 10: Void Pointer Byte Shifting
    // Instruction: A void* has no sizing information. Cast 'raw_buffer' to a char*
    //              to shift it forward by exactly 14 bytes, then cast back to void* and store in 'res'.
    {
        char buffer[32] = "Restructuring_Workbook_Done";
        void* raw_buffer = static_cast<void*>(buffer);
        void* res = nullptr;
        
        // TODO: Shift raw_buffer forward by exactly 14 bytes and store in res
        
        bool ok = (res == static_cast<void*>(buffer + 14) && strcmp(static_cast<char*>(res), "Workbook_Done") == 0);
        reportStatus("Problem 10: Void pointer byte shifting", ok);
        if (ok) passed++; total++;
    }

    // PROBLEM 11: Struct Member Offset Calculations
    // Instruction: A PaddedStruct stores 'char x' and 'double y'. Due to memory alignment requirements,
    //              the compiler pads 'x' with 7 bytes so 'y' is aligned to an 8-byte boundary.
    //              Access the value of member 'y' using only pointer offsets from 'ptr_base' (cast to char* to step by bytes).
    {
        PaddedStruct s = {'A', 99.88};
        void* ptr_base = &s;
        double val = 0.0;
        
        // TODO: Access the double member 'y' using byte offsets from ptr_base
        
        bool ok = (std::abs(val - 99.88) < 1e-5);
        reportStatus("Problem 11: Padded struct offset resolution", ok);
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
