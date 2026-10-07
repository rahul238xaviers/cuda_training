#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstring>
#include <cstdint>

using namespace std;

// =========================================================================
// BEGINNER WORKBOOK: Basic Offsets & Pointer Arithmetic
//
// Module: 1.1 - Memory Addressing & Pointer Foundations
// Level:  Beginner
//
// Focus: Practical pointer advancement, raw byte serialization, two-pointer
//        convergence, and circular ring buffer pointer arithmetic.
//
// Compilation:
//   g++ -std=c++20 -O3 exercise/beginner_workbook.cpp -o ../../../output/1.1_beginner
//   ../../../output/1.1_beginner
// =========================================================================

void reportStatus(const std::string& name, bool passed) {
    std::cout << "  " << std::left << std::setw(60) << name;
    if (passed) {
        std::cout << "\033[1;32m[PASSED]\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "--- WORKBOOK: Basic Offsets & Pointer Arithmetic (Beginner) ---" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int passed = 0;
    int total = 0;

    // -------------------------------------------------------------------------
    // PROBLEM 1: Strided Pointer Stepping & Sub-Vector Accumulation
    //
    // Context: In GPU kernel indexing, threads often traverse data with non-unit
    //          strides (e.g. strided reductions or channel-skipping).
    //
    // Given:
    //   - Array `data` of size N = 1024 floats.
    //   - Initial pointer `ptr = data.data() + offset` (offset = 4).
    //   - Upper boundary `end_ptr = data.data() + N`.
    //   - Traversal stride: `stride = 8`.
    //
    // Goal:
    //   - Traverse the array by advancing `ptr` in increments of `stride`.
    //   - Accumulate visited elements into `accumulated_sum`.
    //   - Count the total steps taken in `steps_taken`.
    //
    // Reference:
    //   - See Section 3 (Pointer Arithmetic & Scaling Laws) of theory.md.
    // -------------------------------------------------------------------------
    {
        const int N = 1024;
        std::vector<float> data(N);
        for (int i = 0; i < N; ++i) {
            data[i] = static_cast<float>(i * 0.5f);
        }

        const int offset = 4;
        const int stride = 8;
        float accumulated_sum = 0.0f;
        int steps_taken = 0;

        const float* ptr = data.data() + offset;
        const float* end_ptr = data.data() + N;

        // TODO: Traverse data with strided pointer stepping and accumulate visited elements.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        float expected_sum = 0.0f;
        int expected_steps = 0;
        for (int i = offset; i < N; i += stride) {
            expected_sum += data[i];
            expected_steps++;
        }

        bool p1_passed = (steps_taken == expected_steps) && (std::abs(accumulated_sum - expected_sum) < 1e-3f);
        reportStatus("Problem 1: Strided Pointer Stepping & Accumulation", p1_passed);
        if (p1_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 2: Binary Weight Blob Deserialization via Byte Offsets
    //
    // Context: Deep learning checkpoint formats (GGUF, Safetensors) store metadata
    //          and raw tensors in a contiguous binary byte blob.
    //          A parser reads header fields by incrementing raw `const uint8_t*` pointers.
    //
    // Given:
    //   - Raw byte buffer `blob` containing header metadata and tensor payload:
    //       Byte 0..3:   uint32_t magic number (0x54454E53, "TENS")
    //       Byte 4..7:   uint32_t num_elements (128)
    //       Byte 8..15:  uint64_t tensor_id (9876543210ULL)
    //       Byte 16..:   Array of 128 float elements (4 bytes each)
    //   - Base pointer `raw_ptr` pointing to the start of `blob`.
    //
    // Goal:
    //   - Read header metadata into `out_magic`, `out_num_elements`, and `out_tensor_id`.
    //   - Resolve pointer `out_tensor_data` to the start of the float tensor payload.
    //
    // Reference:
    //   - See Section 1 (Virtual Address Space & Addressing) and Section 3 (Pointer Arithmetic & Scaling Laws) of theory.md.
    // -------------------------------------------------------------------------
    {
        const uint32_t magic = 0x54454E53;
        const uint32_t num_elements = 128;
        const uint64_t tensor_id = 9876543210ULL;
        const size_t header_size = sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t); // 16 bytes
        const size_t total_bytes = header_size + num_elements * sizeof(float);

        std::vector<uint8_t> blob(total_bytes);
        uint8_t* p = blob.data();
        std::memcpy(p, &magic, sizeof(magic)); p += sizeof(magic);
        std::memcpy(p, &num_elements, sizeof(num_elements)); p += sizeof(num_elements);
        std::memcpy(p, &tensor_id, sizeof(tensor_id)); p += sizeof(tensor_id);
        for (uint32_t i = 0; i < num_elements; ++i) {
            float val = static_cast<float>(i * 1.5f);
            std::memcpy(p, &val, sizeof(float)); p += sizeof(float);
        }

        uint32_t out_magic = 0;
        uint32_t out_num_elements = 0;
        uint64_t out_tensor_id = 0;
        const float* out_tensor_data = nullptr;

        const uint8_t* raw_ptr = blob.data();

        // TODO: Deserialize header fields and resolve the float tensor pointer using byte offsets.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p2_passed = (out_magic == magic) &&
                         (out_num_elements == num_elements) &&
                         (out_tensor_id == tensor_id) &&
                         (out_tensor_data != nullptr) &&
                         (out_tensor_data[10] == 15.0f);

        reportStatus("Problem 2: Binary Weight Blob Deserialization", p2_passed);
        if (p2_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 3: In-Place Tensor Reversal via Two-Pointer Convergence
    //
    // Context: Flipping spatial dimensions (e.g. horizontal flip in data augmentation
    //          or backward pass convolution filter rotations) is done in-place.
    //
    // Given:
    //   - Array `arr` of size N = 100 floats.
    //   - Two pointers: `left` pointing to arr[0], and `right` pointing to arr[N - 1].
    //
    // Goal:
    //   - Reverse `arr` in-place using two-pointer convergence.
    //   - Zero auxiliary memory allocations allowed.
    //
    // Reference:
    //   - See Section 10 (Master Revision Cheat Sheet: Two-Pointer Reversal) of theory.md.
    // -------------------------------------------------------------------------
    {
        const int N = 100;
        std::vector<float> arr(N);
        for (int i = 0; i < N; ++i) arr[i] = static_cast<float>(i * 2 + 1);
        std::vector<float> original = arr;

        float* left = arr.data();
        float* right = arr.data() + N - 1;

        // TODO: Reverse the array in-place using two-pointer convergence.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p3_passed = true;
        for (int i = 0; i < N; ++i) {
            if (arr[i] != original[N - 1 - i]) {
                p3_passed = false;
                break;
            }
        }

        reportStatus("Problem 3: In-Place Reversal via Two-Pointer Convergence", p3_passed);
        if (p3_passed) passed++;
        total++;
    }

    // -------------------------------------------------------------------------
    // PROBLEM 4: Circular Ring Buffer Pointer Wrapping
    //
    // Context: Continuous batching schedulers in inference engines (vLLM, TensorRT-LLM)
    //          use circular ring buffers to queue incoming generation requests.
    //
    // Given:
    //   - Ring buffer `ring_buffer` of capacity CAPACITY = 16.
    //   - Base pointer `base_ptr = ring_buffer.data()`, and write pointer `head = base_ptr`.
    //   - Total insertions: TOTAL_INSERTS = 25 (elements 0 through 24).
    //
    // Goal:
    //   - Write elements 0 through 24 sequentially into `*head`.
    //   - Wrap `head` back to `base_ptr` whenever it reaches `base_ptr + CAPACITY`.
    //
    // Reference:
    //   - See Section 10 (Master Revision Cheat Sheet: Circular Ring Buffer) of theory.md.
    // -------------------------------------------------------------------------
    {
        const int CAPACITY = 16;
        std::vector<int> ring_buffer(CAPACITY, -1);
        int* const base_ptr = ring_buffer.data();
        int* head = base_ptr;

        const int TOTAL_INSERTS = 25;

        // TODO: Stream elements into the ring buffer, wrapping head back to base when reaching capacity.
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        bool p4_passed = true;
        // Last 16 elements inserted should be present in the ring buffer
        for (int i = 0; i < CAPACITY; ++i) {
            int expected_val = (TOTAL_INSERTS - CAPACITY) + ((i - (TOTAL_INSERTS % CAPACITY) + CAPACITY) % CAPACITY);
            // Verify buffer contains numbers from 9 to 24
            if (ring_buffer[i] < 9 || ring_buffer[i] > 24) {
                p4_passed = false;
            }
        }
        int expected_head_offset = TOTAL_INSERTS % CAPACITY; // 25 % 16 = 9
        if (head != base_ptr + expected_head_offset) p4_passed = false;

        reportStatus("Problem 4: Circular Ring Buffer Pointer Wrapping", p4_passed);
        if (p4_passed) passed++;
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
