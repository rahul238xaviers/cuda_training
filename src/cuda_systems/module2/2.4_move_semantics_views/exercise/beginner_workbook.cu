// ==============================================================================
// Module 2.4: Move Semantics & Views — Beginner Workbook
// ==============================================================================
// In this workbook, you will learn the mechanics of move semantics and
// non-owning tensor views essential for high-performance memory management:
// 1. Move Constructor for GPU Device Resource Wrapper
// 2. Move Assignment Operator for GPU Device Resource Wrapper
// 3. Perfect Forwarding Factory Function (std::forward)
// 4. Lightweight 1D SpanView Non-Owning Descriptor
// 5. 2D Row-Major Indexing Helper for View Operations
//
// All exercises must print "Passed: X / Y tests." and return 0 on success.
// ==============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <utility>
#include <cassert>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " code=" << err << " \"" << cudaGetErrorString(err)  \
                      << "\"" << std::endl;                                   \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// ==============================================================================
// Exercise 1: Move Constructor for MovableBuffer
// Transfers ownership of device pointer `data_` and `size_` from `other`.
// Resets `other.data_` to nullptr and `other.size_` to 0.
// ==============================================================================
template <typename T>
class MovableBuffer {
private:
    T* data_;
    size_t size_;

public:
    explicit MovableBuffer(size_t n) : data_(nullptr), size_(n) {
        if (size_ > 0) cudaMalloc(&data_, size_ * sizeof(T));
    }

    ~MovableBuffer() {
        if (data_) cudaFree(data_);
    }

    // Move Constructor
    MovableBuffer(MovableBuffer&& other) noexcept : data_(nullptr), size_(0) {
        // TODO:
        // 1. Take data_ and size_ from other
        // 2. Set other.data_ = nullptr, other.size_ = 0
    }

    // Move Assignment Operator (Used in Exercise 2)
    MovableBuffer& operator=(MovableBuffer&& other) noexcept {
        // TODO: Implemented in Exercise 2
        return *this;
    }

    T* data() const { return data_; }
    size_t size() const { return size_; }

    MovableBuffer(const MovableBuffer&) = delete;
    MovableBuffer& operator=(const MovableBuffer&) = delete;
};

// ==============================================================================
// Exercise 2: Move Assignment Operator for MovableBuffer
// If this != &other:
// 1. cudaFree current data_
// 2. data_ = other.data_; size_ = other.size_;
// 3. other.data_ = nullptr; other.size_ = 0;
// ==============================================================================
template <typename T>
void assign_moved(MovableBuffer<T>& dest, MovableBuffer<T>&& src) {
    // TODO:
    // dest = std::move(src);
}

// ==============================================================================
// Exercise 3: Perfect Forwarding Factory Helper
// Implement a function template that perfectly forwards arguments to construct
// a value in place:
// ==============================================================================
template <typename T, typename... Args>
T make_forwarded(Args&&... args) {
    // TODO:
    // return T(std::forward<Args>(args)...);
    return T();
}

struct Point {
    int x, y;
    Point() : x(0), y(0) {}
    Point(int a, int b) : x(a), y(b) {}
};

// ==============================================================================
// Exercise 4: Lightweight 1D SpanView Non-Owning Descriptor
// Contains a non-owning raw pointer and size.
// Operates on GPU or CPU memory without taking ownership.
// ==============================================================================
template <typename T>
struct SpanView {
    T* data;
    size_t size;

    __host__ __device__ SpanView(T* d, size_t s) : data(d), size(s) {}

    __host__ __device__ T& operator[](size_t i) {
        // TODO: Return data[i]
        return data[0];
    }
};

__global__ void fill_span_kernel(SpanView<float> span, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < span.size) {
        span[idx] = val;
    }
}

// ==============================================================================
// Exercise 5: 2D Row-Major Indexing Helper
// Given row r, col c, and total columns `cols`, compute linear offset.
// ==============================================================================
__host__ __device__ inline size_t compute_2d_offset(size_t r, size_t c, size_t cols) {
    // TODO:
    // return r * cols + c;
    return 0;
}

// ==============================================================================
// Verification Test Harness
// ==============================================================================
int main() {
    int passed = 0;
    const int total_tests = 5;

    // --------------------------------------------------------------------------
    // Test 1: Move Constructor
    // --------------------------------------------------------------------------
    {
        MovableBuffer<float> buf1(64);
        float* orig_ptr = buf1.data();

        MovableBuffer<float> buf2(std::move(buf1));

        if (buf2.data() == orig_ptr && buf2.size() == 64 &&
            buf1.data() == nullptr && buf1.size() == 0) {
            std::cout << "[Test 1: Move Constructor] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 1: Move Constructor] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 2: Move Assignment Operator
    // --------------------------------------------------------------------------
    {
        MovableBuffer<float> dest(16);
        MovableBuffer<float> src(32);
        float* src_ptr = src.data();

        dest = std::move(src);

        if (dest.data() == src_ptr && dest.size() == 32 &&
            src.data() == nullptr && src.size() == 0) {
            std::cout << "[Test 2: Move Assignment Operator] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 2: Move Assignment Operator] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 3: Perfect Forwarding
    // --------------------------------------------------------------------------
    {
        Point p = make_forwarded<Point>(10, 20);
        if (p.x == 10 && p.y == 20) {
            std::cout << "[Test 3: Perfect Forwarding] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 3: Perfect Forwarding] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 4: SpanView Kernel
    // --------------------------------------------------------------------------
    {
        int N = 32;
        float* d_data;
        CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_data, 0, N * sizeof(float)));

        SpanView<float> view(d_data, N);
        fill_span_kernel<<<1, 32>>>(view, 77.0f);
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<float> h_out(N);
        CHECK_CUDA(cudaMemcpy(h_out.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (std::fabs(h_out[i] - 77.0f) > 1e-4f) ok = false;
        }
        if (ok) {
            std::cout << "[Test 4: SpanView Kernel] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 4: SpanView Kernel] FAILED" << std::endl;
        }

        cudaFree(d_data);
    }

    // --------------------------------------------------------------------------
    // Test 5: 2D Row-Major Indexing
    // --------------------------------------------------------------------------
    {
        size_t off1 = compute_2d_offset(2, 3, 8); // 2 * 8 + 3 = 19
        size_t off2 = compute_2d_offset(0, 5, 8); // 0 * 8 + 5 = 5
        if (off1 == 19 && off2 == 5) {
            std::cout << "[Test 5: 2D Row-Major Offset] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 5: 2D Row-Major Offset] FAILED" << std::endl;
        }
    }

    std::cout << "Passed: " << passed << " / " << total_tests << " tests." << std::endl;
    return (passed == total_tests) ? 0 : 1;
}
