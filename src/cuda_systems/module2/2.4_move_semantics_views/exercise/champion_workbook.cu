// ==============================================================================
// Module 2.4: Move Semantics and Tensor Views
// Level: Champion Workbook
//
// Focus:
// 1. High-Performance 3D/4D TensorView abstractions for Transformer architectures.
// 2. Zero-copy Token Window extraction for KV-caching and Sequence Attention.
// 3. Perfect-Forwarding CUDA Kernel Dispatcher with custom stream execution.
// 4. Stream-Aware Asynchronous Move Semantics ensuring GPU synchronization safety.
// 5. Zero-Copy Head Reshaping: [Batch, Seq, Hidden] -> [Batch, Seq, NumHeads, HeadDim].
// ==============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <utility>
#include <cassert>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl;  \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// ==============================================================================
// 2D Tensor View Definition
// ==============================================================================
template <typename T>
struct TensorView2D {
    T* data{nullptr};
    size_t rows{0};
    size_t cols{0};
    size_t stride_r{0};
    size_t stride_c{0};

    __host__ __device__ TensorView2D() = default;
    __host__ __device__ TensorView2D(T* p, size_t r, size_t c, size_t sr, size_t sc)
        : data(p), rows(r), cols(c), stride_r(sr), stride_c(sc) {}

    __host__ __device__ T& operator()(size_t r, size_t c) {
        return data[r * stride_r + c * stride_c];
    }
    __host__ __device__ const T& operator()(size_t r, size_t c) const {
        return data[r * stride_r + c * stride_c];
    }
};

// ==============================================================================
// Exercise 1: 3D Strided TensorView for Transformer Activations [Batch, Seq, Hidden]
// ==============================================================================
template <typename T>
struct TensorView3D {
    T* data{nullptr};
    size_t batch{0};
    size_t seq_len{0};
    size_t hidden_dim{0};
    size_t stride_b{0};
    size_t stride_s{0};
    size_t stride_h{0};

    __host__ __device__ TensorView3D() = default;
    __host__ __device__ TensorView3D(T* p, size_t b, size_t s, size_t h,
                                     size_t sb, size_t ss, size_t sh)
        : data(p), batch(b), seq_len(s), hidden_dim(h),
          stride_b(sb), stride_s(ss), stride_h(sh) {}

    // TODO: Exercise 1: Implement 3D coordinate element indexing:
    // offset = b * stride_b + s * stride_s + h * stride_h
    __host__ __device__ T& operator()(size_t b, size_t s, size_t h) {
        // [YOUR CODE HERE]
        static T dummy{};
        return dummy;
    }

    __host__ __device__ const T& operator()(size_t b, size_t s, size_t h) const {
        // [YOUR CODE HERE]
        static T dummy{};
        return dummy;
    }

    // ==============================================================================
    // Exercise 2: Zero-Copy Token Window Extraction
    // Extract a 2D slice for a single batch item `b_idx` and token range [token_start, token_start + num_tokens).
    // Returns a TensorView2D of dimensions (num_tokens, hidden_dim).
    // The underlying data pointer starts at:
    //   data + b_idx * stride_b + token_start * stride_s
    // Row stride is stride_s, col stride is stride_h.
    // ==============================================================================
    __host__ __device__ TensorView2D<T> slice_tokens(size_t b_idx, size_t token_start, size_t num_tokens) const {
        // TODO: Compute adjusted pointer and return TensorView2D
        // [YOUR CODE HERE]
        return TensorView2D<T>();
    }
};

// ==============================================================================
// Exercise 3: Perfect Forwarding CUDA Kernel Dispatcher
// Dispatches any CUDA global kernel with arbitrary arguments, preserving value
// categories using perfect forwarding across streams.
// ==============================================================================
template <typename KernelFunc, typename... Args>
void dispatch_kernel(cudaStream_t stream, dim3 grid, dim3 block, KernelFunc kernel, Args&&... args) {
    // TODO: Exercise 3: Launch kernel<<<grid, block, 0, stream>>>(std::forward<Args>(args)...);
    // [YOUR CODE HERE]
}

// Global kernel for scaling 2D view
template <typename T>
__global__ void scale_subview_kernel(TensorView2D<T> view, T scale) {
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < view.rows && c < view.cols) {
        view(r, c) = view(r, c) * scale;
    }
}

// ==============================================================================
// Exercise 4: Stream-Aware Asynchronous Move-Only Tensor
// Owns GPU device memory and a dedicated cudaStream_t.
// Adheres to move semantics while ensuring any pending stream work is synchronized
// before resource transfer to avoid race conditions.
// ==============================================================================
template <typename T>
class AsyncDeviceTensor {
private:
    T* data_{nullptr};
    size_t size_{0};
    cudaStream_t stream_{nullptr};

public:
    AsyncDeviceTensor() = default;

    AsyncDeviceTensor(size_t n) : size_(n) {
        CHECK_CUDA(cudaStreamCreate(&stream_));
        if (size_ > 0) {
            CHECK_CUDA(cudaMalloc(&data_, size_ * sizeof(T)));
        }
    }

    ~AsyncDeviceTensor() {
        if (data_) {
            if (stream_) cudaStreamSynchronize(stream_);
            cudaFree(data_);
            data_ = nullptr;
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    // Disable copy
    AsyncDeviceTensor(const AsyncDeviceTensor&) = delete;
    AsyncDeviceTensor& operator=(const AsyncDeviceTensor&) = delete;

    // TODO: Exercise 4a: Move Constructor with Stream Synchronization
    // 1. If other.stream_ is non-null, synchronize it (cudaStreamSynchronize)
    // 2. Transfer data_, size_, stream_ from other
    // 3. Nullify other.data_, other.size_, other.stream_
    AsyncDeviceTensor(AsyncDeviceTensor&& other) noexcept 
        : data_(nullptr), size_(0), stream_(nullptr) {
        // [YOUR CODE HERE]
    }

    // TODO: Exercise 4b: Move Assignment Operator with Stream Synchronization
    // If this != &other:
    // 1. Synchronize and clean up this existing stream_ and data_
    // 2. Synchronize other.stream_
    // 3. Transfer data_, size_, stream_
    // 4. Nullify other
    AsyncDeviceTensor& operator=(AsyncDeviceTensor&& other) noexcept {
        // [YOUR CODE HERE]
        return *this;
    }

    T* data() const { return data_; }
    size_t size() const { return size_; }
    cudaStream_t stream() const { return stream_; }
};

// ==============================================================================
// Exercise 5: Zero-Copy Head Reshaping [Batch, Seq, Hidden] -> [Batch, Seq, Heads, HeadDim]
// In Multi-Head Attention, Hidden = NumHeads * HeadDim.
// Transforms a 3D contiguous view into a 4D view without memory reallocation or copy.
// ==============================================================================
template <typename T>
struct TensorView4D {
    T* data{nullptr};
    size_t dim0{0}; // Batch
    size_t dim1{0}; // Seq
    size_t dim2{0}; // Heads
    size_t dim3{0}; // HeadDim

    size_t stride0{0};
    size_t stride1{0};
    size_t stride2{0};
    size_t stride3{0};

    __host__ __device__ TensorView4D() = default;
    __host__ __device__ TensorView4D(T* p, size_t d0, size_t d1, size_t d2, size_t d3,
                                     size_t s0, size_t s1, size_t s2, size_t s3)
        : data(p), dim0(d0), dim1(d1), dim2(d2), dim3(d3),
          stride0(s0), stride1(s1), stride2(s2), stride3(s3) {}

    __host__ __device__ T& operator()(size_t b, size_t s, size_t h, size_t d) {
        return data[b * stride0 + s * stride1 + h * stride2 + d * stride3];
    }
};

template <typename T>
TensorView4D<T> reshape_to_multihead(const TensorView3D<T>& view3d, size_t num_heads, size_t head_dim) {
    // TODO: Exercise 5:
    // Given view3d with dimensions [B, S, H] where H = num_heads * head_dim.
    // Construct and return TensorView4D with:
    //   dim0 = B, dim1 = S, dim2 = num_heads, dim3 = head_dim
    // Strides:
    //   stride3 = view3d.stride_h
    //   stride2 = head_dim * stride3
    //   stride1 = view3d.stride_s
    //   stride0 = view3d.stride_b
    // [YOUR CODE HERE]
    return TensorView4D<T>();
}

// ==============================================================================
// Verification Test Harness
// ==============================================================================
bool test_tensor_view_3d() {
    // 2 x 3 x 4 tensor = 24 elements
    std::vector<int> host_data(24);
    for (int i = 0; i < 24; ++i) host_data[i] = i;

    // Strides for row-major [2, 3, 4]:
    // stride_h = 1
    // stride_s = 4
    // stride_b = 3 * 4 = 12
    TensorView3D<int> view(host_data.data(), 2, 3, 4, 12, 4, 1);

    if (view(0, 0, 0) != 0) return false;
    if (view(0, 2, 3) != 11) return false; // 0*12 + 2*4 + 3 = 11
    if (view(1, 1, 2) != 18) return false; // 1*12 + 1*4 + 2 = 18

    view(1, 2, 3) = 999;
    if (host_data[23] != 999) return false;

    return true;
}

bool test_slice_tokens() {
    // 2 batches, 4 tokens, 8 hidden dimension = 64 elements
    std::vector<float> host_data(64);
    for (int i = 0; i < 64; ++i) host_data[i] = static_cast<float>(i);

    TensorView3D<float> view(host_data.data(), 2, 4, 8, 32, 8, 1);

    // Slice batch 1, tokens 1..2 (num_tokens = 2)
    TensorView2D<float> token_subview = view.slice_tokens(1, 1, 2);

    if (token_subview.rows != 2 || token_subview.cols != 8) return false;
    // Batch 1, token 1 starts at index 1*32 + 1*8 = 40
    if (token_subview(0, 0) != 40.0f) return false;
    // Batch 1, token 2, hidden dim 3 -> 1*32 + 2*8 + 3 = 51
    if (token_subview(1, 3) != 51.0f) return false;

    return true;
}

bool test_dispatch_kernel() {
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    float* d_ptr;
    const size_t R = 4, C = 4;
    CHECK_CUDA(cudaMalloc(&d_ptr, R * C * sizeof(float)));

    std::vector<float> h_init(R * C, 2.0f);
    CHECK_CUDA(cudaMemcpyAsync(d_ptr, h_init.data(), R * C * sizeof(float), cudaMemcpyHostToDevice, stream));

    TensorView2D<float> view(d_ptr, R, C, C, 1);

    dim3 block(2, 2);
    dim3 grid(2, 2);
    dispatch_kernel(stream, grid, block, scale_subview_kernel<float>, view, 3.0f);

    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<float> h_res(R * C);
    CHECK_CUDA(cudaMemcpy(h_res.data(), d_ptr, R * C * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < R * C; ++i) {
        if (h_res[i] != 6.0f) {
            cudaFree(d_ptr);
            cudaStreamDestroy(stream);
            return false;
        }
    }

    cudaFree(d_ptr);
    cudaStreamDestroy(stream);
    return true;
}

bool test_async_device_tensor() {
    AsyncDeviceTensor<float> t1(1024);
    float* p1 = t1.data();
    cudaStream_t s1 = t1.stream();
    if (p1 == nullptr || s1 == nullptr) return false;

    // Move construct
    AsyncDeviceTensor<float> t2(std::move(t1));
    if (t2.data() != p1 || t2.stream() != s1 || t2.size() != 1024) return false;
    if (t1.data() != nullptr || t1.stream() != nullptr || t1.size() != 0) return false;

    // Move assign
    AsyncDeviceTensor<float> t3(16);
    t3 = std::move(t2);
    if (t3.data() != p1 || t3.stream() != s1 || t3.size() != 1024) return false;
    if (t2.data() != nullptr || t2.stream() != nullptr || t2.size() != 0) return false;

    return true;
}

bool test_reshape_multihead() {
    // Batch=2, Seq=3, Hidden=8 (num_heads=2, head_dim=4)
    std::vector<int> data(48);
    for (int i = 0; i < 48; ++i) data[i] = i;

    TensorView3D<int> view3d(data.data(), 2, 3, 8, 24, 8, 1);
    TensorView4D<int> view4d = reshape_to_multihead(view3d, 2, 4);

    if (view4d.dim0 != 2 || view4d.dim1 != 3 || view4d.dim2 != 2 || view4d.dim3 != 4) return false;

    // Coordinate in 3D: b=1, s=2, h=5
    // In 4D: b=1, s=2, head=1 (h/4), dim=1 (h%4)
    // Offset should be identical: 1*24 + 2*8 + 5 = 45
    if (view4d(1, 2, 1, 1) != 45) return false;
    if (view4d(0, 1, 1, 2) != 14) return false; // 0*24 + 1*8 + (1*4 + 2) = 14

    return true;
}

int main() {
    int passed = 0;

    if (test_tensor_view_3d()) {
        std::cout << "[Test 1: TensorView3D Indexing] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 1: TensorView3D Indexing] FAILED\n";
    }

    if (test_slice_tokens()) {
        std::cout << "[Test 2: Zero-Copy Token Slicing] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 2: Zero-Copy Token Slicing] FAILED\n";
    }

    if (test_dispatch_kernel()) {
        std::cout << "[Test 3: Perfect-Forwarding Kernel Dispatcher] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 3: Perfect-Forwarding Kernel Dispatcher] FAILED\n";
    }

    if (test_async_device_tensor()) {
        std::cout << "[Test 4: Stream-Aware Move Semantics] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 4: Stream-Aware Move Semantics] FAILED\n";
    }

    if (test_reshape_multihead()) {
        std::cout << "[Test 5: Zero-Copy Multi-Head Reshaping] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 5: Zero-Copy Multi-Head Reshaping] FAILED\n";
    }

    std::cout << "Passed: " << passed << " / 5 tests.\n";
    return (passed == 5) ? 0 : 1;
}
