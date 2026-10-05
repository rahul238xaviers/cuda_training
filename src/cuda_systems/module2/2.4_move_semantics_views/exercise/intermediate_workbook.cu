// ==============================================================================
// Module 2.4: Move Semantics and Tensor Views
// Level: Intermediate Workbook
//
// Focus:
// 1. Move-only RAII DeviceTensor2D adhering to the Rule of 5.
// 2. Strided non-owning TensorView2D (rows, cols, row_stride, col_stride).
// 3. Zero-copy Subtensor Slicing: O(1) view derivation with adjusted offsets.
// 4. Zero-copy Transposition: Inverting strides without data movement.
// 5. 2D CUDA Kernel dispatch operating on strided TensorViews.
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
// Exercise 1: Move-Only DeviceTensor2D (Rule of 5)
// Represents an owning 2D device allocation stored in contiguous row-major memory.
// Non-copyable, movable.
// ==============================================================================
template <typename T>
class DeviceTensor2D {
private:
    T* data_{nullptr};
    size_t rows_{0};
    size_t cols_{0};

public:
    DeviceTensor2D() = default;

    DeviceTensor2D(size_t r, size_t c) : rows_(r), cols_(c) {
        if (rows_ > 0 && cols_ > 0) {
            CHECK_CUDA(cudaMalloc(&data_, rows_ * cols_ * sizeof(T)));
        }
    }

    ~DeviceTensor2D() {
        if (data_) {
            cudaFree(data_);
            data_ = nullptr;
        }
    }

    // Prohibit copy semantics
    DeviceTensor2D(const DeviceTensor2D&) = delete;
    DeviceTensor2D& operator=(const DeviceTensor2D&) = delete;

    // TODO: Exercise 1a: Implement Move Constructor
    // Transfer ownership of data_, rows_, cols_.
    // Nullify / zero-out other.
    DeviceTensor2D(DeviceTensor2D&& other) noexcept 
        : data_(nullptr), rows_(0), cols_(0) {
        // [YOUR CODE HERE]
    }

    // TODO: Exercise 1b: Implement Move Assignment Operator
    // If this != &other:
    // 1. Free existing data_
    // 2. Transfer data_, rows_, cols_
    // 3. Reset other
    DeviceTensor2D& operator=(DeviceTensor2D&& other) noexcept {
        // [YOUR CODE HERE]
        return *this;
    }

    T* data() const { return data_; }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t num_elements() const { return rows_ * cols_; }
};

// ==============================================================================
// Non-Owning Strided 2D Tensor View
// ==============================================================================
template <typename T>
struct TensorView2D {
    T* data{nullptr};
    size_t rows{0};
    size_t cols{0};
    size_t stride_row{0};
    size_t stride_col{0};

    __host__ __device__ TensorView2D() = default;

    __host__ __device__ TensorView2D(T* ptr, size_t r, size_t c, size_t s_r, size_t s_c)
        : data(ptr), rows(r), cols(c), stride_row(s_r), stride_col(s_c) {}

    // ==============================================================================
    // Exercise 2: Strided Element Accessor
    // Returns a reference to the element at (r, c) using strided addressing:
    //   offset = r * stride_row + c * stride_col
    // ==============================================================================
    __host__ __device__ T& operator()(size_t r, size_t c) {
        // TODO: Return reference to element at row r, column c
        // [YOUR CODE HERE]
        static T dummy{};
        return dummy;
    }

    __host__ __device__ const T& operator()(size_t r, size_t c) const {
        // TODO: Return const reference to element at row r, column c
        // [YOUR CODE HERE]
        static T dummy{};
        return dummy;
    }

    // ==============================================================================
    // Exercise 3: Zero-Copy Sub-Tensor Slicing
    // Returns a new TensorView2D representing a window of dimensions (sub_rows, sub_cols)
    // starting at (start_row, start_col).
    // The underlying pointer shifts to (start_row * stride_row + start_col * stride_col).
    // Strides remain the same.
    // ==============================================================================
    __host__ __device__ TensorView2D<T> slice(size_t start_r, size_t sub_rows,
                                              size_t start_c, size_t sub_cols) const {
        // TODO: Compute new data pointer and return sliced TensorView2D
        // [YOUR CODE HERE]
        return TensorView2D<T>();
    }

    // ==============================================================================
    // Exercise 4: Zero-Copy Transpose View
    // Returns a new TensorView2D representing the transposed matrix.
    // Transposition inverts dimensions (rows <-> cols) and swaps strides (stride_row <-> stride_col).
    // Zero GPU data copy is performed.
    // ==============================================================================
    __host__ __device__ TensorView2D<T> transpose() const {
        // TODO: Return transposed TensorView2D with swapped dimensions and swapped strides
        // [YOUR CODE HERE]
        return TensorView2D<T>();
    }
};

// ==============================================================================
// Exercise 5: 2D CUDA Kernel Operating on TensorView2D
// Given a TensorView2D, sets each element view(r, c) = val.
// Threads are organized in a 2D grid/block configuration.
// ==============================================================================
template <typename T>
__global__ void fill_view_2d_kernel(TensorView2D<T> view, T val) {
    // TODO:
    // 1. Compute 2D row index r and col index c from blockIdx and threadIdx
    // 2. If r < view.rows and c < view.cols, assign view(r, c) = val
    // [YOUR CODE HERE]
}

// ==============================================================================
// Verification Test Harness
// ==============================================================================
bool test_move_semantics() {
    DeviceTensor2D<float> tensor_a(128, 64);
    float* original_ptr = tensor_a.data();
    if (original_ptr == nullptr) return false;

    // Move construction
    DeviceTensor2D<float> tensor_b(std::move(tensor_a));
    if (tensor_b.data() != original_ptr || tensor_b.rows() != 128 || tensor_b.cols() != 64) {
        return false;
    }
    if (tensor_a.data() != nullptr || tensor_a.rows() != 0 || tensor_a.cols() != 0) {
        return false;
    }

    // Move assignment
    DeviceTensor2D<float> tensor_c(32, 16);
    tensor_c = std::move(tensor_b);
    if (tensor_c.data() != original_ptr || tensor_c.rows() != 128 || tensor_c.cols() != 64) {
        return false;
    }
    if (tensor_b.data() != nullptr || tensor_b.rows() != 0 || tensor_b.cols() != 0) {
        return false;
    }

    return true;
}

bool test_strided_access() {
    std::vector<int> host_data(16);
    // 4x4 matrix, row-major: stride_row = 4, stride_col = 1
    for (int i = 0; i < 16; ++i) host_data[i] = i * 10;

    TensorView2D<int> view(host_data.data(), 4, 4, 4, 1);
    if (view(0, 0) != 0) return false;
    if (view(1, 2) != 60) return false; // 1*4 + 2 = 6 -> 60
    if (view(3, 3) != 150) return false; // 3*4 + 3 = 15 -> 150

    view(2, 1) = 999;
    if (host_data[2 * 4 + 1] != 999) return false;

    return true;
}

bool test_subtensor_slicing() {
    // 6x6 matrix
    std::vector<int> matrix(36);
    for (int i = 0; i < 36; ++i) matrix[i] = i;

    TensorView2D<int> full_view(matrix.data(), 6, 6, 6, 1);

    // Slice 3x3 window starting at (row 2, col 1)
    TensorView2D<int> sub = full_view.slice(2, 3, 1, 3);
    if (sub.rows != 3 || sub.cols != 3) return false;
    if (sub.stride_row != 6 || sub.stride_col != 1) return false;

    // sub(0, 0) should point to matrix(2, 1) -> 2*6 + 1 = 13
    if (sub(0, 0) != 13) return false;
    // sub(1, 2) should point to matrix(3, 3) -> 3*6 + 3 = 21
    if (sub(1, 2) != 21) return false;

    sub(0, 0) = 777;
    if (matrix[13] != 777) return false;

    return true;
}

bool test_zero_copy_transpose() {
    // 3 rows, 4 columns
    std::vector<int> data(12);
    for (int i = 0; i < 12; ++i) data[i] = i;

    TensorView2D<int> mat(data.data(), 3, 4, 4, 1);
    TensorView2D<int> mat_t = mat.transpose();

    if (mat_t.rows != 4 || mat_t.cols != 3) return false;
    if (mat_t.stride_row != 1 || mat_t.stride_col != 4) return false;

    // In original mat: (r=1, c=2) is element 1*4 + 2 = 6
    // In mat_t: (r=2, c=1) should access that exact same element!
    if (mat_t(2, 1) != 6) return false;
    if (mat_t(3, 0) != 3) return false; // mat(0, 3) = 3

    return true;
}

bool test_cuda_2d_kernel() {
    const size_t R = 8, C = 8;
    DeviceTensor2D<float> d_tensor(R, C);
    if (d_tensor.data() == nullptr) return false;

    CHECK_CUDA(cudaMemset(d_tensor.data(), 0, R * C * sizeof(float)));

    // Full view
    TensorView2D<float> full_view(d_tensor.data(), R, C, C, 1);

    // Slice inner 4x4 region starting at (2, 2)
    TensorView2D<float> inner_sub = full_view.slice(2, 4, 2, 4);

    dim3 block(4, 4);
    dim3 grid(1, 1);
    fill_view_2d_kernel<<<grid, block>>>(inner_sub, 5.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_out(R * C);
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_tensor.data(), R * C * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify: only rows 2..5 and cols 2..5 should be 5.0f, everything else 0.0f
    for (size_t r = 0; r < R; ++r) {
        for (size_t c = 0; c < C; ++c) {
            float expected = (r >= 2 && r < 6 && c >= 2 && c < 6) ? 5.0f : 0.0f;
            if (h_out[r * C + c] != expected) {
                return false;
            }
        }
    }

    return true;
}

int main() {
    int passed = 0;

    if (test_move_semantics()) {
        std::cout << "[Test 1: Rule of 5 Move Semantics] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 1: Rule of 5 Move Semantics] FAILED\n";
    }

    if (test_strided_access()) {
        std::cout << "[Test 2: Strided TensorView2D Accessor] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 2: Strided TensorView2D Accessor] FAILED\n";
    }

    if (test_subtensor_slicing()) {
        std::cout << "[Test 3: Zero-Copy Sub-Tensor Slicing] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 3: Zero-Copy Sub-Tensor Slicing] FAILED\n";
    }

    if (test_zero_copy_transpose()) {
        std::cout << "[Test 4: Zero-Copy Transpose View] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 4: Zero-Copy Transpose View] FAILED\n";
    }

    if (test_cuda_2d_kernel()) {
        std::cout << "[Test 5: 2D CUDA Strided View Kernel] PASSED\n";
        passed++;
    } else {
        std::cout << "[Test 5: 2D CUDA Strided View Kernel] FAILED\n";
    }

    std::cout << "Passed: " << passed << " / 5 tests.\n";
    return (passed == 5) ? 0 : 1;
}
