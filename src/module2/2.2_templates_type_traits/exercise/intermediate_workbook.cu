// ==============================================================================
// Module 2.2: Templates & Type Traits — Intermediate Workbook
// ==============================================================================
// In this workbook, you will develop generic kernels and precision traits
// used in mixed-precision deep learning libraries:
// 1. Mixed-Precision Type Converter (float and half conversion traits)
// 2. Generic Block Reduction Kernel with Trait-Driven Accumulator
// 3. SFINAE-Guarded Vectorized Load Dispatch
// 4. Numerical Stability Epsilon Traits (EpsilonTraits<T>)
// 5. Matrix Accessor Class Template with Row/Col Major Specialization
//
// All exercises must print "Passed: X / Y tests." and return 0 on success.
// ==============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <type_traits>
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
// Exercise 1: Mixed-Precision Type Converter
// Implement TypeConverter<T> with static methods:
// to_float(T x) -> float
// from_float(float x) -> T
// Provide specializations for float and half (__half).
// ==============================================================================
template <typename T>
struct TypeConverter;

template <>
struct TypeConverter<float> {
    __host__ __device__ static float to_float(float x) { return x; }
    __host__ __device__ static float from_float(float x) { return x; }
};

template <>
struct TypeConverter<half> {
    // TODO:
    // Implement to_float using __half2float(x) and from_float using __float2half(x)
    __host__ __device__ static float to_float(half x) { return 0.0f; }
    __host__ __device__ static half from_float(float x) { return __float2half(0.0f); }
};

// ==============================================================================
// Exercise 2: Generic Block Reduction with Trait-Driven Accumulator
// Inputs of type T are accumulated into AccT = AccumulatorTraits<T>::Type.
// AccumulatorTraits<float>::Type = float, AccumulatorTraits<half>::Type = float.
// ==============================================================================
template <typename T>
struct AccumulatorTraits {
    using Type = float;
};

template <typename T>
__global__ void generic_reduction_kernel(const T* in, typename AccumulatorTraits<T>::Type* out, int N) {
    using AccT = typename AccumulatorTraits<T>::Type;
    __shared__ AccT s_data[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO:
    // 1. Convert in[gid] to float using TypeConverter<T>::to_float if in bounds, else 0.0f
    // 2. Load into s_data[tid], __syncthreads()
    // 3. Perform standard shared memory power-of-two reduction loop:
    //    for (int s = blockDim.x / 2; s > 0; s >>= 1) ...
    // 4. If tid == 0, atomicAdd to *out (using float atomicAdd)
}

// ==============================================================================
// Exercise 3: SFINAE-Guarded Vectorized Load Helper
// Only enable 128-bit vectorization if sizeof(T) * 4 == 16 (i.e. sizeof(T) == 4).
// Returns true if type is 4 bytes, false otherwise.
// ==============================================================================
template <typename T, typename std::enable_if_t<sizeof(T) == 4, int> = 0>
bool is_vectorizable_4x() {
    // TODO: Return true for 4-byte types
    return false;
}

template <typename T, typename std::enable_if_t<sizeof(T) != 4, int> = 0>
bool is_vectorizable_4x() {
    // TODO: Return false for non-4-byte types
    return false;
}

// ==============================================================================
// Exercise 4: Numerical Stability Epsilon Traits
// In normalization layers (RMSNorm, LayerNorm), adding machine-appropriate epsilon
// prevents division by zero: inv_rms = 1.0f / sqrt(mean_sq + eps).
// EpsilonTraits<float>::eps() = 1e-5f
// EpsilonTraits<half>::eps() = 1e-3f
// ==============================================================================
template <typename T>
struct EpsilonTraits;

template <>
struct EpsilonTraits<float> {
    __host__ __device__ static constexpr float eps() { return 1e-5f; }
};

template <>
struct EpsilonTraits<half> {
    // TODO: Return 1e-3f for half
    __host__ __device__ static constexpr float eps() { return 0.0f; }
};

template <typename T>
__host__ __device__ float compute_safe_inv_rms(float mean_sq) {
    // TODO:
    // return 1.0f / sqrtf(mean_sq + EpsilonTraits<T>::eps());
    return 0.0f;
}

// ==============================================================================
// Exercise 5: Matrix Accessor Class Template with Major-Order Specialization
// Access element (r, c) with dims (rows, cols):
// If RowMajor = true:  index = r * cols + c
// If RowMajor = false: index = c * rows + r (ColMajor)
// ==============================================================================
template <typename T, bool RowMajor>
struct MatrixAccessor {
    const T* data;
    int rows;
    int cols;

    __host__ __device__ MatrixAccessor(const T* d, int r, int c) : data(d), rows(r), cols(c) {}

    // TODO:
    // Implement operator()(int r, int c) const
    __host__ __device__ T operator()(int r, int c) const {
        return 0;
    }
};

// ==============================================================================
// Verification Test Harness
// ==============================================================================
int main() {
    int passed = 0;
    const int total_tests = 5;

    // --------------------------------------------------------------------------
    // Test 1: Half TypeConverter
    // --------------------------------------------------------------------------
    {
        half h = TypeConverter<half>::from_float(3.5f);
        float f = TypeConverter<half>::to_float(h);
        if (std::fabs(f - 3.5f) < 1e-3f) {
            std::cout << "[Test 1: Half TypeConverter] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 1: Half TypeConverter] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 2: Generic Block Reduction
    // --------------------------------------------------------------------------
    {
        int N = 256;
        std::vector<float> h_in(N, 1.5f);
        float h_out = 0.0f;

        float *d_in, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float)));

        generic_reduction_kernel<float><<<1, 256>>>(d_in, d_out, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        float expected = 256 * 1.5f; // 384.0f
        if (std::fabs(h_out - expected) < 1e-2f) {
            std::cout << "[Test 2: Generic Block Reduction] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 2: Generic Block Reduction] FAILED" << std::endl;
        }

        cudaFree(d_in); cudaFree(d_out);
    }

    // --------------------------------------------------------------------------
    // Test 3: SFINAE Vectorizable Check
    // --------------------------------------------------------------------------
    {
        bool f_vec = is_vectorizable_4x<float>();  // sizeof(float) == 4 -> true
        bool h_vec = is_vectorizable_4x<half>();   // sizeof(half) == 2 -> false
        bool d_vec = is_vectorizable_4x<double>(); // sizeof(double) == 8 -> false

        if (f_vec && !h_vec && !d_vec) {
            std::cout << "[Test 3: SFINAE Vectorizable Check] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 3: SFINAE Vectorizable Check] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 4: Numerical Stability Epsilon Traits
    // --------------------------------------------------------------------------
    {
        float inv_f = compute_safe_inv_rms<float>(0.0f); // 1 / sqrt(0 + 1e-5) = 316.2277f
        float inv_h = compute_safe_inv_rms<half>(0.0f);  // 1 / sqrt(0 + 1e-3) = 31.6227f

        if (std::fabs(inv_f - 316.2277f) < 1.0f &&
            std::fabs(inv_h - 31.6227f) < 0.5f) {
            std::cout << "[Test 4: Epsilon Traits] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 4: Epsilon Traits] FAILED" << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // Test 5: Matrix Accessor Specialization
    // --------------------------------------------------------------------------
    {
        int rows = 4, cols = 4;
        std::vector<int> mat(rows * cols);
        for (int i = 0; i < rows * cols; ++i) mat[i] = i;

        MatrixAccessor<int, true> row_mat(mat.data(), rows, cols);
        MatrixAccessor<int, false> col_mat(mat.data(), rows, cols);

        // For (r=1, c=2):
        // RowMajor: 1 * 4 + 2 = 6
        // ColMajor: 2 * 4 + 1 = 9
        if (row_mat(1, 2) == 6 && col_mat(1, 2) == 9) {
            std::cout << "[Test 5: Matrix Accessor Specialization] PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "[Test 5: Matrix Accessor Specialization] FAILED" << std::endl;
        }
    }

    std::cout << "Passed: " << passed << " / " << total_tests << " tests." << std::endl;
    return (passed == total_tests) ? 0 : 1;
}
