// ==============================================================================
// Module 2.3: Smart Pointers & RAII — Intermediate Workbook
// ==============================================================================
// In this workbook, you will implement production RAII resource abstractions:
// 1. CudaBuffer<T> Owning RAII Device Buffer
// 2. Multi-Stream Concurrent Execution using RAII Stream Guards
// 3. ScopedStream Guard for Context Switching
// 4. Cross-Stream Event Synchronization Barrier (cudaStreamWaitEvent)
// 5. Exception-Safe Multi-Buffer Composite Allocator
//
// All exercises must print "Passed: X / Y tests." and return 0 on success.
// ==============================================================================

#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__             \
                << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" \
                << std::endl;                                                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ==============================================================================
// Exercise 1: CudaBuffer<T> Owning RAII Device Buffer
// Encapsulates a device allocation of `count` elements of type T.
// - Allocates with cudaMalloc in constructor
// - Deallocates with cudaFree in destructor (only if ptr != nullptr)
// - Disable copy constructor and copy assignment operator
// - Provides get(), size(), and bytes() methods
// ==============================================================================
template <typename T> class CudaBuffer {
private:
  T *data_;
  size_t count_;

public:
  explicit CudaBuffer(size_t count) : data_(nullptr), count_(count) {
    // TODO: Allocate data_ with cudaMalloc
  }

  ~CudaBuffer() {
    // TODO: Deallocate data_ with cudaFree
  }

  T *get() const { return data_; }
  size_t size() const { return count_; }
  size_t bytes() const { return count_ * sizeof(T); }

  // Disable copying
  CudaBuffer(const CudaBuffer &) = delete;
  CudaBuffer &operator=(const CudaBuffer &) = delete;
};

// ==============================================================================
// Exercise 2: Multi-Stream Concurrent Execution using RAII
// Create a StreamGuard class that owns a cudaStream_t.
// Launch two independent operations into stream1 and stream2.
// ==============================================================================
class StreamGuard {
private:
  cudaStream_t stream_;

public:
  StreamGuard() {
    // TODO: cudaStreamCreate(&stream_);
    stream_ = nullptr;
  }

  ~StreamGuard() {
    // TODO: if (stream_) { cudaStreamSynchronize(stream_);
    // cudaStreamDestroy(stream_); }
  }

  cudaStream_t get() const { return stream_; }

  StreamGuard(const StreamGuard &) = delete;
  StreamGuard &operator=(const StreamGuard &) = delete;
};

__global__ void inc_kernel(float *data, float val, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    data[idx] += val;
}

// ==============================================================================
// Exercise 3: ScopedStream Guard for Context Switching
// Sets a target stream as active for the scope of the guard.
// ==============================================================================
class ScopedStream {
private:
  cudaStream_t stream_;

public:
  explicit ScopedStream(cudaStream_t s) : stream_(s) {}
  cudaStream_t get() const { return stream_; }
};

// ==============================================================================
// Exercise 4: Cross-Stream Event Synchronization Barrier
// Record an event in producer_stream, then make consumer_stream wait on it
// using cudaStreamWaitEvent before executing consumer kernel.
// ==============================================================================
class EventBarrier {
private:
  cudaEvent_t event_;

public:
  EventBarrier() {
    // TODO: cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    event_ = nullptr;
  }

  ~EventBarrier() {
    // TODO: if (event_) cudaEventDestroy(event_);
  }

  void record(cudaStream_t producer) {
    // TODO: cudaEventRecord(event_, producer);
  }

  void wait(cudaStream_t consumer) {
    // TODO: cudaStreamWaitEvent(consumer, event_, 0);
  }
};

// ==============================================================================
// Exercise 5: Exception-Safe Multi-Buffer Composite Allocator
// Allocates two buffers (e.g. weights and biases).
// If any allocation fails or size is 0, cleans up any previously allocated
// buffer.
// ==============================================================================
class DualBufferManager {
public:
  std::unique_ptr<CudaBuffer<float>> buf_a;
  std::unique_ptr<CudaBuffer<float>> buf_b;

  DualBufferManager(size_t size_a, size_t size_b) {
    // TODO:
    // 1. Allocate buf_a = std::make_unique<CudaBuffer<float>>(size_a);
    // 2. Allocate buf_b = std::make_unique<CudaBuffer<float>>(size_b);
  }
};

// ==============================================================================
// Verification Test Harness
// ==============================================================================
int main() {
  int passed = 0;
  const int total_tests = 5;

  // --------------------------------------------------------------------------
  // Test 1: CudaBuffer<T>
  // --------------------------------------------------------------------------
  {
    CudaBuffer<float> buf(64);
    if (buf.get() != nullptr && buf.size() == 64 &&
        buf.bytes() == 64 * sizeof(float)) {
      CHECK_CUDA(cudaMemset(buf.get(), 0, buf.bytes()));
      std::cout << "[Test 1: CudaBuffer RAII] PASSED" << std::endl;
      passed++;
    } else {
      std::cout << "[Test 1: CudaBuffer RAII] FAILED" << std::endl;
    }
  }

  // --------------------------------------------------------------------------
  // Test 2: Multi-Stream Concurrent Execution
  // --------------------------------------------------------------------------
  {
    CudaBuffer<float> buf1(128);
    CudaBuffer<float> buf2(128);
    if (buf1.get() != nullptr && buf2.get() != nullptr) {
      CHECK_CUDA(cudaMemset(buf1.get(), 0, buf1.bytes()));
      CHECK_CUDA(cudaMemset(buf2.get(), 0, buf2.bytes()));

      {
        StreamGuard s1;
        StreamGuard s2;
        if (s1.get() != nullptr && s2.get() != nullptr) {
          inc_kernel<<<1, 128, 0, s1.get()>>>(buf1.get(), 10.0f, 128);
          inc_kernel<<<1, 128, 0, s2.get()>>>(buf2.get(), 20.0f, 128);
        }
      } // Both streams synchronized on destruction

      std::vector<float> h1(128), h2(128);
      CHECK_CUDA(cudaMemcpy(h1.data(), buf1.get(), buf1.bytes(),
                            cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(h2.data(), buf2.get(), buf2.bytes(),
                            cudaMemcpyDeviceToHost));

      if (h1[0] == 10.0f && h2[0] == 20.0f) {
        std::cout << "[Test 2: Multi-Stream Execution] PASSED" << std::endl;
        passed++;
      } else {
        std::cout << "[Test 2: Multi-Stream Execution] FAILED" << std::endl;
      }
    } else {
      std::cout << "[Test 2: Multi-Stream Execution] FAILED" << std::endl;
    }
  }

  // --------------------------------------------------------------------------
  // Test 3: ScopedStream
  // --------------------------------------------------------------------------
  {
    StreamGuard guard;
    if (guard.get() != nullptr) {
      ScopedStream scoped(guard.get());
      if (scoped.get() == guard.get()) {
        std::cout << "[Test 3: ScopedStream] PASSED" << std::endl;
        passed++;
      } else {
        std::cout << "[Test 3: ScopedStream] FAILED" << std::endl;
      }
    } else {
      std::cout << "[Test 3: ScopedStream] FAILED" << std::endl;
    }
  }

  // --------------------------------------------------------------------------
  // Test 4: Cross-Stream EventBarrier
  // --------------------------------------------------------------------------
  {
    StreamGuard s1;
    StreamGuard s2;
    EventBarrier barrier;
    CudaBuffer<float> buf(64);
    if (buf.get() != nullptr && s1.get() != nullptr && s2.get() != nullptr) {
      CHECK_CUDA(cudaMemset(buf.get(), 0, buf.bytes()));

      // Producer in s1
      inc_kernel<<<1, 64, 0, s1.get()>>>(buf.get(), 5.0f, 64);
      barrier.record(s1.get());

      // Consumer in s2 waits on s1's completion
      barrier.wait(s2.get());
      inc_kernel<<<1, 64, 0, s2.get()>>>(buf.get(), 10.0f, 64);

      CHECK_CUDA(cudaStreamSynchronize(s2.get()));

      std::vector<float> h(64);
      CHECK_CUDA(
          cudaMemcpy(h.data(), buf.get(), buf.bytes(), cudaMemcpyDeviceToHost));
      if (h[0] == 15.0f) {
        std::cout << "[Test 4: Cross-Stream EventBarrier] PASSED" << std::endl;
        passed++;
      } else {
        std::cout << "[Test 4: Cross-Stream EventBarrier] FAILED" << std::endl;
      }
    } else {
      std::cout << "[Test 4: Cross-Stream EventBarrier] FAILED" << std::endl;
    }
  }

  // --------------------------------------------------------------------------
  // Test 5: DualBufferManager Composite Allocator
  // --------------------------------------------------------------------------
  {
    DualBufferManager mgr(64, 128);
    if (mgr.buf_a != nullptr && mgr.buf_b != nullptr &&
        mgr.buf_a->size() == 64 && mgr.buf_b->size() == 128 &&
        mgr.buf_a->get() != nullptr && mgr.buf_b->get() != nullptr) {
      std::cout << "[Test 5: DualBufferManager Composite] PASSED" << std::endl;
      passed++;
    } else {
      std::cout << "[Test 5: DualBufferManager Composite] FAILED" << std::endl;
    }
  }

  std::cout << "Passed: " << passed << " / " << total_tests << " tests."
            << std::endl;
  return (passed == total_tests) ? 0 : 1;
}
