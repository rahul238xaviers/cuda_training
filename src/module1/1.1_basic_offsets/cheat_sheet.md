# Chapter 1.1: Memory Offsets & Pointer Arithmetic — Cheat Sheet

A concise reference of every technique, pattern, and mental model you built and validated in Chapter 1.1.

---

### 1. Pointer Scaling
- **Rule**: `ptr + N` jumps by `N * sizeof(*ptr)` bytes in physical memory.
- **Example**:
  - `uint8_t* p + 4` -> moves 4 bytes (4 * 1)
  - `float* p + 4`   -> moves 16 bytes (4 * 4)
  - `Float4* p + 4`  -> moves 64 bytes (4 * 16)
- **Key Takeaway**: To move by raw byte offsets, the pointer must be a 1-byte type (`uint8_t*` or `char*`).

---

### 2. Extracting Raw Pointer from Vector
- **Syntax**: `uint8_t* ptr = vec.data();`
- **Mental Model**: `std::vector` is a 24-byte stack manager object holding heap pointers. `.data()` extracts the raw C-style pointer to the start of the heap array.
- **Pointer to Pointer**: If the vector holds pointers `vector<void*> vec`, then `vec.data()` returns `void**` (a pointer to a pointer).

---

### 3. Interpreting Untyped Memory (`reinterpret_cast`)
- **Syntax**: `float* weights = reinterpret_cast<float*>(raw_bytes + 16);`
- **Mental Model**: RAM is just untyped bytes. Types are an interpretation lens. `reinterpret_cast` changes the lens without moving or modifying data.

---

### 4. In-Place Reversal (Two Pointers)
- **Pattern**:
  ```cpp
  float* left = arr;
  float* right = arr + (N - 1); // N - 1 because 0-indexed!
  while (left < right) {
      std::swap(*left, *right);
      left++;
      right--;
  }
  ```
- **Key Takeaway**: Last element is always `base + (N - 1)`. `base + N` is one-past-the-end out-of-bounds.

---

### 5. Circular Ring Buffer (Pointer Wrapping)
- **Pattern**:
  ```cpp
  *write_ptr = new_value;
  write_ptr++;
  if (write_ptr == buffer_base + CAPACITY) {
      write_ptr = buffer_base; // Wrap back to beginning
  }
  ```
- **Modulo Rule**: `index = item_counter % CAPACITY`. The output of `%` is already a 0-based index (never subtract 1).

---

### 6. Fast Bitwise Alignment (Snapping to Cache Line)
- **Formula**:
  ```cpp
  uintptr_t aligned_addr = (addr + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1);
  size_t padding_bytes = aligned_addr - addr;
  ```
- **Mental Model**:
  - `+ (ALIGNMENT - 1)`: Adds the maximum possible remainder. If already aligned, it stays in the block. If misaligned by even 1 byte, it crosses into the next block.
  - `& ~(ALIGNMENT - 1)`: The clear-bits mask erases the remainder bits, landing squarely on the boundary.

---

### 7. 2D Pitched Memory (`cudaMallocPitch`)
- **Concepts**:
  - `width_in_bytes`: Useful data payload per row (e.g. 32 floats = 128 bytes).
  - `PITCH`: Physical byte stride from row start to next row start (e.g. 256 bytes).
  - `padding`: `PITCH - width_in_bytes` (untouched slack bytes to keep rows aligned).
- **Accessing element `(r, c)`**:
  ```cpp
  float* row_ptr = reinterpret_cast<float*>(base_ptr + r * PITCH);
  float val = row_ptr[c];
  ```

---

### 8. 128-Bit Vectorized Stream Copy (`Float4`)
- **Struct**:
  ```cpp
  struct alignas(16) Float4 {
      float x, y, z, w; // 4 floats = 16 bytes = 128 bits
  };
  ```
- **Copy Pattern**:
  ```cpp
  const Float4* src_vec = reinterpret_cast<const Float4*>(src.data());
  Float4* dst_vec = reinterpret_cast<Float4*>(dst.data());
  for (int i = 0; i < N / 4; ++i) {
      dst_vec[i] = src_vec[i]; // 16 bytes copied in 1 CPU/GPU instruction
  }
  ```
- **Why**: 4x fewer load instructions, near 100% memory bus saturation (40 to 100+ GB/s).

---

### 9. Custom Aligned Bump Allocator
- **Pattern**:
  ```cpp
  uintptr_t aligned_addr = (curr_bump + (req.align - 1)) & ~(req.align - 1);
  if (aligned_addr + req.size <= pool_end) {
      void* allocated_ptr = reinterpret_cast<void*>(aligned_addr);
      curr_bump = aligned_addr + req.size;
  }
  ```
- **Why**: Pre-allocates a large pool once. Carves memory in sub-nanoseconds with zero OS `malloc`/`cudaMalloc` runtime syscalls.

---

### 10. Double-Buffered Pipeline (Pointer Ping-Pong)
- **Pattern**:
  ```cpp
  for (int i = 0; i < num_batches; ++i) {
      // 1. PCIe loads into transfer_buf:
      fill(transfer_buf);
      
      // 2. GPU computes on compute_buf (if i > 0):
      if (i > 0) compute(compute_buf);
      
      // 3. Zero-copy swap:
      std::swap(compute_buf, transfer_buf);
  }
  ```
- **Why**: Keeps GPU 100% saturated while PCIe transfers the next batch in parallel. Swapping pointers takes 1 clock cycle without copying data in RAM.

---

### 11. Tiled 2D Matrix Transpose
- **Why Tiling**: Naive transpose writes across columns, jumping by `DIM` floats every write and thrashing the cache. 16x16 tiles keep all reads and writes warm in L1 cache.
- **Explicit Frame-of-Reference Mapping**:
  ```cpp
  const int tiles_per_dim = DIM / TILE_SIZE;
  const int total_tiles = tiles_per_dim * tiles_per_dim;

  for (int tile_number = 0; tile_number < total_tiles; ++tile_number) {
      int global_tile_start_row = (tile_number / tiles_per_dim) * TILE_SIZE;
      int global_tile_start_col = (tile_number % tiles_per_dim) * TILE_SIZE;

      for (int current_row = global_tile_start_row; current_row < global_tile_start_row + TILE_SIZE; ++current_row) {
          for (int current_col = global_tile_start_col; current_col < global_tile_start_col + TILE_SIZE; ++current_col) {
              dst[current_col * DIM + current_row] = src[current_row * DIM + current_col];
          }
      }
  }
  ```
- **CUDA Equivalent**: `global_tile_start` maps to `blockIdx * TILE_SIZE`, inner loops map to `threadIdx`.
