# Chapter 1.6: Tensor Permutations, Flattening & View Operations

Deep learning frameworks expose `transpose`, `permute`, `view`, `reshape`, and `flatten` as zero-cost operations. Understanding why they can be zero-cost — and when they cannot — requires understanding how strides turn memory layout into a multidimensional view.

This chapter is the bridge between raw pointer arithmetic and PyTorch/NumPy tensor semantics. It is directly relevant to understanding why `torch.permute()` can return a tensor that causes an uncoalesced GPU kernel, and how to fix it.

---

## 1. The Stride Representation: Memory Layout as Metadata

A tensor is not just a flat array. It is a pair: `(data_pointer, stride_tuple)`.

```text
Tensor A of shape [3, 4] stored in row-major:
  data_ptr: points to 12 floats in memory
  strides: (4, 1)     ← advance 4 elements to move along dim 0 (row), 1 to move along dim 1 (col)

To access A[i][j]: data_ptr[ i * 4 + j * 1 ]
```

This is the general formula: for an N-dimensional tensor with strides `(s_0, s_1, ..., s_{N-1})`:
```
flat_index = sum over all dimensions: index_d * stride_d
```

Any operation that only changes strides — without touching the data — is a **zero-copy view operation**.

---

## 2. Transpose: Swapping Strides

Transposing a matrix `A[R][C]` (shape R×C, strides (C, 1)) to `A^T[C][R]` simply swaps the stride tuple:

```text
Original A:       shape=(3,4), strides=(4, 1)
Transposed A^T:   shape=(4,3), strides=(1, 4)

A[i][j]   accesses:  data_ptr[ i*4 + j*1 ]
A^T[j][i] accesses:  data_ptr[ j*1 + i*4 ]  ← same data, same formula, different index meaning
```

No data is moved. The transposed tensor points to the **same physical memory** with different strides. This is why `torch.t()` and `torch.transpose()` are zero-copy.

```cpp
// Equivalent in C++ with explicit strides:
float* data = ...; // Original data, 12 floats, row-major [3][4]
int s0_orig = 4, s1_orig = 1;  // Row-major strides

// After transpose (logical [4][3]):
int s0_T = 1, s1_T = 4;       // Swapped!

// Access transposed element (j, i):
float val = data[j * s0_T + i * s1_T];   // = data[j*1 + i*4]
```

---

## 3. The Contiguity Problem: Why Transpose Kills Kernel Performance

After transposing, the tensor is **non-contiguous**: elements that are adjacent in the logical tensor are not adjacent in physical memory.

```text
A^T (logical view):    A^T[0][0]=A[0][0], A^T[0][1]=A[1][0], A^T[0][2]=A[2][0]
Physical addresses:     0x1000              0x1010              0x1020
(Stride = 4 * 4B = 16 bytes between logical neighbors)

A CUDA warp reading row 0 of A^T reads addresses: 0x1000, 0x1010, 0x1020...
→ Stride 16 bytes between each element → 4 sectors per warp → 75% bus waste!
```

PyTorch's `tensor.is_contiguous()` returns `False` after a transpose. Functions like `torch.mm` call `tensor.contiguous()` internally if needed, which triggers a **memory copy** to produce a new row-major buffer.

### Making a Non-Contiguous Tensor Contiguous

```cpp
// CPU-side equivalent: copy transposed data to new contiguous buffer
void make_contiguous_transpose(const float* src, float* dst,
                                int R, int C) {
    // src has shape [R][C], strides [C, 1]
    // dst will have shape [C][R], strides [R, 1] (true row-major for transposed shape)
    for (int j = 0; j < C; ++j) {
        for (int i = 0; i < R; ++i) {
            dst[j * R + i] = src[i * C + j];  // Physically copies the transposed data
        }
    }
}
```

On GPU, this is the matrix transpose kernel from Chapter 3.3 (with shared memory tiling to avoid uncoalesced reads).

---

## 4. Permute: Generalizing Transpose to N Dimensions

`torch.permute(dims)` reorders the dimensions of a tensor by reordering its stride tuple:

```text
Tensor Q shape [B, H, S, D] with strides [H*S*D, S*D, D, 1]:
After torch.permute(0, 2, 1, 3) → shape [B, S, H, D]:
  New strides: [H*S*D, D, S*D, 1]
  (Stride for dim 1 is now S*D — moved from dim 2, stride for dim 2 is now D — from dim 1)
```

This zero-copy permutation is how PyTorch attention switches between `[B, H, S, D]` (batch, heads first) and `[B, S, H, D]` (sequence first) views without memory movement.

```cpp
// Multi-dimensional stride access in CUDA kernel:
__global__ void permute_kernel(
    const float* src,   // Shape [B][H][S][D], strides [H*S*D, S*D, D, 1]
    float* dst,         // Shape [B][S][H][D], strides [S*H*D, H*D, D, 1]
    int B, int H, int S, int D)
{
    int b = blockIdx.z;
    int s = blockIdx.y;
    int h = blockIdx.x;
    int d = threadIdx.x;

    if (d < D) {
        // Source: [b][h][s][d] with original strides
        int src_idx = b*(H*S*D) + h*(S*D) + s*D + d;
        // Destination: [b][s][h][d] with new contiguous strides
        int dst_idx = b*(S*H*D) + s*(H*D) + h*D + d;
        dst[dst_idx] = src[src_idx];
    }
}
```

---

## 5. Reshape vs. View: When Zero-Copy Is Possible

### `view()` — Zero-Copy, Requires Contiguity

`torch.view(new_shape)` only changes the shape interpretation, keeping the same flat data. It is **zero-copy** but requires the tensor to be contiguous (strides must match what a fresh row-major allocation of the new shape would have).

```text
A shape [6] with strides [1]:
  view(2, 3) → shape [2][3], strides [3, 1]  ✓ contiguous → zero-copy allowed

A shape [4][3] with strides [1, 4]  (column-major, non-contiguous):
  view(12) → FAIL! Strides don't match a flat contiguous array
  → Must call .contiguous() first to get strides [3, 1], then view(12)
```

### `reshape()` — Zero-Copy When Possible, Copies When Necessary

`torch.reshape(new_shape)` is `view` with a fallback: if the tensor is contiguous, it is zero-copy; if not, it copies first.

```cpp
// C++ mental model:
// If tensor is contiguous:
//   reshape() == view() == just change the shape metadata
// If not contiguous:
//   reshape() == contiguous() + view() == memory copy + shape change
```

---

## 6. Flatten: Collapsing Dimensions

Flattening is a special case of reshape that reduces multiple dimensions to one:

```text
Tensor shape [B, C, H, W] with strides [C*H*W, H*W, W, 1]:
After flatten(1, 3) → shape [B, C*H*W] with strides [C*H*W, 1]
  (Only valid zero-copy if dims 1, 2, 3 are contiguous in memory)
```

For a CUDA kernel that processes batched image data as a 2D matrix of features:
```cpp
// Flatten [B, C, H, W] to [B, C*H*W] for linear layer input:
// If contiguous, just update the shape record (zero cost)
// Then: flat_index = batch * (C*H*W) + channel * (H*W) + row * W + col
```

---

## 7. The Contiguity Checklist

Before writing a CUDA kernel that expects a specific memory layout, validate:

```cpp
// PyTorch-side checks:
assert(tensor.is_contiguous());            // Ensure row-major, stride=(last_dim, ..., 1)
tensor = tensor.contiguous();              // Force copy if not contiguous

// In C++ CUDA code, check strides manually:
bool is_row_major = (strides[ndim-1] == 1);
for (int i = ndim-2; i >= 0; --i) {
    is_row_major = is_row_major && (strides[i] == strides[i+1] * shape[i+1]);
}
assert(is_row_major && "Kernel requires contiguous row-major tensor!");
```

---

## 8. Summary

| Operation | Memory Cost | Result |
| :--- | :--- | :--- |
| `transpose(i, j)` | Zero — swaps strides | Non-contiguous tensor |
| `permute(dims)` | Zero — reorders strides | Non-contiguous if dims reordered |
| `view(shape)` | Zero — changes shape only | Requires contiguous input |
| `reshape(shape)` | Zero if contiguous, copy if not | Always works |
| `flatten(start, end)` | Zero if inner dims contiguous | Collapses dimensions |
| `contiguous()` | Memory copy + new allocation | Forces row-major layout |
| `.is_contiguous()` | No cost (metadata check) | True if strides match row-major |
