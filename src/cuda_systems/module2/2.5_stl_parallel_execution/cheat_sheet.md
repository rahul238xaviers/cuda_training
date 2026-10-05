# Topic 2.5: Host Parallel Execution & CPU-GPU Overlap — Cheat Sheet

A concise reference for CPU-GPU co-design, double buffering, and host execution policies.

---

### 1. The Ping-Pong Double Buffering Model
- Keep GPU fully saturated while CPU prepares next batch:
  - Time Step 1: CPU prepares Batch 1 -> Copies to Device Buffer A -> GPU computes Batch 1.
  - Time Step 2: CPU prepares Batch 2 -> Copies to Device Buffer B while GPU computes Batch 1 concurrently.

---

### 2. CUDA Events for Host-Device Synchronization
- Measure precise GPU kernel duration without blocking CPU threads:
  ```cpp
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, stream);
  kernel<<<grid, block, 0, stream>>>(args...);
  cudaEventRecord(stop, stream);

  // CPU does other work here...
  cudaEventSynchronize(stop);
  float elapsed_ms;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  ```

---

### 3. LLM Systems Context
- **Token Sampling Pipeline**: GPU executes LM-Head GEMM + Softmax, writes top-k logits to host pinned buffer, while CPU executes greedy/nucleus sampling and token serialization in parallel.

---

### 4. Common Pitfalls
- Calling `cudaDeviceSynchronize()` inside inner training loops (stalls the entire GPU pipeline and drains all SM execution queues).
