---
name: theory-blocks
description: >-
  Standard block formatting guidelines for authoring theory and textbook chapters
  in this repository. Enforces specialized formula cards, interactive diagrams,
  Markdown tables, and callouts over generic text blocks.
---

# Theory & Textbook Block Authoring Standards

When adding or updating content in `theory.md` or textbook markdown files, you MUST use the platform's native first-class block types instead of generic monospace `text` blocks.

## 1. Mathematical & Systems Formulas (`formula`)
Use the `formula` code fence for mathematical definitions, address calculations, hardware alignments, and algorithmic complexities. This renders inside a custom styled `.formula-card`.

```formula
Address(ptr + N) = Address(ptr) + N * sizeof(T)
Complexity(Partition) = O(V) = 32,000 operations
aligned_addr = (addr + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1)
```

**Rule**: Never write equations or ASCII math formulas in generic ```` ```text ```` blocks. Always use ```` ```formula ````.

## 2. Interactive SVG Diagrams (`diagram:<type>`)
The reader natively mounts interactive SVG diagrams when using the `diagram:<type>` fence followed by a JSON payload:

```diagram:two-pointer-partition
{
  "title": "In-Place Two-Pointer Threshold Partitioning (Top-P Logit Separation)",
  "subtitle": "Interactive Step-by-Step Pointers & Swapping across Pivot = 2.0"
}
```

Available diagram types registered in `TextbookDiagrams.tsx`:
- `diagram:two-pointer-partition`: Interactive two-pointer logit threshold partitioning.
- `diagram:tree-reduction`: Interactive CUDA halving stride and odd-length tail reduction.
- `diagram:pointer-swap`: Zero-copy 64-bit register pointer swap.
- `diagram:double-buffering`: Asynchronous PCIe host-to-device transfer and GPU compute overlap.
- `diagram:two-pass-reduction`: Two-pass grid-strided global memory reduction.
- `diagram:ring-allreduce`: Distributed ring-allreduce reduce-scatter and allgather stages.
- `diagram:stride-layout`: 2D logical row-major matrix folding into flat 1D memory.
- `diagram:pitched-memory`: Hardware memory alignment and `cudaMallocPitch` row padding.
- `diagram:cache-line`: 64-byte cache lines, boundary straddles, and bus transactions.

## 3. Data Pipelines & Comparisons (Markdown Tables)
Use structured Markdown tables whenever comparing stages, hardware components, or data types:

| Stage | Subsystem | Input Data | Output Data | Physical Representation |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Encode** | Tokenizer (CPU/Host) | String | Token IDs | Dense 32-bit integer array |
| **Stage 2: Compute** | Transformer SMs | Token IDs | Logits vector: `float[V]` | `V = 32,000` raw floats |

**Rule**: Do NOT use ASCII-art arrow chains in generic ```` ```text ```` blocks. Format them as structured tables.

## 4. Hardware Alerts & Callouts
Use GitHub-style alert callouts to emphasize critical hardware constraints and memory behaviors:

> [!IMPORTANT]
> In the logits vector, the **array index** is the integer Token ID, while the **array value** is the unnormalized floating-point score!

> [!NOTE]
> Auto-regressive token generation is 100% memory bandwidth bound because batch size is 1 per sequence.

> [!WARNING]
> Allocating a temporary vector inside the token loop triggers heap locks and evicts active KV-cache lines.

## 5. Programming Code (`cpp` / `cu`)
Always specify the exact language identifier (`cpp` or `cu`) so the IDE-grade syntax highlighter and line-number gutter are properly rendered:

```cpp
int left = 0;
int right = N - 1;
while (left <= right) {
    if (logits[left] >= PIVOT) {
        left++;
    } else if (logits[right] < PIVOT) {
        right--;
    } else {
        std::swap(logits[left], logits[right]);
        left++;
        right--;
    }
}
```

## Summary Checklist
- [ ] No generic ```` ```text ```` blocks for formulas or pipeline workflows.
- [ ] Formulas enclosed in ```` ```formula ````.
- [ ] Registered diagrams enclosed in ```` ```diagram:<name> ````.
- [ ] Tabular data and pipeline stages formatted in Markdown tables.
- [ ] Architectural constraints highlighted with `> [!IMPORTANT]` or `> [!NOTE]`.
- [ ] Plain text ASCII math only (NO LaTeX).
