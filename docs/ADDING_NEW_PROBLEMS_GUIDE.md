# Guide: Adding New Problems to Workbooks

This guide documents the exact protocol required whenever an instructor, developer, or AI agent adds a new problem to an existing or new workbook.

Following this standard ensures that:
1. **The Book App** tracks the problem deterministically in the local SQLite database.
2. **Student Solutions** automatically receive the new problem stubs without overwriting their existing solved code.
3. **Sidebar & Mastery Bars** display accurate completion counts and dynamic badges (e.g. `+1 New`).
4. **The Lab Editor** renders interactive problem pills with auto-jump navigation.

---

## The 4-Step Standard Workflow

Whenever a new problem is added, follow these four steps in order:

```
[Step 1] Add problem stub to exercise/<tier>_workbook.cpp
   │
[Step 2] Register problem in book/src/config/problems.json
   │
[Step 3] Implement reference code in solution/<tier>_workbook.cpp
   │
[Step 4] Validate build & test execution
```

---

### Step 1: Write the Exercise Stub in `exercise/`

Open the target workbook:
* Volume 1: `src/cpp_systems/<chapter_folder>/exercise/<tier>_workbook.cpp`
* Volume 2: `src/cuda_systems/<module_folder>/<chapter_folder>/exercise/<tier>_workbook.cu`
* Volume 3: `src/gpu_kernels/<module_folder>/<kernel_folder>/exercise/<tier>_workbook.cu`

Place the new problem block in sequential order (e.g., if Problems 1 to 3 exist, add Problem 4 before the scorecard block):

```cpp
    // -------------------------------------------------------------------------
    // PROBLEM <N>: <Concise Problem Title>
    //
    // Context: Architectural or systems motivation (why this matters for GPU/LLM).
    // Task: Clear, concrete instructions on inputs, outputs, and constraints.
    // -------------------------------------------------------------------------
    {
        // 1. Setup test fixtures and synthetic data
        const int N = 128;
        std::vector<float> input_tensor(N, 1.0f);
        std::vector<float> output_tensor(N, 0.0f);

        // TODO: Brief instruction for the student
        // --- YOUR CODE STARTS HERE ---

        // --- YOUR CODE ENDS HERE ---

        // 2. Validate correctness
        bool p<N>_passed = true;
        for (int i = 0; i < N; ++i) {
            if (output_tensor[i] != input_tensor[i]) {
                p<N>_passed = false;
                break;
            }
        }

        // 3. Report status and update scorecard counters
        reportStatus("Problem <N>: <Concise Problem Title>", p<N>_passed);
        if (p<N>_passed) passed++;
        total++;
    }
```

#### Formatting Rules for Problem Blocks:
1. **Header format**: Must start with `// PROBLEM <N>: <Title>` (or `// Problem <N>: <Title>`).
2. **Marker comments**: Must enclose the student code area with:
   `// --- YOUR CODE STARTS HERE ---`
   `// --- YOUR CODE ENDS HERE ---`
3. **Status call**: Use `reportStatus("Problem <N>: <Title>", p<N>_passed)`.
4. **Counter update**: Always include `if (p<N>_passed) passed++; total++;`.

---

### Step 2: Register in Declarative Manifest (`problems.json`)

Open [book/src/config/problems.json](book/src/config/problems.json).

Add or update the chapter entry under `chapters.<chapterId>.<tier>`:

```json
{
  "chapters": {
    "1.2": {
      "champion": [
        {
          "problem_num": 1,
          "title": "CSR Matrix Compression (Dense -> CSR Packing)",
          "label": "Problem 1: CSR Matrix Compression (Dense -> CSR Packing)"
        },
        ...
        {
          "problem_num": 4,
          "title": "Generalized 3D Tensor Stride Permutation",
          "label": "Problem 4: Generalized 3D Tensor Stride Permutation"
        }
      ]
    }
  }
}
```

#### Field Specifications:
* `problem_num`: Integer index matching the problem block (e.g. `4`).
* `title`: The human-readable title displayed on the editor checklist pill.
* `label`: The exact string printed by `reportStatus(...)` so the test runner matches it unambiguously.

---

### Step 3: Implement the Solution in `solution/`

Open the matching workbook in `solution/`:
* Volume 1: `src/cpp_systems/<chapter_folder>/solution/<tier>_workbook.cpp`
* Volume 2: `src/cuda_systems/<module_folder>/<chapter_folder>/solution/<tier>_workbook.cu`
* Volume 3: `src/gpu_kernels/<module_folder>/<kernel_folder>/solution/<tier>_workbook.cu`

Add the working reference solution inside the problem block:

```cpp
    // --- YOUR CODE STARTS HERE ---
    for (int i = 0; i < N; ++i) {
        output_tensor[i] = input_tensor[i];
    }
    // --- YOUR CODE ENDS HERE ---
```

#### How Automated Synchronization Protects Existing Students:
* If a student already has a solved `solution/` file on disk from an earlier repository state, the app detects that Problem `<N>` is missing from their solution.
* The auto-sync engine creates a safety backup (`<tier>_workbook.cpp.sync_<timestamp>.bak`) and automatically appends the new problem stub into their solution file right above the scorecard.
* Existing solutions written by the student are **never overwritten or modified**.

---

### Step 4: Validate Build and Test Runner

Compile and run the updated workbook to ensure zero compiler warnings and clean execution:

```bash
# For Volume 1 C++ Workbooks:
clang++ -std=c++20 -O3 src/cpp_systems/<chapter>/solution/<tier>_workbook.cpp -o output/workbook_test && ./output/workbook_test

# For Volume 2 & 3 CUDA Workbooks (on NVIDIA machine):
nvcc -O3 -std=c++17 src/cuda_systems/<module>/<chapter>/solution/<tier>_workbook.cu -o output/cuda_test && ./output/cuda_test
```

Verify in the web app (`http://localhost:3000`):
1. The sidebar reflects the new problem count (e.g. `[+1 New]` or updated `passed/total`).
2. The Problem Checklist Bar above Monaco displays the new problem pill (`P<N>`).
3. Clicking the pill jumps directly to `// PROBLEM <N>`.

---

## Agent Invariants (Mandatory for AI Assistants)

When an AI agent is instructed to add or modify exercises:
1. **Never commit code to git** unless explicitly instructed by the user.
2. **Never modify existing solved problems** in a user's solution file.
3. **Always register in `problems.json`** so that the database and UI stay in sync with the file system.
4. **Always ensure `reportStatus` strings** match the manifest `label` exactly.
