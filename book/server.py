#!/usr/bin/env python3
import http.server
import socketserver
import json
import os
import subprocess
import urllib.parse
from pathlib import Path

PORT = 8080
BASE_DIR = Path(__file__).resolve().parent.parent

class BookServer(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).resolve().parent / "static"), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/api/toc":
            self.send_json(self.get_toc())
        elif path == "/api/chapter":
            chapter_id = query.get("id", ["1.1"])[0]
            self.send_json(self.get_chapter_content(chapter_id))
        elif path == "/api/playground":
            pg_path = BASE_DIR / "playground.cpp"
            content = pg_path.read_text() if pg_path.exists() else ""
            self.send_json({"code": content})
        else:
            # Fall back to serving static files or index.html for SPA
            static_file = Path(__file__).resolve().parent / "static" / path.lstrip("/")
            if not static_file.exists() or static_file.is_dir():
                self.path = "/index.html"
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if path == "/api/run-playground":
            code = payload.get("code")
            if code is not None:
                (BASE_DIR / "playground.cpp").write_text(code)
            res = self.compile_and_run_playground()
            self.send_json(res)
        elif path == "/api/save-playground":
            code = payload.get("code", "")
            (BASE_DIR / "playground.cpp").write_text(code)
            self.send_json({"status": "ok"})
        elif path == "/api/run-workbook":
            chapter_id = payload.get("chapter_id", "1.1")
            tier = payload.get("tier", "beginner")
            res = self.compile_and_run_workbook(chapter_id, tier)
            self.send_json(res)
        elif path == "/api/ask-antigravity":
            prompt = payload.get("prompt", "")
            res = self.call_antigravity(prompt)
            self.send_json(res)
        else:
            self.send_response(404)
            self.end_headers()

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def get_toc(self):
        # Scan module1 directories
        mod1_dir = BASE_DIR / "src" / "module1"
        subtopics = [
            ("1.1", "1.1_basic_offsets", "Basic Memory Offsets & Pointer Arithmetic", True),
            ("1.2", "1.2_strides_indirection", "Strides & Pointer Indirection", False),
            ("1.3", "1.3_reductions_swaps", "Reductions & Swaps", False),
            ("1.4", "1.4_multi_array_strided", "Multi-Array Strided Traversal", False),
            ("1.5", "1.5_row_col_indexing", "Row/Column Matrix Indexing", False),
            ("1.6", "1.6_permute_flatten", "Permute & Flatten Operations", False),
            ("1.7", "1.7_subgrids_padding", "Subgrids & Padding", False),
            ("1.8", "1.8_pack_wrap_stencil", "Packing, Wrapping & Stencils", False),
            ("1.9", "1.9_single_array_alloc", "Single Array Allocation Schemes", False),
            ("1.10", "1.10_multidim_alignment", "Multidimensional Alignment", False),
            ("1.11", "1.11_arenas_placements", "Arenas & Custom Placements", False),
            ("1.12", "1.12_lifetimes_ownership", "Lifetimes & Ownership Semantics", False),
            ("1.13", "1.13_null_bounds_checks", "Null & Bounds Verification", False),
            ("1.14", "1.14_size_padding_punning", "Size, Padding & Type Punning", False),
            ("1.15", "1.15_casts_double_pointers", "Casts & Double Pointers", False),
            ("1.16", "1.16_pointer_relations_copying", "Pointer Relations & Copying", False),
            ("1.17", "1.17_embeddings_projections", "Embeddings & Projections", False),
            ("1.18", "1.18_attention_quantization", "Attention & Quantization Packing", False),
            ("1.19", "1.19_ml_layer_offsets", "ML Layer Offsets & Layouts", False),
            ("1.20", "1.20_pooling_masks_cycles", "Pooling, Masks & Cyclic Buffers", False)
        ]

        chapters = []
        for cid, folder, title, completed in subtopics:
            chapters.append({
                "id": cid,
                "folder": folder,
                "title": title,
                "completed": completed,
                "has_cheat_sheet": (mod1_dir / folder / "cheat_sheet.md").exists()
            })

        # List 22 Metal kernels for Phase 3 reference
        metal_kernels = [
            {"id": 1, "name": "Embedding Forward", "file": "embedding_forward.metal", "cuda": "embedding_forward.cu"},
            {"id": 2, "name": "RMSNorm Forward", "file": "rms_norm_forward.metal", "cuda": "rms_norm_forward.cu"},
            {"id": 3, "name": "RMSNorm Backward", "file": "rms_norm_backward.metal", "cuda": "rms_norm_backward.cu"},
            {"id": 4, "name": "RoPE Forward", "file": "rope_forward.metal", "cuda": "rope_forward.cu"},
            {"id": 5, "name": "RoPE Backward", "file": "rope_backward.metal", "cuda": "rope_backward.cu"},
            {"id": 6, "name": "FlashAttention Forward", "file": "flash_attn_fwd.metal", "cuda": "flash_attn_fwd.cu"},
            {"id": 7, "name": "Fused Attention Backward", "file": "fused_attn_bwd.metal", "cuda": "fused_attn_bwd.cu"},
            {"id": 8, "name": "GEMM BF16", "file": "gemm_bf16.metal", "cuda": "gemm_bf16.cu"},
            {"id": 9, "name": "GEMM Projection", "file": "gemm_proj.metal", "cuda": "gemm_proj.cu"},
            {"id": 10, "name": "GEMM Projection Trans B", "file": "gemm_proj_trans_b.metal", "cuda": "gemm_proj_trans_b.cu"},
            {"id": 11, "name": "GEMM GQA", "file": "gemm_gqa.metal", "cuda": "gemm_gqa.cu"},
            {"id": 12, "name": "GEMM FFN", "file": "gemm_ffn.metal", "cuda": "gemm_ffn.cu"},
            {"id": 13, "name": "GEMM Backward", "file": "gemm_backward.metal", "cuda": "gemm_backward.cu"},
            {"id": 14, "name": "SwiGLU Forward", "file": "swiglu_forward.metal", "cuda": "swiglu_forward.cu"},
            {"id": 15, "name": "SwiGLU Backward", "file": "swiglu_backward.metal", "cuda": "swiglu_backward.cu"},
            {"id": 16, "name": "Fused SwiGLU GEMM", "file": "fused_swiglu_gemm.metal", "cuda": "fused_swiglu_gemm.cu"},
            {"id": 17, "name": "Residual Add", "file": "residual_add.metal", "cuda": "residual_add.cu"},
            {"id": 18, "name": "Fused Add Norm", "file": "fused_add_norm.metal", "cuda": "fused_add_norm.cu"},
            {"id": 19, "name": "Fused Backward Add Norm", "file": "fused_backward_add_norm.metal", "cuda": "fused_backward_add_norm.cu"},
            {"id": 20, "name": "Cross Entropy", "file": "cross_entropy.metal", "cuda": "cross_entropy.cu"},
            {"id": 21, "name": "Compute Loss", "file": "compute_loss.metal", "cuda": "compute_loss.cu"},
            {"id": 22, "name": "AdamW Optimizer Step", "file": "adamw_step.metal", "cuda": "adamw_step.cu"}
        ]

        return {
            "current_bookmark": "1.2",
            "phase1_chapters": chapters,
            "phase3_kernels": metal_kernels,
            "stats": {
                "chapters_done": 1,
                "total_chapters": 20,
                "percent": 5,
                "exercises_passed": 10
            }
        }

    def get_chapter_content(self, chapter_id):
        # Map id to folder
        folder = None
        mod1_dir = BASE_DIR / "src" / "module1"
        for p in mod1_dir.iterdir():
            if p.is_dir() and p.name.startswith(chapter_id + "_"):
                folder = p
                break

        if not folder:
            return {"error": "Chapter not found"}

        theory_path = folder / "theory.md"
        cheat_path = folder / "cheat_sheet.md"

        theory = theory_path.read_text() if theory_path.exists() else "# Theory coming soon"
        cheat = cheat_path.read_text() if cheat_path.exists() else "# Cheat sheet coming soon"

        # Read beginner/interm/champion solution or exercise code
        beginner_code = (folder / "solution" / "beginner_workbook.cpp").read_text() if (folder / "solution" / "beginner_workbook.cpp").exists() else ""
        interm_code = (folder / "solution" / "intermediate_workbook.cpp").read_text() if (folder / "solution" / "intermediate_workbook.cpp").exists() else ""
        champion_code = (folder / "solution" / "champion_workbook.cpp").read_text() if (folder / "solution" / "champion_workbook.cpp").exists() else ""

        return {
            "id": chapter_id,
            "title": folder.name.replace("_", " ").title(),
            "theory": theory,
            "cheat_sheet": cheat,
            "workbooks": {
                "beginner": beginner_code,
                "intermediate": interm_code,
                "champion": champion_code
            }
        }

    def compile_and_run_playground(self):
        pg_bin = BASE_DIR / "playground_temp_bin"
        pg_src = BASE_DIR / "playground.cpp"
        compile_cmd = ["clang++", "-std=c++20", "-O3", str(pg_src), "-o", str(pg_bin)]

        comp = subprocess.run(compile_cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
        if comp.returncode != 0:
            return {
                "success": False,
                "stage": "compilation",
                "stdout": comp.stdout,
                "stderr": comp.stderr
            }

        # Run binary
        try:
            run = subprocess.run([str(pg_bin)], cwd=str(BASE_DIR), capture_output=True, text=True, timeout=5)
            if pg_bin.exists():
                pg_bin.unlink()
            return {
                "success": run.returncode == 0,
                "stage": "execution",
                "stdout": run.stdout,
                "stderr": run.stderr,
                "exit_code": run.returncode
            }
        except subprocess.TimeoutExpired:
            if pg_bin.exists():
                pg_bin.unlink()
            return {"success": False, "stage": "execution", "stdout": "", "stderr": "Execution timed out (5s limit)"}

    def compile_and_run_workbook(self, chapter_id, tier):
        folder = None
        mod1_dir = BASE_DIR / "src" / "module1"
        for p in mod1_dir.iterdir():
            if p.is_dir() and p.name.startswith(chapter_id + "_"):
                folder = p
                break

        if not folder:
            return {"success": False, "stderr": "Chapter directory not found"}

        target_file = folder / "solution" / f"{tier}_workbook.cpp"
        if not target_file.exists():
            target_file = folder / "exercise" / f"{tier}_workbook.cpp"

        bin_path = BASE_DIR / "wb_temp_bin"
        compile_cmd = ["clang++", "-std=c++20", "-O3", str(target_file), "-o", str(bin_path)]
        comp = subprocess.run(compile_cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
        if comp.returncode != 0:
            return {"success": False, "stage": "compilation", "stdout": comp.stdout, "stderr": comp.stderr}

        try:
            run = subprocess.run([str(bin_path)], cwd=str(BASE_DIR), capture_output=True, text=True, timeout=10)
            if bin_path.exists():
                bin_path.unlink()
            return {
                "success": run.returncode == 0,
                "stage": "execution",
                "stdout": run.stdout,
                "stderr": run.stderr,
                "exit_code": run.returncode
            }
        except subprocess.TimeoutExpired:
            if bin_path.exists():
                bin_path.unlink()
            return {"success": False, "stage": "execution", "stdout": "", "stderr": "Test timed out"}

    def call_antigravity(self, prompt):
        agy_path = "/Users/rahulkumar/.local/bin/agy"
        if not os.path.exists(agy_path):
            return {"success": False, "response": "Antigravity CLI (agy) not found at " + agy_path}

        # Run agy --print
        cmd = [agy_path, "--print", prompt]
        try:
            res = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, timeout=60)
            return {
                "success": res.returncode == 0,
                "response": res.stdout if res.returncode == 0 else res.stderr
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "response": "Antigravity response timed out (60s limit)"}

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), BookServer) as httpd:
        print(f"📖 Digital Book & Lab Server running at http://localhost:{PORT}")
        httpd.serve_forever()
