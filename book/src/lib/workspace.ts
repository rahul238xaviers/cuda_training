import path from 'path';
import fs from 'fs';

export const WORKSPACE_ROOT = path.resolve(process.cwd(), '..');
export const SRC_DIR = path.join(WORKSPACE_ROOT, 'src');
export const MODULE1_DIR = path.join(SRC_DIR, 'module1');
export const PLAYGROUND_PATH = path.join(WORKSPACE_ROOT, 'playground.cpp');

export interface ChapterMeta {
  id: string;
  folder: string;
  title: string;
  completed: boolean;
  hasCheatSheet: boolean;
  tiers: {
    beginner: { status: 'passed' | 'pending'; tests: number };
    intermediate: { status: 'passed' | 'pending'; tests: number };
    champion: { status: 'passed' | 'pending'; tests: number };
  };
}

export interface MetalKernelMeta {
  id: number;
  name: string;
  category: string;
  metalFile: string;
  cudaFile: string;
  status: 'planned' | 'in_progress' | 'completed';
}

export const CHAPTERS_CONFIG: ChapterMeta[] = [
  { id: '1.1', folder: '1.1_basic_offsets', title: 'Basic Memory Offsets & Pointer Arithmetic', completed: true, hasCheatSheet: true, tiers: { beginner: { status: 'passed', tests: 4 }, intermediate: { status: 'passed', tests: 3 }, champion: { status: 'passed', tests: 3 } } },
  { id: '1.2', folder: '1.2_strides_indirection', title: 'Strides & Pointer Indirection', completed: false, hasCheatSheet: false, tiers: { beginner: { status: 'pending', tests: 0 }, intermediate: { status: 'pending', tests: 0 }, champion: { status: 'pending', tests: 0 } } },
  { id: '1.3', folder: '1.3_reductions_swaps', title: 'Reductions & Swaps', completed: false, hasCheatSheet: false, tiers: { beginner: { status: 'pending', tests: 0 }, intermediate: { status: 'pending', tests: 0 }, champion: { status: 'pending', tests: 0 } } },
  { id: '1.4', folder: '1.4_multi_array_strided', title: 'Multi-Array Strided Traversal', completed: false, hasCheatSheet: false, tiers: { beginner: { status: 'pending', tests: 0 }, intermediate: { status: 'pending', tests: 0 }, champion: { status: 'pending', tests: 0 } } },
  { id: '1.5', folder: '1.5_row_col_indexing', title: 'Row/Column Matrix Indexing', completed: false, hasCheatSheet: false, tiers: { beginner: { status: 'pending', tests: 0 }, intermediate: { status: 'pending', tests: 0 }, champion: { status: 'pending', tests: 0 } } },
  { id: '1.6', folder: '1.6_permute_flatten', title: 'Permute & Flatten Operations', completed: false, hasCheatSheet: false, tiers: { beginner: { status: 'pending', tests: 0 }, intermediate: { status: 'pending', tests: 0 }, champion: { status: 'pending', tests: 0 } } },
  { id: '1.7', folder: '1.7_subgrids_padding', title: 'Subgrids & Padding', completed: false, hasCheatSheet: false, tiers: { beginner: { status: 'pending', tests: 0 }, intermediate: { status: 'pending', tests: 0 }, champion: { status: 'pending', tests: 0 } } },
  { id: '1.8', folder: '1.8_pack_wrap_stencil', title: 'Packing, Wrapping & Stencils', completed: false, hasCheatSheet: false, tiers: { beginner: { status: 'pending', tests: 0 }, intermediate: { status: 'pending', tests: 0 }, champion: { status: 'pending', tests: 0 } } },
  { id: '1.9', folder: '1.9_single_array_alloc', title: 'Single Array Allocation Schemes', completed: false, hasCheatSheet: false, tiers: { beginner: { status: 'pending', tests: 0 }, intermediate: { status: 'pending', tests: 0 }, champion: { status: 'pending', tests: 0 } } },
  { id: '1.10', folder: '1.10_multidim_alignment', title: 'Multidimensional Alignment', completed: false, hasCheatSheet: false, tiers: { beginner: { status: 'pending', tests: 0 }, intermediate: { status: 'pending', tests: 0 }, champion: { status: 'pending', tests: 0 } } },
  { id: '1.11', folder: '1.11_arenas_placements', title: 'Arenas & Custom Placements', completed: false, hasCheatSheet: false, tiers: { beginner: { status: 'pending', tests: 0 }, intermediate: { status: 'pending', tests: 0 }, champion: { status: 'pending', tests: 0 } } },
  { id: '1.12', folder: '1.12_lifetimes_ownership', title: 'Lifetimes & Ownership Semantics', completed: false, hasCheatSheet: false, tiers: { beginner: { status: 'pending', tests: 0 }, intermediate: { status: 'pending', tests: 0 }, champion: { status: 'pending', tests: 0 } } },
  { id: '1.13', folder: '1.13_null_bounds_checks', title: 'Null & Bounds Verification', completed: false, hasCheatSheet: false, tiers: { beginner: { status: 'pending', tests: 0 }, intermediate: { status: 'pending', tests: 0 }, champion: { status: 'pending', tests: 0 } } },
  { id: '1.14', folder: '1.14_size_padding_punning', title: 'Size, Padding & Type Punning', completed: false, hasCheatSheet: false, tiers: { beginner: { status: 'pending', tests: 0 }, intermediate: { status: 'pending', tests: 0 }, champion: { status: 'pending', tests: 0 } } },
  { id: '1.15', folder: '1.15_casts_double_pointers', title: 'Casts & Double Pointers', completed: false, hasCheatSheet: false, tiers: { beginner: { status: 'pending', tests: 0 }, intermediate: { status: 'pending', tests: 0 }, champion: { status: 'pending', tests: 0 } } },
  { id: '1.16', folder: '1.16_pointer_relations_copying', title: 'Pointer Relations & Copying', completed: false, hasCheatSheet: false, tiers: { beginner: { status: 'pending', tests: 0 }, intermediate: { status: 'pending', tests: 0 }, champion: { status: 'pending', tests: 0 } } },
  { id: '1.17', folder: '1.17_embeddings_projections', title: 'Embeddings & Projections', completed: false, hasCheatSheet: false, tiers: { beginner: { status: 'pending', tests: 0 }, intermediate: { status: 'pending', tests: 0 }, champion: { status: 'pending', tests: 0 } } },
  { id: '1.18', folder: '1.18_attention_quantization', title: 'Attention & Quantization Packing', completed: false, hasCheatSheet: false, tiers: { beginner: { status: 'pending', tests: 0 }, intermediate: { status: 'pending', tests: 0 }, champion: { status: 'pending', tests: 0 } } },
  { id: '1.19', folder: '1.19_ml_layer_offsets', title: 'ML Layer Offsets & Layouts', completed: false, hasCheatSheet: false, tiers: { beginner: { status: 'pending', tests: 0 }, intermediate: { status: 'pending', tests: 0 }, champion: { status: 'pending', tests: 0 } } },
  { id: '1.20', folder: '1.20_pooling_masks_cycles', title: 'Pooling, Masks & Cyclic Buffers', completed: false, hasCheatSheet: false, tiers: { beginner: { status: 'pending', tests: 0 }, intermediate: { status: 'pending', tests: 0 }, champion: { status: 'pending', tests: 0 } } },
];

export const METAL_KERNELS: MetalKernelMeta[] = [
  { id: 1, name: 'Embedding Forward', category: 'Embedding', metalFile: 'embedding_forward.metal', cudaFile: 'embedding_forward.cu', status: 'planned' },
  { id: 2, name: 'RMSNorm Forward', category: 'Normalization', metalFile: 'rms_norm_forward.metal', cudaFile: 'rms_norm_forward.cu', status: 'planned' },
  { id: 3, name: 'RMSNorm Backward', category: 'Normalization', metalFile: 'rms_norm_backward.metal', cudaFile: 'rms_norm_backward.cu', status: 'planned' },
  { id: 4, name: 'RoPE Forward', category: 'Positional', metalFile: 'rope_forward.metal', cudaFile: 'rope_forward.cu', status: 'planned' },
  { id: 5, name: 'RoPE Backward', category: 'Positional', metalFile: 'rope_backward.metal', cudaFile: 'rope_backward.cu', status: 'planned' },
  { id: 6, name: 'FlashAttention Forward', category: 'Attention', metalFile: 'flash_attn_fwd.metal', cudaFile: 'flash_attn_fwd.cu', status: 'planned' },
  { id: 7, name: 'Fused Attention Backward', category: 'Attention', metalFile: 'fused_attn_bwd.metal', cudaFile: 'fused_attn_bwd.cu', status: 'planned' },
  { id: 8, name: 'GEMM BF16', category: 'Matrix Multiplication', metalFile: 'gemm_bf16.metal', cudaFile: 'gemm_bf16.cu', status: 'planned' },
  { id: 9, name: 'GEMM Projection', category: 'Matrix Multiplication', metalFile: 'gemm_proj.metal', cudaFile: 'gemm_proj.cu', status: 'planned' },
  { id: 10, name: 'GEMM Projection Trans B', category: 'Matrix Multiplication', metalFile: 'gemm_proj_trans_b.metal', cudaFile: 'gemm_proj_trans_b.cu', status: 'planned' },
  { id: 11, name: 'GEMM GQA', category: 'Matrix Multiplication', metalFile: 'gemm_gqa.metal', cudaFile: 'gemm_gqa.cu', status: 'planned' },
  { id: 12, name: 'GEMM FFN', category: 'Matrix Multiplication', metalFile: 'gemm_ffn.metal', cudaFile: 'gemm_ffn.cu', status: 'planned' },
  { id: 13, name: 'GEMM Backward', category: 'Matrix Multiplication', metalFile: 'gemm_backward.metal', cudaFile: 'gemm_backward.cu', status: 'planned' },
  { id: 14, name: 'SwiGLU Forward', category: 'Activation', metalFile: 'swiglu_forward.metal', cudaFile: 'swiglu_forward.cu', status: 'planned' },
  { id: 15, name: 'SwiGLU Backward', category: 'Activation', metalFile: 'swiglu_backward.metal', cudaFile: 'swiglu_backward.cu', status: 'planned' },
  { id: 16, name: 'Fused SwiGLU GEMM', category: 'Fused Operator', metalFile: 'fused_swiglu_gemm.metal', cudaFile: 'fused_swiglu_gemm.cu', status: 'planned' },
  { id: 17, name: 'Residual Add', category: 'Residual', metalFile: 'residual_add.metal', cudaFile: 'residual_add.cu', status: 'planned' },
  { id: 18, name: 'Fused Add Norm', category: 'Fused Operator', metalFile: 'fused_add_norm.metal', cudaFile: 'fused_add_norm.cu', status: 'planned' },
  { id: 19, name: 'Fused Backward Add Norm', category: 'Fused Operator', metalFile: 'fused_backward_add_norm.metal', cudaFile: 'fused_backward_add_norm.cu', status: 'planned' },
  { id: 20, name: 'Cross Entropy', category: 'Loss', metalFile: 'cross_entropy.metal', cudaFile: 'cross_entropy.cu', status: 'planned' },
  { id: 21, name: 'Compute Loss', category: 'Loss', metalFile: 'compute_loss.metal', cudaFile: 'compute_loss.cu', status: 'planned' },
  { id: 22, name: 'AdamW Optimizer Step', category: 'Optimizer', metalFile: 'adamw_step.metal', cudaFile: 'adamw_step.cu', status: 'planned' },
];

export function getChapterFolder(chapterId: string): string | null {
  const ch = CHAPTERS_CONFIG.find((c) => c.id === chapterId);
  return ch ? ch.folder : null;
}
