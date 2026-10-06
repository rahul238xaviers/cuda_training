# Industry-Grade CUDA Kernel Projects: Modality & Lifecycle Architecture

## 1. Vision & Taxonomy
In production high-performance computing (NVIDIA TensorRT-LLM, vLLM, DeepSeek, FlashAttention), GPU kernels are specialized along two orthogonal axes:
1. **Modalities**: **Text (LLMs & Reasoning)**, **Audio (Speech & Acoustics)**, and **Vision/Video (Generative Multimodal)**.
2. **Lifecycle Stages**: **Pre-Training (High Throughput & Backward Gradients)**, **Post-Training & Fine-Tuning (LoRA, DPO, Quantization)**, and **Inference Serving (Low Latency, Memory Bandwidth Bound)**.

This curriculum organizes the kernel projects into distinct **Domain Tracks**, allowing an engineer to dive deep into text-based LLMs, audio processing, or video/vision from raw data ingestion to production deployment.

---

## 2. The Comprehensive Modality & Lifecycle Matrix

| Lifecycle Stage | Text / LLM Track | Audio / Speech Track | Vision & Video Track |
| :--- | :--- | :--- | :--- |
| **Pre-Training** *(Compute-Bound, Backward Gradients, Distributed)* | • Fused Cross-Entropy Loss<br/>• FlashAttention-3 Fwd+Bwd<br/>• Distributed All-Reduce<br/>• Fused AdamW Optimizer | • 2D STFT / Mel-Spectrogram<br/>• CTC Loss with Dynamic Programming<br/>• Multi-Channel Cross-Attention | • 3D Patchification & Tiling<br/>• Spatio-Temporal 3D Attention<br/>• Conv3D / Depthwise Separable |
| **Post-Training** *(Fine-Tuning, Alignment & Calibration)* | • Fused LoRA (W + B*A GEMM)<br/>• DPO / PPO Preference Loss<br/>• FP8 Weight Calibration | • Audio Adapter Slicing<br/>• Latent Audio VAE Loss<br/>• Contrastive Audio-Text (CLAP) | • Vision LoRA Adapters<br/>• Latent Diffusion Flow Matching<br/>• Perceptual & Structural Loss |
| **Inference Serving** *(Memory-Bound, Real-Time Latency, Caching)* | • PagedAttention & KV-Cache<br/>• W4A16 / FP8 Dequant-GEMV<br/>• Speculative Decoding Verifier<br/>• Top-K / Top-P Logit Sampler | • Streaming Causal Conv1D<br/>• EnCodec / Audio Tokenizer<br/>• Real-Time Beam Search Decoder | • VAE Latent Decoder Kernel<br/>• Video Frame Interpolation<br/>• TensorRT-Style Fused Convolutions |

---

## 3. Dedicated Track Specifications

### Track A: Text & Frontier LLMs (The Primary Track)

For engineers focused on Large Language Models, reasoning models, and serving infrastructure:

#### Project T1 (Pre-Training): Fused Cross-Entropy Loss & AdamW Engine
* **The Mission**: Eliminate activation memory spills during pre-training by fusing log-softmax, target gathering, and label loss into a single streaming SRAM reduction kernel.
* **Given Asset**: Real vocabulary logits checkpoint (`[Batch=8, Seq=2048, Vocab=32000]`) and target label tokens.
  ```bash
  curl -L -o data/text/pretrain_logits_slice.bin "https://huggingface.co/datasets/llm-systems-open/benchmarks/resolve/main/pretrain_logits.bin"
  ```
* **Kernel Mechanics**:
  * 1-pass online maximum & normalizer reduction across 32,000 vocab entries per token.
  * Direct loss computation with zero DRAM intermediate allocation.
  * Fused AdamW kernel: 128-bit vectorized (`float4`) parameter, momentum, and variance updates.

#### Project T2 (Post-Training): Fused LoRA & Quantization-Aware Delta GEMM
* **The Mission**: Fine-tune an LLM without altering the base 16-bit / 4-bit weights by projecting low-rank rank-16 adapters directly during forward passes.
* **Given Asset**: Base weights (`4096 x 4096`) + LoRA matrices `A` (`4096 x 16`) and `B` (`16 x 4096`) from a public adapter checkpoint.
  ```bash
  curl -L -o data/text/lora_adapter.safetensors "https://huggingface.co/datasets/llm-systems-open/benchmarks/resolve/main/lora_sample.safetensors"
  ```
* **Kernel Mechanics**:
  * Double GEMM fusion: `out = X * W_base + (X * A) * B * scale`.
  * SRAM tiling reusing activations `X` across both operations to save memory bandwidth.

#### Project T3 (Inference Serving): Production PagedAttention & Speculative Verification
* **The Mission**: Build the core memory engine behind vLLM—allocating non-contiguous physical pages for the KV-cache to eliminate memory fragmentation, paired with a speculative draft token verifier.
* **Given Asset**: Prompt sequences with simulated multi-tenant concurrent request queues.
  ```bash
  curl -L -o data/text/prompts_batch.json "https://huggingface.co/datasets/llm-systems-open/benchmarks/resolve/main/prompts_workload.json"
  ```
* **Kernel Mechanics**:
  * **PagedAttention**: Block-table indirection fetching 16-token cache tiles across fragmented GPU DRAM.
  * **Speculative Verification**: Parallel multi-token acceptance kernel evaluating draft tokens against target model logits simultaneously.
  * **Top-P / Top-K Sampler**: Warp-shuffle bitonic sort selecting the next token in `< 5 microseconds`.

---

### Track B: Audio & Speech AI (Whisper / SpeechLLM)

For engineers focused on voice assistants, speech-to-text, and audio generation:

#### Project A1 (Preprocessing & Framing): Continuous Streaming Mel-Spectrogram
* **The Mission**: Transform a continuous stream of raw microphone waveforms into acoustic features at 100x real-time speed.
* **Given Asset**: Sample public speech audio:
  ```bash
  curl -L -o data/audio/speech_sample.wav "https://huggingface.co/datasets/audio-systems-open/samples/resolve/main/sample_16k.wav"
  ```
* **Kernel Mechanics**:
  * C++ binary WAV header deserializer.
  * GPU framing kernel with Hann windowing.
  * cuFFT + in-place halves swap (`fftshift`).
  * Sparse matrix-vector product with 80-channel triangular Mel filterbank.

#### Project A2 (Inference): Causal Streaming Conformer & Audio Tokenizer
* **The Mission**: Real-time neural audio compression using causal depthwise convolutions and residual vector quantization (RVQ).
* **Kernel Mechanics**:
  * Circular buffer cache for streaming 1D convolutions without latency spikes.
  * Codebook nearest-neighbor argmin vector reduction.

---

### Track C: Vision, Diffusion & Video (DiT Style)

For engineers focused on computer vision, diffusion transformers, and video generation:

#### Project V1 (Preprocessing & Patchification): 3D Spatio-Temporal Video Patch Engine
* **The Mission**: Take raw MP4 video frames and convert them into 3D patch tokens (`T x H x W -> Tokens x Dim`) ready for Diffusion Transformers (DiT).
* **Given Asset**: Sample public video clip:
  ```bash
  curl -L -o data/video/sample_clip.mp4 "https://huggingface.co/datasets/vision-systems-open/samples/resolve/main/sample_5s.mp4"
  ```
* **Kernel Mechanics**:
  * Strided 5D tensor unrolling (`[Batch, Time, Channels, Height, Width]`).
  * 3D cache-blocked spatial transposition.
  * Convolutional projection with register-level unrolling.

#### Project V2 (Inference): VAE Latent Space Image Decoder
* **The Mission**: Decode compact latent representations (`[4, 64, 64]`) back into full high-resolution RGB images (`[3, 512, 512]`) using fused upsampling and normalization.
* **Kernel Mechanics**:
  * Nearest-neighbor / bilinear interpolation with boundary clamp guards.
  * GroupNorm reduction across spatial channels in SRAM.

---

## 4. How the Curriculum Navigation Works

In the Digital Book UI, Volume 3 offers a **Track Selector**:

```text
[ Track: Text / LLMs ] [ Track: Audio / Speech ] [ Track: Vision / Video ]
```

When a student selects **Text / LLMs**, their project progression unfolds as:
1. **Pre-Training Project**: Fused Cross-Entropy & AdamW
2. **Post-Training Project**: Fused LoRA & Quantization
3. **Inference Serving Project**: PagedAttention & Speculative Decoding

Each project starts with:
1. **The Physical Reality**: Why the hardware architecture demands this approach (bandwidth bottleneck, cache lines, VRAM fragmentation).
2. **Given Data**: Exact `curl` commands to pull public open datasets into `data/<track>/`.
3. **Step-by-Step Algorithmic Decomposition**: How to break the complex problem into bite-sized CUDA operations.
4. **Interactive Lab**: Live compile via `nvcc`, execute in the sandbox, and verify against reference outputs.
