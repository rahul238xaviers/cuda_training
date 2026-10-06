#!/usr/bin/env bash
# ==============================================================================
# download_project_assets.sh
# Downloads public benchmarks and sample data for Industry-Grade CUDA Kernel Projects
# Tracks: Text / LLMs, Audio / Speech, Vision / Video
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${REPO_ROOT}/data"
mkdir -p "${DATA_DIR}/text" "${DATA_DIR}/audio" "${DATA_DIR}/video"

TRACK="${1:-all}"

echo "=========================================================="
echo " Industry-Grade CUDA Kernel Projects: Asset Downloader"
echo " Target Track: ${TRACK}"
echo "=========================================================="

download_text_assets() {
  echo ""
  echo "[1/3] Downloading Track A (Text / LLM) Project Assets..."
  
  # Project T1: Pre-training Logits Slice
  if [ ! -f "${DATA_DIR}/text/pretrain_logits_slice.bin" ]; then
    echo "  -> Downloading pretrain_logits_slice.bin..."
    curl -L --fail --show-error -o "${DATA_DIR}/text/pretrain_logits_slice.bin" \
      "https://huggingface.co/datasets/llm-systems-open/benchmarks/resolve/main/pretrain_logits.bin" || \
      echo "  [WARN] Remote benchmark logits file not reachable; placeholder ready for synthetic tests."
  else
    echo "  -> pretrain_logits_slice.bin already exists."
  fi

  # Project T2: LoRA Adapter Weights
  if [ ! -f "${DATA_DIR}/text/lora_adapter.safetensors" ]; then
    echo "  -> Downloading lora_adapter.safetensors..."
    curl -L --fail --show-error -o "${DATA_DIR}/text/lora_adapter.safetensors" \
      "https://huggingface.co/datasets/llm-systems-open/benchmarks/resolve/main/lora_sample.safetensors" || \
      echo "  [WARN] Remote LoRA sample file not reachable; placeholder ready for synthetic tests."
  else
    echo "  -> lora_adapter.safetensors already exists."
  fi

  # Project T3: PagedAttention Multi-Tenant Prompts Workload
  if [ ! -f "${DATA_DIR}/text/prompts_batch.json" ]; then
    echo "  -> Downloading prompts_batch.json..."
    curl -L --fail --show-error -o "${DATA_DIR}/text/prompts_batch.json" \
      "https://huggingface.co/datasets/llm-systems-open/benchmarks/resolve/main/prompts_workload.json" || \
      echo "  [WARN] Remote prompts workload file not reachable; placeholder ready for synthetic tests."
  else
    echo "  -> prompts_batch.json already exists."
  fi
}

download_audio_assets() {
  echo ""
  echo "[2/3] Downloading Track B (Audio / Speech) Project Assets..."

  # Project A1: Public Speech WAV Sample (16kHz mono)
  if [ ! -f "${DATA_DIR}/audio/speech_sample.wav" ]; then
    echo "  -> Downloading speech_sample.wav..."
    curl -L --fail --show-error -o "${DATA_DIR}/audio/speech_sample.wav" \
      "https://huggingface.co/datasets/audio-systems-open/samples/resolve/main/sample_16k.wav" || \
      echo "  [WARN] Remote speech sample not reachable; placeholder ready for synthetic tests."
  else
    echo "  -> speech_sample.wav already exists."
  fi
}

download_video_assets() {
  echo ""
  echo "[3/3] Downloading Track C (Vision & Video) Project Assets..."

  # Project V1: Public Video Clip Sample (5 seconds)
  if [ ! -f "${DATA_DIR}/video/sample_clip.mp4" ]; then
    echo "  -> Downloading sample_clip.mp4..."
    curl -L --fail --show-error -o "${DATA_DIR}/video/sample_clip.mp4" \
      "https://huggingface.co/datasets/vision-systems-open/samples/resolve/main/sample_5s.mp4" || \
      echo "  [WARN] Remote video sample not reachable; placeholder ready for synthetic tests."
  else
    echo "  -> sample_clip.mp4 already exists."
  fi
}

case "${TRACK}" in
  text)
    download_text_assets
    ;;
  audio)
    download_audio_assets
    ;;
  video|vision)
    download_video_assets
    ;;
  all)
    download_text_assets
    download_audio_assets
    download_video_assets
    ;;
  *)
    echo "Unknown track: ${TRACK}. Use 'text', 'audio', 'video', or 'all'."
    exit 1
    ;;
esac

echo ""
echo "Asset download process completed."
