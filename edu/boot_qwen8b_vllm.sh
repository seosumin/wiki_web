#!/bin/bash
# Qwen3-8B vLLM 서버 (교육툴 보고서 생성용, gemma4 env 재사용 — vllm 0.24.0).
# 모델은 ext4 로컬(/home/sumin/models/Qwen3-8B)에 복사해둠 → 빠른 로딩(9P 캐시 대비).
# 종료: fuser -k 8005/tcp 후 남은 VLLM::EngineCore PID를 kill -9 (GPU orphan 방지).
#       pkill -f 금지 — self-kill 함정.
source /home/sumin/miniconda3/etc/profile.d/conda.sh
conda activate gemma4

export LD_LIBRARY_PATH="/home/sumin/miniconda3/envs/gemma4/lib:$LD_LIBRARY_PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# WSL2에서 vLLM V1은 핀 메모리를 기본 비활성 → UvaBuffer("UVA is not available") 실패.
# WSL2 핀 메모리는 실제 동작하므로 명시적으로 켠다 (없으면 엔진 코어 init 실패).
export VLLM_WSL2_ENABLE_PIN_MEMORY=1

MODEL="/home/sumin/models/Qwen3-8B"
SERVED_NAME="Qwen/Qwen3-8B"
PORT="${VLLM_PORT:-8005}"

if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
  echo "ERROR: port $PORT already in use — 기존 vLLM을 먼저 종료하세요 (fuser -k $PORT/tcp)"
  exit 1
fi

echo "=== vLLM Qwen3-8B (env: gemma4, vllm 0.24.0) ==="
echo "Model: $MODEL   Port: $PORT   ServedName: $SERVED_NAME"

# Qwen3-8B는 8B/bf16(~16GB) — 96GB GPU에 여유. thinking 여부는 요청 단(enable_thinking)에서 제어.
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}" vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching
