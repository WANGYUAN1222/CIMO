#!/bin/bash
# ══════════════════════════════════════════════
# vLLM 启动脚本 —— Llama-3.3-70B-Instruct
# ══════════════════════════════════════════════

# MODEL_PATH="/data_share_from_3090/wy_code/code/EE/NER/utd24/LLM-Research/Llama-3.3-70B-Instruct"
# --served-model-name  "llama3.3-70b" \


MODEL_PATH="/data_share_from_3090/wy_code/code/EE/NER/utd24/gemma-4"
#  --served-model-name  "gemma-4" \
# 指定使用的 GPU 编号（70B 建议至少 4 张 A100/80G 或 8 张 A100/40G）
export CUDA_VISIBLE_DEVICES=0,1,2

python -m vllm.entrypoints.openai.api_server \
    --model              "$MODEL_PATH" \
    --served-model-name  "gemma-4" \
    --tensor-parallel-size 3 \
    --dtype              float16 \
    --max-model-len      16384 \
    --gpu-memory-utilization 0.90 \
    --host               0.0.0.0 \
    --port               8000 \
    --trust-remote-code

# ── 常用参数说明 ────────────────────────────────
# --tensor-parallel-size   : GPU 数量（与 CUDA_VISIBLE_DEVICES 中的卡数一致）
# --dtype bfloat16         : 70B 推荐 bfloat16；显存不足可改 float16
# --max-model-len          : 最大上下文长度，按需调大（如 16384）
# --gpu-memory-utilization : 显存占用比例，0.90 留 10% 余量
# --served-model-name      : API 调用时使用的模型名，可自定义