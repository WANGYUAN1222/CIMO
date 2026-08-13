#!/bin/bash
set -e  # 遇到错误立即退出

ENV_NAME="vllm_env"
PYTHON_VERSION="3.10"

echo "========================================"
echo "  创建新 Conda 环境: $ENV_NAME"
echo "========================================"

# ── 1. 创建新环境 ────────────────────────────
echo ""
echo "[1/5] 创建 Python $PYTHON_VERSION 环境..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# ── 2. 激活环境 ──────────────────────────────
echo ""
echo "[2/5] 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# ── 3. 安装 PyTorch 2.5.1 + CUDA 12.4 ───────
echo ""
echo "[3/5] 安装 PyTorch 2.5.1 (cu124)..."

# 国内网络优先走清华镜像，如可访问官方则注释掉第一行换第二行
pip install torch==2.5.1 torchvision torchaudio \
    --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu124

# 如果清华镜像有问题，改用官方源：
# pip install torch==2.5.1 torchvision torchaudio \
#     --index-url https://download.pytorch.org/whl/cu124

# ── 4. 安装 vLLM 0.7.3 ──────────────────────
echo ""
echo "[4/5] 安装 vLLM 0.7.3..."
pip install vllm==0.7.3

# ── 5. 安装常用依赖 ──────────────────────────
echo ""
echo "[5/5] 安装常用依赖..."
pip install \
    transformers>=4.45.0 \
    accelerate \
    sentencepiece \
    tiktoken \
    einops \
    packaging \
    ninja \
    numpy

# ── 6. 验证安装 ──────────────────────────────
echo ""
echo "========================================"
echo "  验证安装结果"
echo "========================================"
python -c "
import torch
import vllm
print(f'✅ Python     : $(python --version)')
print(f'✅ PyTorch    : {torch.__version__}')
print(f'✅ vLLM       : {vllm.__version__}')
print(f'✅ CUDA 可用  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU 数量   : {torch.cuda.device_count()}')
    print(f'✅ 当前 GPU  : {torch.cuda.get_device_name(0)}')
    print(f'✅ CUDA 版本 : {torch.version.cuda}')
"

echo ""
echo "========================================"
echo "  环境创建完成！"
echo "  使用方式: conda activate $ENV_NAME"
echo "========================================"