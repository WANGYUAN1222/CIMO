#!/bin/bash

#SBATCH --job-name=h100_predict
#SBATCH --output=/public/home/lab1/wangyuan_workspace/wy_code/code/EE/NER/utd24/sbatch_logs/output_%j.log
#SBATCH --error=/public/home/lab1/wangyuan_workspace/wy_code/code/EE/NER/utd24/sbatch_logs/error_%j.log
#SBATCH --partition=H100      # 申请 H100 分区
#SBATCH --nodes=1             # 只需要 1 个节点
#SBATCH --gres=gpu:2        # 修改 1：将 gpu:5 改为 gpu:8 

# 如果你想指定跑在 h100-3 节点上，取消下面这行的注释
#  # SBATCH --nodelist=h100-3
# 打印一些调试信息（很有用）
echo "=== 环境信息 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working dir: $(pwd)"
which python
python --version
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "=== 开始训练 ==="

# 2. 切换到代码所在的目录
cd /public/home/lab1/wangyuan_workspace/wy_code/code/EE/NER/utd24/

# 3. 运行你的 python 脚本
python extraction_qwen_lora_evaluate.py

