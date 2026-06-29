#!/bin/bash

#SBATCH --job-name=AI-Learning-Agent
#SBATCH --partition=gpu
#SBATCH --nodelist=node1
#SBATCH --error=/home/pkc/MM/projects/autoresearch/logs/agent_error_%j.log
#SBATCH --output=/home/pkc/MM/projects/autoresearch/logs/agent_output_%j.log
#SBATCH --gres=shard:90

# ── Absolute paths ────────────────────────────────────────────────────────────
PROJECT_ROOT=/home/pkc/MM/projects
AGENT_DIR=${PROJECT_ROOT}/autoresearch # Update this if the repo folder name changes

cd ${AGENT_DIR}

mkdir -p ${AGENT_DIR}/logs

# ── Load system modules ───────────────────────────────────────────────────────
module load anaconda3-2024.2
module load cuda-12.8

# ── Set environment and python path ───────────────────────────────────────────
source ${PROJECT_ROOT}/test-vllm-env/bin/activate
export PYTHONPATH=${AGENT_DIR}:${PYTHONPATH}

echo "[ENV] Python  : $(which python)"

# ── Misc env flags ────────────────────────────────────────────────────────────
export TF_ENABLE_ONEDNN_OPTS=0
# Note: Keep these 0 if downloading the model weights for the very first time. Set to 1 if the model is already cached.
export HF_HUB_OFFLINE=0       
export TRANSFORMERS_OFFLINE=0 
export HF_DATASETS_OFFLINE=0  

# ── Launch generation ─────────────────────────────────────────────────────────
python -u ${AGENT_DIR}/generate.py \
    --subject "Machine Learning" \
    --model "Qwen/Qwen2.5-32B-Instruct"
