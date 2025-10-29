#!/bin/bash
#SBATCH --job-name=remove_undownloaded_audio
#SBATCH --output=/share/users/student/m/mnsiah/logs/remove_undownloaded_audio-%j.out
#SBATCH --error=/share/users/student/m/mnsiah/logs/remove_undownloaded_audio-%j.err
#SBATCH --time=00:50:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G


# -----------------------
# Load your conda environment
# -----------------------
source ~/.bashrc
conda activate fusion

export HF_HOME=/share/users/student/m/mnsiah/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
#export TORCH_HOME=/share/users/student/m/mnsiah/.cache/torch
export TMPDIR=/share/users/student/m/mnsiah/tmp
mkdir -p $HF_HOME $TORCH_HOME $TMPDIR


# -----------------------
# Run training
# -----------------------
python remove_audio.py
