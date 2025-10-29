#!/bin/bash
#SBATCH --job-name=data-gen
#SBATCH --output=/share/users/student/m/mnsiah/logs/data-gen-%j.out
#SBATCH --error=/share/users/student/m/mnsiah/logs/data-gen-%j.err
#SBATCH --time=08:00:00
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
python data_gen.py
#salloc --ntasks 1 --cpus-per-task 128
