#!/bin/bash
#SBATCH --job-name=fusion_train
#SBATCH --output=/share/users/student/m/mnsiah/logs/fusion_train-%j.out
#SBATCH --error=/share/users/student/m/mnsiah/logs/fusion_train-%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:H100.80gb
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
#python single_audio_train.py
#python context_audio_train.py
#python text_train.py
python metrics.py
#python fusion_loss.py
#python new_metrics.py
#python per_class_val.py


###H100.80gb