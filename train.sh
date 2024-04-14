#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=62gb
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.out
#SBATCH --job-name=fiveDifTask
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100-4
nvidia-smi
METHOD=channel
LR=1e-2
N_PREFIX=10
TASK=fiveSimTask
SPLIT=train
MODEL=gpt2-large
CUDA_LAUNCH_BLOCKING=1 python train.py\
    --gpt2 $MODEL\
    --task $TASK\
    --split $SPLIT\
    --method $METHOD\
    --n_gpu 1\
    --tensorize_dir tensorized\
    --out_dir checkpoints/$MODEL/$TASK-$SPLIT/prefix={$N_PREFIX}-{$METHOD}-lr={$LR}-initByVocab\
    --batch_size 16\
    --lr $LR\
    --n_prefix_tokens $N_PREFIX\
    --num_training_steps 10000\

