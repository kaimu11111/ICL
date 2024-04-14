#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=62gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=test_fiveDifTask
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100-4

TRAIN_METHOD=channel
TEST_METHOD=channel
LR=1e-2
N_PREFIX=10
DATASET=ethos-directed_vs_generalized
TRAIN_TASK=fiveSimTask
SPLIT=train
MODEL=gpt2-large
TRAIN_SIZE=100
STEP=10000
K=4
python test.py\
    --dataset $DATASET\
    --gpt $MODEL\
    --method $TEST_METHOD\
    --test_batch_size 16\
    --out_dir out/$MODEL\
    --k $K\
    --embedding_dir embeddings/\
    --concept_temperature 50\
    --similarity_temperature 0.1\
    --train_size $TRAIN_SIZE\
    --difficulty concept_calibrated\
    --n_prefix_tokens $N_PREFIX\
    --concept_dir concept_likelihood/gpt2-large/$TRAIN_TASK-$SPLIT-$TRAIN_SIZE/$DATASET-$TRAIN_METHOD-prefix=$N_PREFIX-lr=$LR-$STEP\
    --prefix_embed_file checkpoints/gpt2-large/$TRAIN_TASK-$SPLIT/prefix={$N_PREFIX}-{$TRAIN_METHOD}-lr={$LR}-initByVocab/model-$STEP.pt\
    --prior easiest\
    --reorder\
    # --prior most_similar\
