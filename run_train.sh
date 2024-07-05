#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
python HOPE.py \
  --input_file /home/razvan/handobjectnet_baseline/data/syn_colibri_v1/ \
  --output_file ./checkpoints/hup/model- \
  --train \
  --val \
  --batch_size 64 \
  --model_def hopenet \
  --gpu \
  --gpu_number 0 \
  --learning_rate 1e-3 \
  --lr_step 10 \
  --lr_step_gamma 0.95 \
  --log_batch 100 \
  --val_epoch 1 \
  --snapshot_epoch 10 \
  --num_iterations 1000 \
#  --pretrained_model ./checkpoints/obman/model-0.pkl

