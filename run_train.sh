#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
python HOPE.py \
  --input_file /home/users/caramr2/hup3d/hup3d-data/ \
  --output_file ./checkpoints/hup/model-rs \
  --train \
  --val \
  --batch_size 64 \
  --model_def resnet18c \
  --gpu 0\
  --gpu_number 0 \
  --learning_rate 1e-2 \
  --lr_step 10 \
  --lr_step_gamma 0.95 \
  --log_batch 100 \
  --val_epoch 1 \
  --snapshot_epoch 10 \
  --num_iterations 100 \
#  --pretrained_model ./checkpoints/obman/model-0.pkl

