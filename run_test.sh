#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
python HOPE.py \
  --input_file /home/users/caramr2/hup3d/hup3d-data/  \
  --test \
  --batch_size 64 \
  --model_def resnet18c \
  --gpu 0\
  --gpu_number 0 \
  --pretrained_model ./checkpoints/hup/model-rs50.pkl

