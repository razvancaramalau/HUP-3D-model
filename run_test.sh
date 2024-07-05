#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
python HOPE.py \
  --input_file /home/razvan/handobjectnet_baseline/data/syn_colibri_v1/ \
  --test \
  --batch_size 64 \
  --model_def hopenet \
  --gpu \
  --gpu_number 0 \
  --pretrained_model ./checkpoints/hup/model-30.pkl

