#!/bin/bash
python run.py \
    --frame_dir "data/UCF50_frames" \
    --train_size 0.75 \ 
    --test_size 0.15 \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --mode 'train'
#  Adjust the parameters above as needed. The script will train the model on the UCF50 dataset and save the trained model to disk.
