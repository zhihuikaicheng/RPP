#!/bin/bash
# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/world/data-gpu-94/sysu-reid/checkpoints
# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/world/data-gpu-94/sysu-reid/checkpoints/ResNet_v2_baseline_decay
# Where the dataset is saved to.
DATASET_DIR=/world/data-gpu-94/sysu-reid/zhangkaicheng/Market-1501
# WHere the log is saved to
LOG_DIR=/world/data-gpu-94/sysu-reid/zhangkaicheng/log
# Wher the tfrecord file is save to
OUTPUT_DIR=/world/data-gpu-94/sysu-reid/zhangkaicheng/Market-1501-tfrecord/bounding_box_train
python train.py \
--learning_rate_one=0.01 \
--learning_rate_two=0.1 \
--learning_rate_decay_type=exponential \
--learning_rate_decay_factor=0.1 \
--weight_decay=0.00004 \
--num_epochs_per_decay=40 \
--dataset_name=Market_1501 \
--dataset_dir=${OUTPUT_DIR} \
--batch_size=32 \
--max_number_of_steps=24000 \
--checkpoint_dir=${TRAIN_DIR} \
--pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/resnet_v2_50.ckpt \
--log_dir=${LOG_DIR} \
--save_model_summary_secs=300 \
--log_every_n_steps=100 \
--optimizer=sgd \
--weight_decay=0.00004 \
--scale_height=384 \
--scale_width=128 \
--GPU_use=3 \


# python train.py \
# --learning_rate=0.01 \
# --learning_rate_decay_type=fixed \
# --dataset_name=Market_1501 \
# --dataset_split_name=train \
# --dataset_dir=${OUTPUT_DIR} \
# --batch_size=32 \
# --max_number_of_steps=16000 \
# --checkpoint_dir=${TRAIN_DIR} \
# --pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/resnet_v2_50.ckpt \
# --log_dir=${LOG_DIR} \
# --save_model_summary_secs=300 \
# --log_every_n_steps=100 \
# --optimizer=sgd \
# --weight_decay=0.00004 \
# --scale_height=384 \
# --scale_width=128 \
# --GPU_use=3 \

# python train.py \
# --learning_rate=0.001 \
# --learning_rate_decay_type=fixed \
# --dataset_name=Market_1501 \
# --dataset_dir=${OUTPUT_DIR} \
# --batch_size=32 \
# --max_number_of_steps=24000 \
# --checkpoint_dir=${TRAIN_DIR} \
# --pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/resnet_v2_50.ckpt \
# --log_dir=${LOG_DIR} \
# --save_model_summary_secs=300 \
# --log_every_n_steps=100 \
# --optimizer=sgd \
# --weight_decay=0.00004 \
# --scale_height=384 \
# --scale_width=128 \
# --GPU_use=3 \

