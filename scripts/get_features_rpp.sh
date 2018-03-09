#!/bin/bash
# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/world/data-gpu-94/sysu-reid/checkpoints
# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/world/data-gpu-94/sysu-reid/checkpoints/ResNet_PCB_and_RPP_v2
# Where the dataset is saved to.
DATASET_DIR=/home/yuanziyi/Market-1501
# WHere the log is saved to
LOG_DIR=/home/yuanziyi/log
# Wher the tfrecord file is save to
PROBE_OUTPUT_DIR=/world/data-gpu-94/sysu-reid/zhangkaicheng/Market-1501-tfrecord/query
# Wher the tfrecord file is save to
GALLERY_OUTPUT_DIR=/world/data-gpu-94/sysu-reid/zhangkaicheng/Market-1501-tfrecord/bounding_box_test
# python get_features_rpp.py \
# --dataset_name=Market_1501 \
# --probe_dataset_dir=${PROBE_OUTPUT_DIR} \
# --gallery_dataset_dir=${GALLERY_OUTPUT_DIR} \
# --batch_size=8 \
# --max_number_of_steps=10001 \
# --checkpoint_dir=${TRAIN_DIR} \
# --pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/resnet_v2_50.ckpt \
# --log_dir=${LOG_DIR} \
# --weight_decay=0.00004 \
# --ckpt_num=143185 \
# --scale_height=384 \
# --scale_width=128 \
# --GPU_use=4 \
# --only_pcb=False \
# --only_classifier=False

# python get_features_rpp.py \
# --dataset_name=Market_1501 \
# --probe_dataset_dir=${PROBE_OUTPUT_DIR} \
# --gallery_dataset_dir=${GALLERY_OUTPUT_DIR} \
# --batch_size=8 \
# --max_number_of_steps=10001 \
# --checkpoint_dir=${TRAIN_DIR} \
# --pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/resnet_v2_50.ckpt \
# --log_dir=${LOG_DIR} \
# --weight_decay=0.00004 \
# --ckpt_num=151229 \
# --scale_height=384 \
# --scale_width=128 \
# --GPU_use=4 \
# --only_pcb=False \
# --only_classifier=False

python get_features_rpp.py \
--dataset_name=Market_1501 \
--probe_dataset_dir=${PROBE_OUTPUT_DIR} \
--gallery_dataset_dir=${GALLERY_OUTPUT_DIR} \
--batch_size=8 \
--max_number_of_steps=10001 \
--checkpoint_dir=${TRAIN_DIR} \
--pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/resnet_v2_50.ckpt \
--log_dir=${LOG_DIR} \
--weight_decay=0.00004 \
--ckpt_num=105144 \
--scale_height=384 \
--scale_width=128 \
--GPU_use=7 \
--only_pcb=False \
--only_classifier=True

python get_features_rpp.py \
--dataset_name=Market_1501 \
--probe_dataset_dir=${PROBE_OUTPUT_DIR} \
--gallery_dataset_dir=${GALLERY_OUTPUT_DIR} \
--batch_size=8 \
--max_number_of_steps=10001 \
--checkpoint_dir=${TRAIN_DIR} \
--pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/resnet_v2_50.ckpt \
--log_dir=${LOG_DIR} \
--weight_decay=0.00004 \
--ckpt_num=118145 \
--scale_height=384 \
--scale_width=128 \
--GPU_use=7 \
--only_pcb=False \
--only_classifier=True

python get_features_rpp.py \
--dataset_name=Market_1501 \
--probe_dataset_dir=${PROBE_OUTPUT_DIR} \
--gallery_dataset_dir=${GALLERY_OUTPUT_DIR} \
--batch_size=8 \
--max_number_of_steps=10001 \
--checkpoint_dir=${TRAIN_DIR} \
--pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/resnet_v2_50.ckpt \
--log_dir=${LOG_DIR} \
--weight_decay=0.00004 \
--ckpt_num=137678 \
--scale_height=384 \
--scale_width=128 \
--GPU_use=7 \
--only_pcb=False \
--only_classifier=True