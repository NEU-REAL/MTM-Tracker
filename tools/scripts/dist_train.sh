#!/usr/bin/env bash

set -x

#python -m torch.distributed.launch --nproc_per_node=1 train_track.py \
#--launcher pytorch --batch_size 32 --epoch 40 --cfg_file cfgs/kitti/mtm-cyc.yaml \
#--tcp_port 18819 --fix_random_seed --sync_bn --extra_tag base \
#--max_ckpt_save_num 15

python -m torch.distributed.launch --nproc_per_node=1 train_track.py \
--launcher pytorch --batch_size 32 --epoch 40 --cfg_file cfgs/kitti/mtm-van.yaml \
--tcp_port 18819 --fix_random_seed --sync_bn --extra_tag base \
--max_ckpt_save_num 15

#python -m torch.distributed.launch --nproc_per_node=1 train_track.py \
#--launcher pytorch --batch_size 32 --epoch 40 --cfg_file cfgs/kitti/mtm-ped.yaml \
#--tcp_port 18819 --fix_random_seed --sync_bn --extra_tag base \
#--max_ckpt_save_num 15

#python -m torch.distributed.launch --nproc_per_node=1 train_track.py \
#--launcher pytorch --batch_size 24 --epoch 40 --cfg_file cfgs/kitti/mtm-car.yaml \
#--tcp_port 18819 --fix_random_seed --sync_bn --extra_tag base \
#--max_ckpt_save_num 15