#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

#CONFIG_1_1=$1
#GPUS=$2

#$PYTHON /workspace/tools/train.py --config-file /workspace/configs/rsipac_segformer_1_1.yaml
#wait
#$PYTHON /workspace/tools/train.py --config-file /workspace/configs/rsipac_segformer_step_1_2.yaml   #因为加载已训练模型，所以会多占用几百M显存
#wait
##$PYTHON /workspace/tools/train.py --config-file /workspace/configs/rsipac_segformer_2_1.yaml
##wait
##$PYTHON /workspace/tools/train.py --config-file /workspace/configs/rsipac_segformer_step_2_2.yaml
#$PYTHON /workspace/run.py

#$PYTHON tools/train.py --config-file configs/rsipac_segformer_landcover30.yaml

#56:
#$PYTHON tools/train.py --config-file configs/sandong/sandong_56_segformer_b2_ce_dice_scale2.yaml
#wait
$PYTHON tools/train.py --config-file configs/sandong/sandong_56_segformer_b3_ce_focal_scale2.yaml
#$PYTHON tools/train.py --config-file configs/sandong/sandong_56_segformer_b4_ce_focal_scale2.yaml

#70:
#$PYTHON tools/train.py --config-file configs/sandong/sandong_70_segformer_b1_aug2_ce_dice_scale2.yaml
wait
#$PYTHON tools/train.py --config-file configs/sandong/sandong_70_segformer_b1_aug2_dice_sce_contrast_scale2.yaml

#/data_zs/code/loveDA/pytorchAI_segmentation_loveda/tools
#python tools/train.py --config-file