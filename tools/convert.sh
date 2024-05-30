#!/usr/bin/env bash
DEPLOY_CONFIG=$/chase/mmdeploy/testconfig/deploy_config.py
model_config=$/home/chase/shy/dataset/spjgh/models/yolox_l_8x8_300e_coco_spjgh.py
checkpoint_file=$/home/chase/shy/dataset/spjgh/models/epoch_300.pth
workdir=$test2
imgpath=$/home/chase/shy/138/images/6454.jpg

python3 ./tools/deploy.py /chase/mmdeploy/testconfig/deploy_config.py /home/chase/shy/dataset/spjgh/models/yolox_l_8x8_300e_coco_spjgh.py /home/chase/shy/dataset/spjgh/models/epoch_300.pth /home/chase/shy/138/images/25.jpg --work-dir=test2 --device=cuda:0

