import numpy as np
from mmdet.apis import init_detector, inference_detector
import mmcv
import time
import os
import cv2
config_file = '/home/chase/shy/ascend/faster_rcnn_r50_fpn_2x_coco.py'
checkpoint_file = '/home/chase/shy/ascend/epoch_12.pth'
costTime = []
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
imgPath = '/home/chase/shy/ascend/luomu_images'
savePath = '/home/chase/shy/testdata/draw'
i = 0
for imgName in os.listdir(imgPath):
    print(imgName)
    start = time.time()
    img = cv2.imread(imgPath + '/' + imgName)
    result = inference_detector(model, img)
    timecost = time.time() - start
    print(timecost)
    costTime.append(timecost)
    i += 1
    if i == 100:
        break
print(np.mean(costTime))
# visualize the results in a new window
# model.show_result(img, result)
# model.show_result(img, result, out_file='result.jpg')