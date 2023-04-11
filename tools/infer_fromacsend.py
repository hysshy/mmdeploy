import argparse
import os
from mmcv import DictAction, Config
from mmcv.parallel import MMDataParallel, collate
from mmdet.utils import compat_cfg
from mmdeploy.apis import build_task_processor
from mmdet.datasets import replace_ImageToTensor
from mmdeploy.utils.config_utils import load_config
from mmdet.datasets.pipelines import Compose
from mmdeploy.utils.device import parse_device_id
from mmdeploy.utils.timer import TimeCounter
import cv2
import time
import numpy as np
costTime = []

def get_test_pipeline(config, ifNdarray=True):
    cfg = Config.fromfile(config)
    cfg = compat_cfg(cfg)
    score_thr = cfg.model.test_cfg.score_thr
    score_thr = 0.0
    if ifNdarray:
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
    return test_pipeline, score_thr


def detect(model, img, score_thr, test_pipeline):
    if isinstance(img, np.ndarray):
        # directly add img
        data = dict(img=img)
    else:
        data = dict(img_info=dict(filename=img), img_prefix=None)
    data = test_pipeline(data)
    datas = [data]
    data = collate(datas, samples_per_gpu=1)
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    data.setdefault('mutilTask', True)
    data.setdefault('score_thr', score_thr)
    start = time.time()
    result = model(return_loss=False, rescale=True, **data)
    print(result)

def draw_img(img, imgName, det_bboxes, det_labels, kps, zitais, mohus, savePath, categoriesName, zitai_categoriesName):
    if isinstance(img, str):
        img = cv2.imread(img)

    for i in range(len(det_bboxes)):
        bbox = det_bboxes[i]
        label = det_labels[i]
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 1)
        cv2.putText(img, categoriesName[label], (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if label in [0,1]:
            faceKp = kps[i].astype(int)
            for k in range(5):
                point = (faceKp[k][0], faceKp[k][1])
                cv2.circle(img, point, 3, (255, 0, 0), 0)
            zitai_label = zitais[i]
            if not os.path.exists(savePath+'/'+zitai_categoriesName[zitai_label]):
                os.makedirs(savePath+'/'+zitai_categoriesName[zitai_label])
            cv2.imwrite(savePath+'/'+zitai_categoriesName[zitai_label]+'/'+imgName.replace('.jpg', '_'+str(i)+'.jpg'), img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
            if label == 0:
                mohu_label = str(round(mohus[i][0], 2))
                if not os.path.exists(savePath+'/'+mohu_label):
                    os.makedirs(savePath+'/'+mohu_label)
                cv2.imwrite(savePath+'/'+mohu_label+'/'+imgName.replace('.jpg', '_'+str(i)+'.jpg'), img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
    cv2.imwrite(savePath+'/'+imgName, img)


if __name__ == '__main__':
    imgPath = '/home/chase/Desktop/138/images'
    savePath = '/home/chase/shy/testdata/draw'
    categoriesName = ['face','facewithmask','person', 'lianglunche', 'sanlunche', 'car', 'truck', 'dog', 'cat']
    zitai_categoriesName = ['微右', '微左', '正脸', '下左', '下右', '微下', '重上', '重下', '重右', '重左', '遮挡或半脸']
    deploy_cfg_path = '/home/chase/shy/deploy_config.py'
    model_cfg_path = '/home/chase/shy/yolox_m_8x8_300e_coco_spjgh.py'
    model_file = ['/home/chase/shy/mmdeploy/tools/work_dir8/end2end.engine']

    # load deploy_cfg
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)
    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cuda:0')
    # load the model of the backend
    model = task_processor.init_backend_model(model_file, uri=None)

    destroy_model = model.destroy
    model = MMDataParallel(model, device_ids=[0])
    if hasattr(model.module, 'CLASSES'):
        model.CLASSES = model.module.CLASSES
    test_pipeline, score_thr = get_test_pipeline(model_cfg_path, ifNdarray=True)
    img = cv2.imread(imgPath + '/' + imgName)
    detect(model, img, score_thr, test_pipeline)

