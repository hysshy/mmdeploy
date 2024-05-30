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
from util import prefactor_fence, drawBboxes, drawFacekps, rectLabels
import cv2
import time
import numpy as np
costTime = []

def get_test_pipeline(config, ifNdarray=True):
    cfg = Config.fromfile(config)
    cfg = compat_cfg(cfg)
    score_thr = cfg.model.test_cfg.score_thr
    score_thr2 = cfg.model.test_cfg.score_thr2
    if ifNdarray:
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
    return test_pipeline, score_thr, score_thr2


def detect(model, img, score_thr, score_thr2, test_pipeline):
    start = time.time()
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
    data.setdefault('mutilTask', False)
    data.setdefault('score_thr', score_thr)
    data.setdefault('score_thr2', score_thr2)
    print(data['img_metas'])
    #start = time.time()
    det, label, kps, zitais, mohus, body_bboxes, body_labels, upclouse_styles, clouse_colors = model(return_loss=False, rescale=True, **data)
    timecost = time.time() - start
    print(timecost)
    costTime.append(timecost)
    return det, label, kps, zitais, mohus, body_bboxes, body_labels, upclouse_styles, clouse_colors

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
    imgPath = '/home/chase/shy/138/images'
    savePath = '/home/chase/shy/138/draw'
    categoriesName = ['face','facewithmask','person', 'lianglunche', 'sanlunche', 'car', 'truck', 'dog', 'cat']
    zitai_categoriesName = ['微右', '微左', '正脸', '下左', '下右', '微下', '重上', '重下', '重右', '重左', '遮挡或半脸']
    bodydetector_categoriesName = ['short_sleeves', 'long_sleeves', 'skirt', 'long_trousers', 'short_trousers', 'backbag',
                               'glasses', 'handbag', 'hat', 'haversack', 'trunk']
    clousestyle_categoriesName = ['medium_long_style', 'medium_style', 'long_style']
    clousecolor_categoriesName = ['light_blue', 'light_red', 'khaki', 'gray', 'blue', 'red', 'green', 'brown', 'yellow',
                              'purple', 'white', 'orange', 'deep_blue', 'deep_green', 'deep_red', 'black', 'stripe',
                              'lattice', 'mess', 'decor', 'blue_green']
    deploy_cfg_path = '/chase/mmdeploy/testconfig/deploy_config.py'
    model_cfg_path = '/home/chase/shy/dataset/spjgh/models/yolox_l_8x8_300e_coco_spjgh.py'
    model_file = ['/chase/mmdeploy/test2/end2end.engine']

    # load deploy_cfg
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)
    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cuda:0')
    # load the model of the backend
    model = task_processor.init_backend_model(model_file, uri=None)

    destroy_model = model.destroy
    model = MMDataParallel(model, device_ids=[0])
    if hasattr(model.module, 'CLASSES'):
        model.CLASSES = model.module.CLASSES
    test_pipeline, score_thr, score_thr2 = get_test_pipeline(model_cfg_path, ifNdarray=True)
    i = 0
    costtime2 = []
    for imgName in os.listdir(imgPath):
        print(imgName)
        img = cv2.imread(imgPath+'/'+imgName)
        start=time.time()
        det, label, kps, zitais, mohus, body_bboxes, body_labels, upclouse_styles, clouse_colors = detect(model, img, score_thr, score_thr2, test_pipeline)
        car_bboxes, car_labels, pet_bboxes, pet_labels, person_bboxes, person_labels, upclouse_bboxes_list, upclouse_labels_list, upclouse_colors_list, upclouse_styles_list,\
           downclouse_bboxes_list, downclouse_labels_list, downclouse_colors_list, otherfactor_bboxes_list, otherfactor_labels_list, face_bboxes_list, face_labels_list, face_bboxes,\
           face_labels, face_kps, face_zitais, face_mohus, facefactor_bboxes_list_all, facefactor_labels_list_all = prefactor_fence(det, label, kps, zitais, mohus, body_bboxes, body_labels, upclouse_styles, clouse_colors, None)
        costtime2.append(time.time()-start)
        drawBboxes(img, car_bboxes, car_labels, categoriesName)
        drawBboxes(img, pet_bboxes, pet_labels, categoriesName)
        drawBboxes(img, person_bboxes, person_labels, categoriesName)
        drawBboxes(img, face_bboxes, face_labels, categoriesName)
        drawFacekps(img, face_kps)
        rectLabels(img, face_bboxes, face_zitais, zitai_categoriesName, savePath, imgName)
        rectLabels(img, face_bboxes, face_mohus, zitai_categoriesName, savePath, imgName, type='mohu')
        #drawBboxes(img, body_bboxes, body_labels, bodydetector_categoriesName)
        for i in range(len(upclouse_bboxes_list)):
            drawBboxes(img, upclouse_bboxes_list[i], upclouse_labels_list[i], bodydetector_categoriesName)
        for i in range(len(downclouse_bboxes_list)):
            drawBboxes(img, downclouse_bboxes_list[i], downclouse_labels_list[i], bodydetector_categoriesName)
        for i in range(len(otherfactor_bboxes_list)):
            drawBboxes(img, otherfactor_bboxes_list[i], otherfactor_labels_list[i], bodydetector_categoriesName)
        for i in range(len(upclouse_bboxes_list)):
            rectLabels(img, upclouse_bboxes_list[i], upclouse_colors_list[i], clousecolor_categoriesName, savePath, imgName)
            rectLabels(img, upclouse_bboxes_list[i], upclouse_styles_list[i], clousestyle_categoriesName, savePath, imgName)
        for i in range(len(downclouse_bboxes_list)):
            rectLabels(img, downclouse_bboxes_list[i], downclouse_colors_list[i], clousecolor_categoriesName, savePath, imgName)

        cv2.imwrite(savePath+'/'+imgName, img)




        #draw_img(img, imgName, det, label, kps, zitais, mohus, savePath, categoriesName, zitai_categoriesName)
        i += 1
        if i == 100:
            break
    costTime.pop(0)
    costtime2.pop(0)
    print(np.mean(costTime))
    print(np.mean(costtime2))
    # samples_per_gpu = 1
    # model = ONNXRuntimeDetector(
    #     model, class_names=('luomu', 'back'), device_id=0)
    # model = MMDataParallel(model, device_ids=[0])
    # model.eval()
    # test_pipeline, score_thr = get_test_pipeline(config, ifNdarray=True)
    #
    # for imgName in os.listdir(imgPath):
    #     img = cv2.imread(imgPath+'/'+imgName)
    #     # img = cv2.resize(img,(1024,576))
    #     result = detect(model, img, score_thr, test_pipeline)
    #     draw_img(img, imgName, result, savePath, categoriesName, zitaiCategoriesName)


