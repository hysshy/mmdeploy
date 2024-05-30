import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import cv2
import os
colorList = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255)]

def drawBboxes(img, bboxes, labels, classes, extraLabels=None, extraClasses=None, extraLabels2=None, extraClasses2=None):
    for i in range(len(bboxes)):
        bbox = bboxes[i].astype(int)
        if len(bbox) > 0:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colorList[i % len(colorList)], 1)
            label = labels[i]
            name = classes[label]
            if extraLabels is not None:
                name = name + '_'  + extraClasses[extraLabels[i]]
            if extraLabels2 is not None:
                name = name + '_' + extraClasses2[extraLabels2[i]]
            cv2.putText(img, name, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        colorList[i % len(colorList)], 2)

def drawFacekps(img, face_kps):
    for i in range(len(face_kps)):
        faceKp = face_kps[i].astype(int)
        for k in range(5):
            point = (faceKp[k][0], faceKp[k][1])
            cv2.circle(img, point, 3, (255, 0, 0), 0)

def rectLabels(img, bboxes, labels, classes, savePath, imgName, type='zitai'):
    for i in range(len(bboxes)):
        bbox = bboxes[i].astype(int)
        label = labels[i]
        if type == 'mohu':
            mohu_label = labels[i]
            name = str(round(mohu_label[0], 2))
        else:
            name = classes[label]
        if not os.path.exists(savePath + '/' + name):
            os.makedirs(savePath + '/' + name)
        cv2.imwrite(savePath + '/' + name + '/' + imgName.replace('.jpg', '_'+str(i) + '.jpg'),
                    img[bbox[1]:bbox[3], bbox[0]:bbox[2]])

def IoF(bbox1, bbox2):
    inter_rect_xmin = max(bbox1[0], bbox2[0])
    inter_rect_ymin = max(bbox1[1], bbox2[1])
    inter_rect_xmax = min(bbox1[2], bbox2[2])
    inter_rect_ymax = min(bbox1[3], bbox2[3])
    inter_area = max(0, (inter_rect_xmax - inter_rect_xmin)) * max(0, (inter_rect_ymax - inter_rect_ymin))
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    iou = inter_area / area1
    return iou

def nms_person(bboxes, labels):
    out_bboxes = []
    out_labels = []
    for i in range(len(bboxes)):
        if len(out_bboxes) == 0:
            out_bboxes.append(bboxes[i])
            out_labels.append(labels[i])
        else:
            valued = True
            for j in range(len(out_bboxes)):
                obox = out_bboxes[j]
                box = bboxes[i]
                if IoF([box[0], box[1]/2, box[2], box[3]], obox) > 0.3 or \
                        IoF([obox[0], obox[1]/2, obox[2], obox[3]], box) > 0.3:
                    out_bboxes[j][0] = min(out_bboxes[j][0], box[0])
                    out_bboxes[j][1] = min(out_bboxes[j][1], box[1])
                    out_bboxes[j][2] = max(out_bboxes[j][2], box[2])
                    out_bboxes[j][3] = max(out_bboxes[j][3], box[3])
                    valued = False
                    break
            if valued:
                out_bboxes.append(bboxes[i])
                out_labels.append(labels[i])
    return out_bboxes, out_labels

def prefactor_fence(bboxes, labels, kps, zitais, mohus, body_bboxes, body_labels, clouse_styles, clouse_colors, points):

    #电子围栏
    if points is not None:
        # 电子围栏
        infence_index = []
        polygon = Polygon(points)
        #常见目标围栏
        for i in range(len(labels)):
            box = bboxes[i]
            if polygon.contains(Point([box[0], box[1]])) and polygon.contains(Point([box[2], box[3]])):
                infence_index.append(True)
            else:
                infence_index.append(False)
        bboxes = bboxes[infence_index]
        labels = labels[infence_index]
        kps = kps[infence_index]
        zitais = zitais[infence_index]
        mohus = mohus[infence_index]

        #人体属性围栏
        infence_index = []
        for i in range(len(body_bboxes)):
            box = body_bboxes[i]
            if polygon.contains(Point([box[0], box[1]])) and polygon.contains(Point([box[2], box[3]])):
                infence_index.append(True)
            else:
                infence_index.append(False)
        body_bboxes = body_bboxes[infence_index]
        body_labels = body_labels[infence_index]
        clouse_styles = clouse_styles[infence_index]
        clouse_colors = clouse_colors[infence_index]

    #划分出人脸
    face_index = np.where((labels >= 0) & (labels <= 1))
    face_bboxes = bboxes[face_index]
    face_labels = labels[face_index]
    face_kps = kps[face_index]
    face_mohus = mohus[face_index]
    face_zitais = zitais[face_index]
    #划分出人体
    person_index = np.where((labels >= 2) & (labels <= 3))
    person_bboxes = bboxes[person_index]
    person_labels = labels[person_index]
    person_bboxes, person_labels = nms_person(person_bboxes, person_labels)
    #划分出车辆
    car_index = np.where((labels >= 4) & (labels <= 6))
    car_bboxes = bboxes[car_index]
    car_labels = labels[car_index]
    #划分出宠物
    pet_index = np.where((labels >= 7) & (labels <= 8))
    pet_bboxes = bboxes[pet_index]
    pet_labels = labels[pet_index]
    #划分出下衣
    downclouse_index = np.where((body_labels >= 2)&(body_labels <= 4))
    downclouse_bboxes = body_bboxes[downclouse_index]
    downclouse_labels = body_labels[downclouse_index]
    downclouse_colors = clouse_colors[downclouse_index]
    #划分出上衣
    upclouse_index = np.where((body_labels >= 0)&(body_labels <= 1))
    upclouse_bboxes = body_bboxes[upclouse_index]
    upclouse_labels = body_labels[upclouse_index]
    upclouse_styles = clouse_styles[upclouse_index]
    upclouse_colors = clouse_colors[upclouse_index]
    #划分出其他目标
    otherfactor_index = np.where(body_labels > 4)
    otherfactor_bboxes = body_bboxes[otherfactor_index]
    otherfactor_labels = body_labels[otherfactor_index]
    #划分出人脸属性目标
    glasses_index = np.where(body_labels==6)
    glasses_bboxes = body_bboxes[glasses_index]
    glasses_labels = body_labels[glasses_index]
    #划分出帽子属性目标
    hat_index = np.where(body_labels==8)
    hat_bboxes = body_bboxes[hat_index]
    hat_labels = body_labels[hat_index]

    #属性和人体对应
    upclouse_bboxes_list = []
    upclouse_labels_list = []
    upclouse_styles_list = []
    upclouse_colors_list = []
    downclouse_bboxes_list = []
    downclouse_labels_list = []
    downclouse_colors_list = []
    otherfactor_bboxes_list = []
    otherfactor_labels_list = []
    face_bboxes_list =[]
    face_labels_list = []
    for i in range(len(person_bboxes)):
        upclouse_bboxes_item, upclouse_labels_item, upclouse_styles_item, upclouse_colors_item, downclouse_bboxes_item, downclouse_labels_item, \
        downclouse_colors_item, otherfactor_bboxes_item, otherfactor_labels_item, face_bboxes_item, face_labels_item = [], [], [], [], [], [], [], [], [], [], []
        for j in range(len(upclouse_bboxes)):
            if IoF(upclouse_bboxes[j], person_bboxes[i]) > 0.8:
                upclouse_bboxes_item.append(upclouse_bboxes[j])
                upclouse_labels_item.append(upclouse_labels[j])
                upclouse_styles_item.append(upclouse_styles[j])
                upclouse_colors_item.append(upclouse_colors[j])
                break
        for j in range(len(downclouse_bboxes)):
            if IoF(downclouse_bboxes[j], person_bboxes[i]) > 0.5:
                downclouse_bboxes_item.append(downclouse_bboxes[j])
                downclouse_labels_item.append(downclouse_labels[j])
                downclouse_colors_item.append(downclouse_colors[j])
                break
        for j in range(len(otherfactor_bboxes)):
            if IoF(otherfactor_bboxes[j], person_bboxes[i]) > 0.1:
                otherfactor_bboxes_item.append(otherfactor_bboxes[j])
                otherfactor_labels_item.append(otherfactor_labels[j])
        for j in range(len(face_bboxes)):
            if IoF(face_bboxes[j], person_bboxes[i]) > 0.8:
                face_bboxes_item.append(face_bboxes[j])
                face_labels_item.append(face_labels[j])
                break
        upclouse_bboxes_list.append(upclouse_bboxes_item)
        upclouse_labels_list.append(upclouse_labels_item)
        upclouse_styles_list.append(upclouse_styles_item)
        upclouse_colors_list.append(upclouse_colors_item)
        downclouse_bboxes_list.append(downclouse_bboxes_item)
        downclouse_labels_list.append(downclouse_labels_item)
        downclouse_colors_list.append(downclouse_colors_item)
        otherfactor_bboxes_list.append(otherfactor_bboxes_item)
        otherfactor_labels_list.append(otherfactor_labels_item)
        face_bboxes_list.append(face_bboxes_item)
        face_labels_list.append(face_labels_item)

    #属性和人脸对应
    face_bboxes_list_all = []
    face_labels_list_all = []
    facefactor_bboxes_list_all =[]
    facefactor_labels_list_all = []
    for i in range(len(face_bboxes)):
        facefactor_bboxes_item, facefactor_labels_item = [], []
        for j in range(len(glasses_bboxes)):
            if IoF(glasses_bboxes[j], face_bboxes[i]) > 0.5:
                facefactor_bboxes_item.append(glasses_bboxes[j])
                facefactor_labels_item.append(glasses_labels[j])
                break
        for j in range(len(hat_bboxes)):
            if abs(hat_bboxes[j][3] - face_bboxes[i][1]) < face_bboxes[i][3]-face_bboxes[i][1] and \
                    abs(face_bboxes[i][0]+face_bboxes[i][2]-hat_bboxes[j][0]-hat_bboxes[j][2]) < face_bboxes[i][2]-face_bboxes[i][0]:
                facefactor_bboxes_item.append(hat_bboxes[j])
                facefactor_labels_item.append(hat_labels[j])
                break
        facefactor_bboxes_list_all.append(facefactor_bboxes_item)
        facefactor_labels_list_all.append(facefactor_labels_item)



    return car_bboxes, car_labels, pet_bboxes, pet_labels, person_bboxes, person_labels, upclouse_bboxes_list, upclouse_labels_list, upclouse_colors_list, upclouse_styles_list,\
           downclouse_bboxes_list, downclouse_labels_list, downclouse_colors_list, otherfactor_bboxes_list, otherfactor_labels_list, face_bboxes_list, face_labels_list, face_bboxes,\
           face_labels, face_kps, face_zitais, face_mohus, facefactor_bboxes_list_all, facefactor_labels_list_all
