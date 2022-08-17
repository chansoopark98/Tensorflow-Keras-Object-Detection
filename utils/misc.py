import tensorflow as tf
import numpy as np
import cv2
from collections import namedtuple

TEST_CLASSES = ['outdoor', 'indoor']

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

COCO_CLASSES = ['person',
                'bicycle',
                'car',
                'motorcycle',
                'airplane',
                'bus',
                'train',
                'truck',
                'boat',
                'traffic light',
                'fire hydrant',
                'stop sign',
                'parking meter',
                'bench',
                'bird',
                'cat',
                'dog',
                'horse',
                'sheep',
                'cow',
                'elephant',
                'bear',
                'zebra',
                'giraffe',
                'backpack',
                'umbrella',
                'handbag',
                'tie',
                'suitcase',
                'frisbee',
                'skis',
                'snowboard',
                'sports ball',
                'kite',
                'baseball bat',
                'baseball glove',
                'skateboard',
                'surfboard',
                'tennis racket',
                'bottle',
                'wine glass',
                'cup',
                'fork',
                'knife',
                'spoon',
                'bowl',
                'banana',
                'apple',
                'sandwich',
                'orange',
                'broccoli',
                'carrot',
                'hot dog',
                'pizza',
                'donut',
                'cake',
                'chair',
                'couch',
                'potted plant',
                'bed',
                'dining table',
                'toilet',
                'tv',
                'laptop',
                'mouse',
                'remote',
                'keyboard',
                'cell phone',
                'microwave',
                'oven',
                'toaster',
                'sink',
                'refrigerator',
                'book',
                'clock',
                'vase',
                'scissors',
                'teddy bear',
                'hair drier',
                'toothbrush']


def rgb2bgr(tpl):
    return (tpl[2], tpl[1], tpl[0])


Label = namedtuple('Label', ['name', 'color'])


def voc_color_map(index):
    label_defs = [
        Label('aeroplane', rgb2bgr((0, 0, 0))),
        Label('bicycle', rgb2bgr((111, 74, 0))),
        Label('bird', rgb2bgr((81, 0, 81))),
        Label('boat', rgb2bgr((128, 64, 128))),
        Label('bottle', rgb2bgr((244, 35, 232))),
        Label('bus', rgb2bgr((230, 150, 140))),
        Label('car', rgb2bgr((70, 70, 70))),
        Label('cat', rgb2bgr((102, 102, 156))),
        Label('chair', rgb2bgr((190, 153, 153))),
        Label('cow', rgb2bgr((150, 120, 90))),
        Label('diningtable', rgb2bgr((153, 153, 153))),
        Label('dog', rgb2bgr((250, 170, 30))),
        Label('horse', rgb2bgr((220, 220, 0))),
        Label('motorbike', rgb2bgr((107, 142, 35))),
        Label('person', rgb2bgr((52, 151, 52))),
        Label('pottedplant', rgb2bgr((70, 130, 180))),
        Label('sheep', rgb2bgr((220, 20, 60))),
        Label('sofa', rgb2bgr((0, 0, 142))),
        Label('train', rgb2bgr((0, 0, 230))),
        Label('tvmonitor', rgb2bgr((119, 11, 32)))]
    return label_defs[index]

def coco_color_map(index):
    label_defs = [
        Label('aeroplane', rgb2bgr((0, 0, 0))),
        Label('bicycle', rgb2bgr((111, 74, 0))),
        Label('bird', rgb2bgr((81, 0, 81))),
        Label('boat', rgb2bgr((128, 64, 128))),
        Label('bottle', rgb2bgr((244, 35, 232))),
        Label('bus', rgb2bgr((230, 150, 140))),
        Label('car', rgb2bgr((70, 70, 70))),
        Label('cat', rgb2bgr((102, 102, 156))),
        Label('chair', rgb2bgr((190, 153, 153))),
        Label('cow', rgb2bgr((150, 120, 90))),
        Label('diningtable', rgb2bgr((153, 153, 153))),
        Label('dog', rgb2bgr((250, 170, 30))),
        Label('horse', rgb2bgr((220, 220, 0))),
        Label('motorbike', rgb2bgr((107, 142, 35))),
        Label('person', rgb2bgr((52, 151, 52))),
        Label('pottedplant', rgb2bgr((70, 130, 180))),
        Label('sheep', rgb2bgr((220, 20, 60))),
        Label('sofa', rgb2bgr((0, 0, 142))),
        Label('train', rgb2bgr((0, 0, 230))),
        Label('tvmonitor', rgb2bgr((119, 11, 32))),
        Label('aeroplane', rgb2bgr((0, 0, 0))),
        Label('bicycle', rgb2bgr((111, 74, 0))),
        Label('bird', rgb2bgr((81, 0, 81))),
        Label('boat', rgb2bgr((128, 64, 128))),
        Label('bottle', rgb2bgr((244, 35, 232))),
        Label('bus', rgb2bgr((230, 150, 140))),
        Label('car', rgb2bgr((70, 70, 70))),
        Label('cat', rgb2bgr((102, 102, 156))),
        Label('chair', rgb2bgr((190, 153, 153))),
        Label('cow', rgb2bgr((150, 120, 90))),
        Label('diningtable', rgb2bgr((153, 153, 153))),
        Label('dog', rgb2bgr((250, 170, 30))),
        Label('horse', rgb2bgr((220, 220, 0))),
        Label('motorbike', rgb2bgr((107, 142, 35))),
        Label('person', rgb2bgr((52, 151, 52))),
        Label('pottedplant', rgb2bgr((70, 130, 180))),
        Label('sheep', rgb2bgr((220, 20, 60))),
        Label('sofa', rgb2bgr((0, 0, 142))),
        Label('train', rgb2bgr((0, 0, 230))),
        Label('tvmonitor', rgb2bgr((119, 11, 32))),
        Label('aeroplane', rgb2bgr((0, 0, 0))),
        Label('bicycle', rgb2bgr((111, 74, 0))),
        Label('bird', rgb2bgr((81, 0, 81))),
        Label('boat', rgb2bgr((128, 64, 128))),
        Label('bottle', rgb2bgr((244, 35, 232))),
        Label('bus', rgb2bgr((230, 150, 140))),
        Label('car', rgb2bgr((70, 70, 70))),
        Label('cat', rgb2bgr((102, 102, 156))),
        Label('chair', rgb2bgr((190, 153, 153))),
        Label('cow', rgb2bgr((150, 120, 90))),
        Label('diningtable', rgb2bgr((153, 153, 153))),
        Label('dog', rgb2bgr((250, 170, 30))),
        Label('horse', rgb2bgr((220, 220, 0))),
        Label('motorbike', rgb2bgr((107, 142, 35))),
        Label('person', rgb2bgr((52, 151, 52))),
        Label('pottedplant', rgb2bgr((70, 130, 180))),
        Label('sheep', rgb2bgr((220, 20, 60))),
        Label('sofa', rgb2bgr((0, 0, 142))),
        Label('train', rgb2bgr((0, 0, 230))),
        Label('tvmonitor', rgb2bgr((119, 11, 32))),
        Label('aeroplane', rgb2bgr((0, 0, 0))),
        Label('bicycle', rgb2bgr((111, 74, 0))),
        Label('bird', rgb2bgr((81, 0, 81))),
        Label('boat', rgb2bgr((128, 64, 128))),
        Label('bottle', rgb2bgr((244, 35, 232))),
        Label('bus', rgb2bgr((230, 150, 140))),
        Label('car', rgb2bgr((70, 70, 70))),
        Label('cat', rgb2bgr((102, 102, 156))),
        Label('chair', rgb2bgr((190, 153, 153))),
        Label('cow', rgb2bgr((150, 120, 90))),
        Label('diningtable', rgb2bgr((153, 153, 153))),
        Label('dog', rgb2bgr((250, 170, 30))),
        Label('horse', rgb2bgr((220, 220, 0))),
        Label('motorbike', rgb2bgr((107, 142, 35))),
        Label('person', rgb2bgr((52, 151, 52))),
        Label('pottedplant', rgb2bgr((70, 130, 180))),
        Label('sheep', rgb2bgr((220, 20, 60))),
        Label('sofa', rgb2bgr((0, 0, 142))),
        Label('train', rgb2bgr((0, 0, 230))),
        Label('tvmonitor', rgb2bgr((119, 11, 32)))
    ]
    return label_defs[index]

def draw_bounding(img , bboxes, labels, scores, img_size, label_list):
    if np.max(bboxes) < 10:
        bboxes[:, [0,2]] = bboxes[:, [0,2]]*img_size[1]
        bboxes[:, [1,3]] = bboxes[:, [1,3]]*img_size[0]
    
    for i, bbox in enumerate(bboxes):
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])
        img_box = np.copy(img)
        _, color = coco_color_map(int(labels[i] - 1))
        cv2.rectangle(img_box, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.rectangle(img_box, (xmin - 1, ymin), (xmax + 1, ymin - 20), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if scores is not None:
            cv2.putText(img_box, label_list[int(labels[i]-1)] + '_' + str(round(scores[i],2)), (xmin + 5, ymin - 5), font, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(img_box, label_list[int(labels[i]-1)], (xmin + 5, ymin - 5), font, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        alpha = 0.8
        cv2.addWeighted(img_box, alpha, img, 1. - alpha, 0, img)


@tf.function
def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    """네트워크의 회귀 위치 결과를 (center_x, center_y, h, w) 형식의 box로 변환하는 과정
     변환 :
         $$ predicted :_center * center_variance = frac {real_center - prior_center} {prior_hw}$$
         $$ exp (예측_hw * size_variance) = frac {real_hw} {prior_hw} $$

     Args :
         locations (batch_size, num_priors, 4) : 네트워크의 회귀 출력. 출력도 포함
         priors (num_priors, 4) 또는 (batch_size / 1, num_priors, 4) : priors box
         center_variance : 중심 스케일을 변경 상수
         size_variance : 크기 스케일 변경 상수
     Returns:
         bbox : priors : [[center_x, center_y, h, w]]
             이미지 크기에 상대적입니다.
     """
    
    # if tf.rank(priors) + 1 == tf.rank(locations):
        # frozen graph 만들때
        # priors rank = 2, locations = 3
        # print('bigger convert_locations_to_boxes')
        
    priors = tf.expand_dims(priors, 0)
    
    
    return tf.concat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        tf.math.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=tf.rank(locations) - 1)


@tf.function
def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors_rank = tf.rank(center_form_priors)
    # boxes_rank = tf.rank(center_form_boxes)
    
    if tf.rank(center_form_priors) + 1 == tf.rank(center_form_boxes):
        print('bigger convert_boxes_to_locations')
        center_form_priors = tf.expand_dims(center_form_priors, 0)

    return tf.concat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        tf.math.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], axis=tf.rank(center_form_boxes) - 1)


# experimental_relax_shapes=True인 경우 인스턴스 객체를 생성할 때 작은 graph를 생성할 수 있게 함 (XLA 타입으로 컴파일)
# 참고 자료 : https://www.tensorflow.org/xla?hl=ko
@tf.function(experimental_relax_shapes=True)
def area_of(left_top, right_bottom):
    """bbox 좌표값 (좌상단, 우하단)으로 사각형 넓이 계산.
    Args:
        left_top (N, 2): left 좌상단 좌표값.
        right_bottom (N, 2): 우하단 좌표값.
    Returns:
        area (N): 사각형 넓이.
    """

    hw = tf.clip_by_value(right_bottom - left_top, 0.0, 10000)
    return hw[..., 0] * hw[..., 1]


@tf.function(experimental_relax_shapes=True)
def iou_of(boxes0, boxes1, eps=1e-5):
    """두 bbox간 iou 계산.
    Args:
        boxes0 (N, 4): ground truth boxes 좌표값.
        boxes1 (N or 1, 4): predicted boxes 좌표값.
        eps: 0으로 치환되는 것을 막기위한 엡실론 상수값 .
    Returns:
        iou (N): IoU 값.
    """
    overlap_left_top = tf.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = tf.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

@tf.function
def center_form_to_corner_form(locations):
    output = tf.concat([locations[..., :2] - locations[..., 2:] / 2,
                        locations[..., :2] + locations[..., 2:] / 2], tf.rank(locations) - 1)
    return output

@tf.function
def corner_form_to_center_form(boxes):
    return tf.concat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]
    ], tf.rank(boxes) - 1)
