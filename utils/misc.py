import tensorflow as tf
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


@tf.function
def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    """네트워크의 회귀 위치 결과를 (center_x, center_y, h, w) 형식의 box로 변환하는 과정
     변환 :
         $$ predicted :_center * center_variance = frac {real_center - prior_center} {prior_hw}$$
         $$ exp (예측_hw * size_variance) = frac {real_hw} {prior_hw} $$

     Args :
         locations (batch_size, num_priors, 4) : SSD의 회귀 출력. 출력도 포함
         priors (num_priors, 4) 또는 (batch_size / 1, num_priors, 4) : priors box
         center_variance : 중심 스케일을 변경 상수
         size_variance : 크기 스케일 변경 상수
     Returns:
         bbox : priors : [[center_x, center_y, h, w]]
             이미지 크기에 상대적입니다.
     """
    if tf.rank(priors) + 1 == tf.rank(locations):
        priors = tf.expand_dims(priors, 0)
    return tf.concat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        tf.math.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=tf.rank(locations) - 1)


@tf.function
def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    if tf.rank(center_form_priors) + 1 == tf.rank(center_form_boxes):
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
    return tf.concat([locations[..., :2] - locations[..., 2:] / 2,
                      locations[..., :2] + locations[..., 2:] / 2], tf.rank(locations) - 1)


@tf.function
def corner_form_to_center_form(boxes):
    return tf.concat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]
    ], tf.rank(boxes) - 1)


# 작업중
# def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
#     """
#     Args:
#         box_scores (N, 5): 코너 형식의 box와 확률값
#         iou_threshold: IoU 임계값
#         top_k: 값 유지    If k <= 0, 모든 값 유지
#         candidate_size: 가장 높은 점수를 가지는 후보 경계만 사용.
#     Returns:
#          picked: bbox index 리스트
#     """
#     scores = box_scores[:, 4]
#     boxes = box_scores[:, :-2]
#     picked = []
#     indexes = np.argsort(scores)[::-1]
#     indexes = indexes[:candidate_size]
#
#     while len(indexes) > 0:
#         current = indexes[0]
#         picked.append(current)
#         if 0 < top_k == len(picked) or len(indexes) == 1:
#             break
#         current_box = boxes[current, :]
#         indexes = indexes[1:]
#         rest_boxes = boxes[indexes, :]
#         iou = iou_of(
#             rest_boxes,
#             np.expand_dims(current_box, axis=0),
#         ).numpy()
#         indexes = indexes[iou <= iou_threshold]
#
#     return box_scores[picked, :]


