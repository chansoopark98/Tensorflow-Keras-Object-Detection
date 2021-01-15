import tensorflow as tf
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np 

""" CSNet : misc.py
    bbox 및 predict, eval에 필요한 fucntion 모음
"""
# PASCAL VOC 클래스(20)
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'] 

@tf.function
def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    """ 회귀 위치 결과를 (center_x, center_y, h, w) 형식의 상자로 변환.

    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.

    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, h, w]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    if tf.rank(priors) + 1 == tf.rank(locations):
        priors = tf.expand_dims(priors, 0)
    return tf.concat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        tf.math.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=tf.rank(locations) - 1)

@tf.function
def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    if tf.rank(center_form_priors) + 1 == tf.rank(center_form_boxes):
        center_form_priors = tf.expand_dims(center_form_priors, 0)
    return tf.concat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        tf.math.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], axis=tf.rank(center_form_boxes) - 1)

@tf.function(experimental_relax_shapes=True)
def area_of(left_top, right_bottom):
    """bbox 사각형 면적 계산
    Args:
        left_top (N, 2): 좌측 상단 위치값
        right_bottom (N, 2): 우측 하단 위치값
    Return:
        사각형 면적 값
    """
    hw = tf.clip_by_value(right_bottom - left_top, 0.0, 10000)
    return hw[..., 0] * hw[..., 1]

# 자카드오버랩(iou 계산)
@tf.function(experimental_relax_shapes=True)
def iou_of(gtBoxes, predictBoxes, eps=1e-5):
    """자카드 오버랩(IoU) 계산
    Args:
        gtBoxes (N, 4): groundtruth 좌표값
        predictBoxes (N or 1, 4): 예측결과 좌표값
        eps: 분모값이 0 방지 상수 값
    Returns:
        IoU 계산 결과
    """
    overlap_left_top = tf.maximum(gtBoxes[..., :2], predictBoxes[..., :2])
    overlap_right_bottom = tf.minimum(gtBoxes[..., 2:], predictBoxes[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(gtBoxes[..., :2], gtBoxes[..., 2:])
    area1 = area_of(predictBoxes[..., :2], predictBoxes[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

@tf.function
def center_form_to_corner_form(locations):
    return tf.concat([locations[..., :2] - locations[..., 2:]/2,
                     locations[..., :2] + locations[..., 2:]/2], tf.rank(locations) - 1)

@tf.function
def corner_form_to_center_form(boxes):
    return tf.concat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
         boxes[..., 2:] - boxes[..., :2]
    ], tf.rank(boxes) - 1)

#
# def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
#     """
#     Args:
#         box_scores (N, 5): boxes in corner-form and probabilities.
#         iou_threshold: intersection over union threshold.
#         top_k: keep top_k results. If k <= 0, keep all the results.
#         candidate_size: only consider the candidates with the highest scores.
#     Returns:
#          picked: a list of indexes of the kept boxes
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
#         # print(indexes.shape)
#         # print(iou<=iou_threshold)
#         indexes = indexes[iou <= iou_threshold]
#
#     return box_scores[picked, :]

# matplotlib bbox drawing 코드
# def draw_bboxes(bboxes, ax, color='red', labels=None, IMAGE_SIZE=[300, 300]):
#     # image = (im - np.min(im))/np.ptp(im)
#     # print(image.shape)
#     if np.max(bboxes) < 10:
#         bboxes[:, [0,2]] = bboxes[:, [0,2]]*IMAGE_SIZE[1]
#         bboxes[:, [1,3]] = bboxes[:, [1,3]]*IMAGE_SIZE[0]
#     for i, bbox in enumerate(bboxes):
#         rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor=color, facecolor='none')
#         # ax.add_patch(rect)
#         ax.add_artist(rect)
#         # print(int(bbox[-1]))
#         if labels is not None:
#             ax.text(bbox[0]+0.5,bbox[1]+0.5, CLASSES[int(labels[i] - 1)],  fontsize=20,
#                 horizontalalignment='left', verticalalignment='top', bbox=dict(facecolor=color, alpha=0.4))