import sys
from typing import List
import itertools
import collections
from utils.misc import *

"""CSNET Prior Box 생성 (= Default box)
중심, 높이 및 너비값 반환
 사전의 중심, 높이 및 너비를 반환합니다. 값은 이미지 크기에 상대적입니다.
 Args :
     specs : 이전 상자의 크기 모양에 대한 SSDSpecs
         spec = [
             Spec (38, 8, BoxSizes (30, 60), [2]),
             Spec (19, 16, BoxSizes (60, 111), [2, 3]),
             Spec (10, 32, BoxSizes (111, 162), [2, 3]),
             Spec (5, 64, BoxSizes (162, 213), [2, 3]),
             Spec (3, 100, BoxSizes (213, 264), [2]),
             Spec (1, 300, BoxSizes (264, 315), [2])
         ]
     image_size : 이미지 크기.
     clamp : 참이면 값을 [0.0, 1.0] 사이로 고정합니다.
 returns:
     priors (num_priors, 4) : [[center_x, center_y, w, h]] priors box
 """

BoxSizes = collections.namedtuple('Boxsizes', ['min', 'max'])
Spec = collections.namedtuple('Spec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])


def create_priors_boxes(specs: List[Spec], image_size, clamp=True):
    priors = []
    for spec in specs:
        # specs
        # index 0 >> size-(48,438) shrinkage-8 CSNet
        scale = image_size / spec.shrinkage
        for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
            x_center = (i + 0.5) / scale
            y_center = (j + 0.5) / scale

            # 작은 bbox
            size = spec.box_sizes.min
            h = w = size / image_size
            priors.append([
                x_center,
                y_center,
                w,
                h
            ])

            # 작은 bbox 높이, 너비 비율 변경
            size = spec.box_sizes.min
            h = w = size / image_size
            if spec.aspect_ratios :
                for ratio in spec.aspect_ratios:
                    ratio = np.sqrt(ratio)
                    priors.append([
                        x_center,
                        y_center,
                        w * ratio,
                        h / ratio
                    ])
                    priors.append([
                        x_center,
                        y_center,
                        w / ratio,
                        h * ratio
                    ])




    # priors > shape(Batch, 13792)
    # 2차원 배열이고 각 배열마다 4개씩 존재(x_center, y_center, w, h) * 13792
    priors = np.array(priors, dtype=np.float32)

    print(priors)
    if clamp:
        np.clip(priors, 0.0, 1.0, out=priors)
    return tf.convert_to_tensor(priors)



@tf.function
def assign_gt2_priors(gt_boxes, gt_labels, corner_form_priors,
                      iou_threshold=0.45):
    """Ground truth <-> priors(default box) 할당
    Args:
        gt_boxes (num_targets, 4): ground truth boxes
        gt_labels (num_targets): ground truth class labels
        priors (num_priors, 4): priors
    Returns:
        boxes (num_priors, 4): gt 박스
        labels (num_priors): gt 라벨
    """
    # size: num_priors x num_targets
    ious = iou_of(tf.expand_dims(gt_boxes, axis=0), tf.expand_dims(corner_form_priors, axis=1))

    # size: num_priors
    best_target_per_prior = tf.math.reduce_max(ious, axis=1)
    best_target_per_prior_index = tf.math.argmax(ious, axis=1)
    # size: num_targets
    best_prior_per_target = tf.math.reduce_max(ious, axis=0)
    best_prior_per_target_index = tf.math.argmax(ious, axis=0)

    targets = tf.range(tf.shape(best_prior_per_target_index)[0], dtype='int64')

    best_target_per_prior_index = tf.tensor_scatter_nd_update(best_target_per_prior_index,
                                                              tf.expand_dims(best_prior_per_target_index, 1), targets)
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior = tf.tensor_scatter_nd_update(best_target_per_prior,
                                                        tf.expand_dims(best_prior_per_target_index, 1),
                                                        tf.ones_like(best_prior_per_target_index,
                                                                     dtype=tf.float32) * 2.0)
    # size: num_priors
    labels = tf.gather(gt_labels, best_target_per_prior_index)

    labels = tf.where(tf.less(best_target_per_prior, iou_threshold), tf.constant(0, dtype='int64'), labels)

    # 라벨이 임계값을 넘기 않는 경우 background(배경) 처리
    boxes = tf.gather(gt_boxes, best_target_per_prior_index)

    return boxes, labels


class MatchingPriors(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = tf.convert_to_tensor(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = tf.convert_to_tensor(gt_labels)
        boxes, labels = assign_gt2_priors(gt_boxes, gt_labels, self.corner_form_priors, self.iou_threshold)
        boxes = corner_form_to_center_form(boxes)
        locations = convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)

        return locations, labels


"""" 오리지날 priors 코드"""
# import sys
# from typing import List
# import itertools
# import collections
# from utils.misc import *
#
# """CSNET Prior Box 생성 (= Default box)
# 중심, 높이 및 너비값 반환
#  사전의 중심, 높이 및 너비를 반환합니다. 값은 이미지 크기에 상대적입니다.
#  Args :
#      specs : 이전 상자의 크기 모양에 대한 SSDSpecs
#          spec = [
#              Spec (38, 8, BoxSizes (30, 60), [2]),
#              Spec (19, 16, BoxSizes (60, 111), [2, 3]),
#              Spec (10, 32, BoxSizes (111, 162), [2, 3]),
#              Spec (5, 64, BoxSizes (162, 213), [2, 3]),
#              Spec (3, 100, BoxSizes (213, 264), [2]),
#              Spec (1, 300, BoxSizes (264, 315), [2])
#          ]
#      image_size : 이미지 크기.
#      clamp : 참이면 값을 [0.0, 1.0] 사이로 고정합니다.
#  returns:
#      priors (num_priors, 4) : [[center_x, center_y, w, h]] priors box
#  """
#
# BoxSizes = collections.namedtuple('Boxsizes', ['min', 'max'])
# Spec = collections.namedtuple('Spec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])
#
#
# def create_priors_boxes(specs: List[Spec], image_size, clamp=True):
#     priors = []
#     for spec in specs:
#         # specs
#         # index 0 >> size-(48,438) shrinkage-8 CSNet
#         scale = image_size / spec.shrinkage
#         for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
#             x_center = (i + 0.5) / scale
#             y_center = (j + 0.5) / scale
#
#             # 작은 bbox
#             size = spec.box_sizes.min
#             h = w = size / image_size
#             priors.append([
#                 x_center,
#                 y_center,
#                 w,
#                 h
#             ])
#
#             # 큰 bbox
#             size = np.sqrt(spec.box_sizes.max * spec.box_sizes.min)
#             h = w = size / image_size
#             priors.append([
#                 x_center,
#                 y_center,
#                 w,
#                 h
#             ])
#
#             # 작은 bbox 높이, 너비 비율 변경
#             size = spec.box_sizes.min
#             h = w = size / image_size
#             if spec.aspect_ratios :
#                 for ratio in spec.aspect_ratios:
#                     ratio = np.sqrt(ratio)
#                     priors.append([
#                         x_center,
#                         y_center,
#                         w * ratio,
#                         h / ratio
#                     ])
#                     priors.append([
#                         x_center,
#                         y_center,
#                         w / ratio,
#                         h * ratio
#                     ])
#
#
#
#
#     # priors > shape(Batch, 13792)
#     # 2차원 배열이고 각 배열마다 4개씩 존재(x_center, y_center, w, h) * 13792
#     priors = np.array(priors, dtype=np.float32)
#
#     print(priors)
#     if clamp:
#         np.clip(priors, 0.0, 1.0, out=priors)
#     return tf.convert_to_tensor(priors)
#
#
#
# @tf.function
# def assign_gt2_priors(gt_boxes, gt_labels, corner_form_priors,
#                       iou_threshold=0.45):
#     """Ground truth <-> priors(default box) 할당
#     Args:
#         gt_boxes (num_targets, 4): ground truth boxes
#         gt_labels (num_targets): ground truth class labels
#         priors (num_priors, 4): priors
#     Returns:
#         boxes (num_priors, 4): gt 박스
#         labels (num_priors): gt 라벨
#     """
#     # size: num_priors x num_targets
#     ious = iou_of(tf.expand_dims(gt_boxes, axis=0), tf.expand_dims(corner_form_priors, axis=1))
#
#     # size: num_priors
#     best_target_per_prior = tf.math.reduce_max(ious, axis=1)
#     best_target_per_prior_index = tf.math.argmax(ious, axis=1)
#     # size: num_targets
#     best_prior_per_target = tf.math.reduce_max(ious, axis=0)
#     best_prior_per_target_index = tf.math.argmax(ious, axis=0)
#
#     targets = tf.range(tf.shape(best_prior_per_target_index)[0], dtype='int64')
#
#     best_target_per_prior_index = tf.tensor_scatter_nd_update(best_target_per_prior_index,
#                                                               tf.expand_dims(best_prior_per_target_index, 1), targets)
#     # 2.0 is used to make sure every target has a prior assigned
#     best_target_per_prior = tf.tensor_scatter_nd_update(best_target_per_prior,
#                                                         tf.expand_dims(best_prior_per_target_index, 1),
#                                                         tf.ones_like(best_prior_per_target_index,
#                                                                      dtype=tf.float32) * 2.0)
#     # size: num_priors
#     labels = tf.gather(gt_labels, best_target_per_prior_index)
#
#     labels = tf.where(tf.less(best_target_per_prior, iou_threshold), tf.constant(0, dtype='int64'), labels)
#
#     # 라벨이 임계값을 넘기 않는 경우 background(배경) 처리
#     boxes = tf.gather(gt_boxes, best_target_per_prior_index)
#
#     return boxes, labels
#
#
# class MatchingPriors(object):
#     def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
#         self.center_form_priors = center_form_priors
#         self.corner_form_priors = center_form_to_corner_form(center_form_priors)
#         self.center_variance = center_variance
#         self.size_variance = size_variance
#         self.iou_threshold = iou_threshold
#
#     def __call__(self, gt_boxes, gt_labels):
#         if type(gt_boxes) is np.ndarray:
#             gt_boxes = tf.convert_to_tensor(gt_boxes)
#         if type(gt_labels) is np.ndarray:
#             gt_labels = tf.convert_to_tensor(gt_labels)
#         boxes, labels = assign_gt2_priors(gt_boxes, gt_labels, self.corner_form_priors, self.iou_threshold)
#         boxes = corner_form_to_center_form(boxes)
#         locations = convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
#
#         return locations, labels