import tensorflow as tf
import numpy as np
from collections import namedtuple
from utils.misc import *

Predictions = namedtuple('Prediction', ('boxes', 'scores', 'labels'))

def batched_nms(boxes, scores, idxs, iou_threshold, top_k=100):
    """
    :Args(bbox, scores, idxs, iou_threshold)
    NMS
    각 인덱스는 각 category에 매핑

    boxes : Tensor[N, 4]
        NMS가 적용될 bbox list
        shape = (x1,y1, x2, y2)
    scores : Tensor[N]
        각 박스별  confidence score
    idxs : Tensor[N]
        category 인덱스
    iou_threshold : float
        임계값

    :return Tensor
    """

    if tf.size(boxes) == 0:
        return tf.convert_to_tensor([], dtype=tf.int32)

    max_coordinate = tf.reduce_max(boxes)
    offsets = idxs * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = tf.image.non_max_suppression(boxes_for_nms, scores, top_k, iou_threshold) # 기존
    # keep, selected_scores = tf.image.non_max_suppression_with_scores(boxes_for_nms, scores, top_k, iou_threshold, soft_nms_sigma=0.5) # 기존
    # soft nms일 경우 selected_socres 추가

    return keep

# #작업중
# def hard_nms(box_scores, iou_threshold, top_k=100, candidate_size=200):
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



def post_process(detections, target_transform, confidence_threshold=0.01, top_k=100, iou_threshold=0.5, classes=21):
    batch_boxes = detections[:, :, classes:]
    if not tf.is_tensor(batch_boxes):
        batch_boxes = tf.convert_to_tensor(batch_boxes)
    batch_scores = tf.nn.softmax(detections[:, :, :classes], axis=2)

    batch_boxes = convert_locations_to_boxes(batch_boxes, target_transform.center_form_priors,
                                             target_transform.center_variance, target_transform.size_variance)
    batch_boxes = center_form_to_corner_form(batch_boxes)

    batch_size = tf.shape(batch_scores)[0]
    results = []
    for image_id in range(batch_size):
        scores, boxes = batch_scores[image_id], batch_boxes[image_id]  # (N, #CLS) (N, 4)

        num_boxes = tf.shape(scores)[0]
        num_classes = tf.shape(scores)[1]
        boxes = tf.reshape(boxes, [num_boxes, 1, 4])
        boxes = tf.broadcast_to(boxes, [num_boxes, num_classes, 4])
        labels = tf.range(num_classes, dtype=tf.float32)
        labels = tf.reshape(labels, [1, num_classes])
        labels = tf.broadcast_to(labels, tf.shape(scores))

        # 배경 라벨이 있는 예측값 제거
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]

        # 모든 클래스 예측을 별도의 인스턴스로 만들어 모든 것을 일괄 처리 과정
        boxes = tf.reshape(boxes, [-1, 4])
        scores = tf.reshape(scores, [-1])
        labels = tf.reshape(labels, [-1])

        # confidence  점수가 낮은 predict bbox 제거
        low_scoring_mask = scores > confidence_threshold
        boxes, scores, labels = tf.boolean_mask(boxes, low_scoring_mask), tf.boolean_mask(scores, low_scoring_mask), tf.boolean_mask(labels, low_scoring_mask)

        keep  = batched_nms(boxes, scores, labels, iou_threshold, top_k)

        boxes, scores, labels = tf.gather(boxes, keep), tf.gather(scores, keep), tf.gather(labels, keep)

        # test soft-nms
        # keep, selected_scores = batched_nms(boxes, scores, labels, iou_threshold, top_k)
        # scores = selected_scores
        # boxes, labels = tf.gather(boxes, keep), tf.gather(labels, keep)

        results.append(Predictions(boxes.numpy(), scores.numpy(), labels.numpy()))

    return results