import tensorflow as tf
import cv2
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
from collections import namedtuple
from typing import List
import itertools
import collections
#import tflite_runtime.interpreter as tflite
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

Predictions = namedtuple('Prediction', ('boxes', 'scores', 'labels'))

BoxSizes = collections.namedtuple('Boxsizes', ['min', 'max'])
Spec = collections.namedtuple('Spec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])

iou_threshold = 0.5 # 0.5
center_variance = 0.1 # 0.1
size_variance = 0.2 # 0.2


def rgb2bgr(tpl):
    return (tpl[2], tpl[1], tpl[0])
Label = namedtuple('Label', ['name', 'color'])

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
def draw_bounding(img , bboxes, labels, img_size):
    # resizing 작업
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
        cv2.putText(img_box, CLASSES[int(labels[i]-1)], (xmin + 5, ymin - 5), font, 0.5,
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
    if tf.rank(priors) + 1 == tf.rank(locations):
        priors = tf.expand_dims(priors, 0)
    return tf.concat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        tf.math.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=tf.rank(locations) - 1)

@tf.function
def center_form_to_corner_form(locations):
    output = tf.concat([locations[..., :2] - locations[..., 2:] / 2,
                        locations[..., :2] + locations[..., 2:] / 2], tf.rank(locations) - 1)

    return output

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

@tf.function
def corner_form_to_center_form(boxes):
    return tf.concat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]
    ], tf.rank(boxes) - 1)

@tf.function
def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    if tf.rank(center_form_priors) + 1 == tf.rank(center_form_boxes):
        center_form_priors = tf.expand_dims(center_form_priors, 0)

    return tf.concat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        tf.math.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], axis=tf.rank(center_form_boxes) - 1)


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


            # # 큰 bbox
            # size = np.sqrt(spec.box_sizes.max * spec.box_sizes.min)
            # h = w = size / image_size
            # priors.append([
            #     x_center,
            #     y_center,
            #     w,
            #     h
            # ])


            # 작은 bbox 높이, 너비 비율 변경
            #size = spec.box_sizes.min 기존
            size = np.sqrt(spec.box_sizes.max * spec.box_sizes.min)
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

    if clamp:
        np.clip(priors, 0.0, 1.0, out=priors)
    return tf.convert_to_tensor(priors)

specs = [
            Spec(28, 8, BoxSizes(11, 22), [2]), # 0.05 / 0.1
            Spec(14, 16, BoxSizes(23, 45), [2]), # 0.1 / 0.2
            Spec(7, 32, BoxSizes(56, 90), [2]), # 0.25 / 0.4
            Spec(4, 64, BoxSizes(90, 134), [2]), # 0.4 / 0.6
            Spec(2, 112, BoxSizes(134, 168), [2]), # 0.6 / 0.75
            Spec(1, 224, BoxSizes(179, 235), [2]) # 0.8 / 1.05
        ]
priors = create_priors_boxes(specs, 224)


target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

TFLITE_FILE_PATH = 'new_tflite_model.tflite'


interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


import time
while True:
    ret, frame = capture.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    start = time.perf_counter_ns()
    input = tf.convert_to_tensor(frame)

    # 이미지 리사이징
    input = tf.image.resize(input, [224, 224])
    input = preprocess_input(input, mode='torch')
    input = tf.expand_dims(input, axis=0)

    duration = (time.perf_counter_ns() - start)
    print(f"전처리 과정 : {duration // 1000000}ms.")
    # 텐서 변환



    # Test the model on random input data.
    #input_shape = input_details[0]['shape']

    start = time.perf_counter_ns()

    interpreter.set_tensor(input_details[0]['index'], input)

    interpreter.invoke()
    duration = (time.perf_counter_ns() - start)
    print(f"추론 과정 : {duration // 1000000}ms.")
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])




    start = time.perf_counter_ns()
    predictions = post_process(output_data, target_transform, top_k=25, classes=21, confidence_threshold=0.4)

    pred_boxes, pred_scores, pred_labels = predictions[0]
    if pred_boxes.size > 0:
        draw_bounding(frame, pred_boxes, labels=pred_labels, img_size=frame.shape[:2])
    duration = (time.perf_counter_ns() - start)
    print(f"포스트 프로세싱 과정 : {duration // 1000000}ms.")

    #cv2.imshow("VideoFrame", frame)
    if cv2.waitKey(1) > 0:
        break

capture.release()
cv2.destroyAllWindows()