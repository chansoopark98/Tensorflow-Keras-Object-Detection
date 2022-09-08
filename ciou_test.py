import numpy as np
import math
import tensorflow.keras.backend as K
import tensorflow as tf

def unstack_bbox(bbox):
    x_min = bbox[:,1]
    y_min = bbox[:,0]
    x_max = bbox[:,3]
    y_max = bbox[:,2]
    bbox = tf.stack([x_min, y_min, x_max, y_max], axis=1)
    return bbox

def get_bbox_area(bbox):
    # xmin, ymin, xmax, ymax
    return (bbox[:, 2] - bbox[:, 0 ]) * (bbox[:, 3] - bbox[:, 1])

def bbox_iou(boxes1, boxes2):
    # ymin, xmin, ymax, xmax - > xmin, ymin, xmax, ymax
    boxes1 = unstack_bbox(boxes1)
    boxes2 = unstack_bbox(boxes2)
    
    boxes1_area = get_bbox_area(boxes1)
    boxes2_area = get_bbox_area(boxes2)

    # coordinates of intersection
    top_left = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    bottom_right = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection_xy = tf.maximum(bottom_right - top_left, 0.0)

    intersection_area = intersection_xy[..., 0] * intersection_xy[..., 1]
    union_area = boxes1_area + boxes2_area - intersection_area

    return 1.0 * intersection_area / (union_area + tf.keras.backend.epsilon())

def bbox_ciou(boxes1_x0y0x1y1, boxes2_x0y0x1y1):
    # area
    boxes1_area = (boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) * (
                boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])
    
    boxes2_area = (boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) * (
                boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])
    
    # top-left and bottom-right coord, shape: (8, 13, 13, 3, 2)
    left_up = tf.maximum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    
    right_down = tf.minimum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])
    

    # intersection area and iou
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-9)

    # top-left and bottom-right coord of the enclosing rectangle, shape: (8, 13, 13, 3, 2)
    enclose_left_up = tf.minimum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    enclose_right_down = tf.maximum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # diagnal ** 2
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_c2 = K.pow(enclose_wh[..., 0], 2) + K.pow(enclose_wh[..., 1], 2)

    # center distances between two rectangles
    x_distance = K.pow(boxes1_x0y0x1y1[..., 0] - boxes2_x0y0x1y1[..., 0], 2)
    y_distance = K.pow(boxes1_x0y0x1y1[..., 1] - boxes2_x0y0x1y1[..., 1], 2)
    p2 = (x_distance + y_distance) + tf.keras.backend.epsilon()
    

    # add av
    
    atan1 = tf.atan((boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) / (boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1]) + tf.keras.backend.epsilon()) # w, h
    atan2 = tf.atan((boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) / (boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1]) + tf.keras.backend.epsilon())
    v = 4.0 * (K.pow(atan1 - atan2, 2) + tf.keras.backend.epsilon()) / (math.pi ** 2)
    a = v / (1 - iou + v)

    ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * a * v
    return 1 - ciou

if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)
    bboxes_1 = [[0.5, 0.55, 2.3, 50]]
    bboxes_1 = tf.convert_to_tensor(bboxes_1, dtype=tf.float32)

    bboxes_2 = [[0., 0.5, 0., 3.9]]
    bboxes_2 = tf.convert_to_tensor(bboxes_2, dtype=tf.float32)

    
    

    ciou = 1- bbox_ciou(boxes1_x0y0x1y1=bboxes_1, boxes2_x0y0x1y1=bboxes_2)
    print(ciou)
