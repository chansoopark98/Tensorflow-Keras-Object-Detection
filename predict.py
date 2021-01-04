import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from utils.priors import *
from model.ssd import ssd
import matplotlib.pyplot as plt
from utils.post_processing import post_process
from matplotlib.image import imread
import os
from preprocessing import prepare_for_prediction
from tqdm import tqdm
from PIL import Image
import cv2

from collections import namedtuple

DATASET_DIR = './dataset'
IMAGE_SIZE = [300, 300]
BATCH_SIZE = 2
MODEL_NAME = 'B0'
checkpoint_filepath = './checkpoints/1112_ep300.h5'
INPUT_DIR = './inputs'
OUTPUT_DIR = './outputs'

iou_threshold = 0.5
center_variance = 0.1
size_variance = 0.2

specs = [
                SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
                SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
                SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
                SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
                SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
                SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
        ]


priors = generate_ssd_priors(specs, IMAGE_SIZE[0])
target_transform = MatchPrior(priors, center_variance, size_variance, iou_threshold)

print("Building SSD Model with EfficientNet{0} backbone..".format(MODEL_NAME))
model = ssd(MODEL_NAME, pretrained=False)
model.summary()
print("모델로드")
model.load_weights(checkpoint_filepath)

filenames = os.listdir(INPUT_DIR)
dataset = tf.data.Dataset.list_files(INPUT_DIR + '/*', shuffle=False)
dataset = dataset.map(prepare_for_prediction)
dataset = dataset.batch(BATCH_SIZE)

x, y = 0, BATCH_SIZE
test_steps = 4952 // BATCH_SIZE + 1

def rgb2bgr(tpl):
    return (tpl[2], tpl[1], tpl[0])
Label   = namedtuple('Label',   ['name', 'color'])
def color_map(index):
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

def draw_bounding(img , bboxes, labels, img_size):
    # resizing 작업
    if np.max(bboxes) < 10:

        bboxes[:, [0,2]] = bboxes[:, [0,2]]*img_size[1]
        bboxes[:, [1,3]] = bboxes[:, [1,3]]*img_size[0]

    for i, bbox in enumerate(bboxes):
        xmin = tf.cast(bbox[0], dtype=tf.int32)
        ymin = tf.cast(bbox[1], dtype=tf.int32)
        xmax = tf.cast(bbox[2], dtype=tf.int32)
        ymax = tf.cast(bbox[3], dtype=tf.int32)
        img_box = np.copy(img)
        _, color = color_map(int(labels[i]-1))
        cv2.rectangle(img_box, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.rectangle(img_box, (xmin - 1, ymin), (xmax + 1, ymin - 20), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_box, CLASSES[int(labels[i]-1)], (xmin + 5, ymin - 5), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        alpha = 0.8
        cv2.addWeighted(img_box, alpha, img, 1. - alpha, 0, img)
import time
start = time.time()
for batch in tqdm(dataset, total=test_steps):

    pred = model.predict_on_batch(batch)
    predictions = post_process(pred, target_transform, confidence_threshold=0.4)

    for i, path in enumerate(filenames[x:y]):

        im = cv2.imread(INPUT_DIR+'/'+path)
        pred_boxes, pred_scores, pred_labels = predictions[i]
        if pred_boxes.size > 0:

            draw_bounding(im, pred_boxes,  labels=pred_labels, img_size=im.shape[:2])
        fn = OUTPUT_DIR + '/' + path +'.jpg'
        cv2.imwrite(fn, im)

    x = y
    y += BATCH_SIZE
end = time.time()

print(((end - start)/4952)*1000)