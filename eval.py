import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from utils.priors import *
from model.pascal_main import ssd
import os
from preprocessing import prepare_dataset
from utils.pascal_post_processing import post_process #
from utils.voc_evaluation import eval_detection_voc
from tqdm import tqdm
from pprint import pprint
import csv


DATASET_DIR = './datasets/'
IMAGE_SIZE = [384, 384]
BATCH_SIZE = 16
MODEL_NAME = 'B0'
checkpoint_filepath = './checkpoints/0218.h5'
TRAIN_MODE = 'voc' # 'voc' or 'coco'

if TRAIN_MODE == 'voc':
    test_data = tfds.load('voc', data_dir=DATASET_DIR, split='test')
    number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
    print("테스트 데이터 개수:", number_test)

    with open('./pascal_labels.txt') as f:
        CLASSES = f.read().splitlines()

else :
    test_data = tfds.load('coco/2017', data_dir=DATASET_DIR, split='validation')
    test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
    number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
    print("테스트 데이터 개수:", number_test)

    with open('./coco_labels.txt') as f:
        CLASSES = f.read().splitlines()

iou_threshold = 0.5
center_variance = 0.1
size_variance = 0.2

specs = [
                Spec(48, 8, BoxSizes(40, 90), [2]),
                Spec(24, 16, BoxSizes(90, 151), [2, 3]),
                Spec(12, 32, BoxSizes(151, 212), [2, 3]),
                Spec(6, 64, BoxSizes(212, 273), [2, 3]),
                Spec(3, 128, BoxSizes(273, 334), [2]),
                Spec(1, 384, BoxSizes(334, 395), [2])
        ]


priors = generate_ssd_priors(specs, IMAGE_SIZE[0])
target_transform = MatchPrior(priors, center_variance, size_variance, iou_threshold)

# instantiate the datasets
validation_dataset = prepare_dataset(test_data, IMAGE_SIZE, BATCH_SIZE, target_transform, TRAIN_MODE, train=False)

print("백본 EfficientNet{0} .".format(MODEL_NAME))
model = ssd(TRAIN_MODE, MODEL_NAME, pretrained=False)

print("Loading Checkpoint..")
model.load_weights(checkpoint_filepath)
model.summary()
test_steps = number_test // BATCH_SIZE + 1
print("테스트 배치 개수:", test_steps)
#flops = get_flops(model, BATCH_SIZE)
#print(f"FLOPS: {flops}")

test_bboxes = []
test_labels = []
if TRAIN_MODE == 'COCO':
    test_difficults = None
    use_07_metric= False
else :
    test_difficults = []
    use_07_metric = True

for sample in test_data:
    label = sample['objects']['label'].numpy()
    bbox = sample['objects']['bbox'].numpy()[:,[1, 0, 3, 2]]
    if TRAIN_MODE == 'voc' :
        is_difficult = sample['objects']['is_difficult'].numpy()
        test_difficults.append(is_difficult)
    test_bboxes.append(bbox)
    test_labels.append(label)


print("Evaluating..")
pred_bboxes = []
pred_labels = []
pred_scores = []
for x, y in tqdm(validation_dataset, total=test_steps):
    pred = model.predict_on_batch(x)
    predictions = post_process(pred, target_transform)
    for prediction in predictions:
        boxes, scores, labels = prediction
        pred_bboxes.append(boxes)
        pred_labels.append(labels.astype(int) - 1)
        pred_scores.append(scores)

answer = eval_detection_voc(pred_bboxes=pred_bboxes,
                   pred_labels=pred_labels,
                   pred_scores=pred_scores,
                   gt_bboxes=test_bboxes,
                   gt_labels=test_labels,
                   gt_difficults=test_difficults,
                   use_07_metric=use_07_metric)
#print("*"*100)
print("AP 결과")
ap_dict = dict(zip(CLASSES, answer['ap']))
pprint(ap_dict)
#print("*"*100)
print("mAP결과:", answer['map'])

w = csv.writer(open("eval.csv", "w"))
w.writerow(["Class", "Average Precision"])
for key, val in ap_dict.items():
    w.writerow([key, val])
