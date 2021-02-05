import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from utils.priors import *
from model.pascal_main import ssd
import os
from preprocessing import prepare_dataset
from utils.pascal_post_processing import post_process
from utils.voc_evaluation import eval_detection_voc
from tqdm import tqdm
from pprint import pprint
import csv

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

def get_flops(model, batch_size=None):
    if batch_size is None:
        batch_size = 1

    real_model = tf.function(model).get_concrete_function(tf.TensorSpec([batch_size] + model.inputs[0].shape[1:], model.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                            run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops


DATASET_DIR = './datasets/'
IMAGE_SIZE = [300, 300]
BATCH_SIZE = 32
MODEL_NAME = 'B0'
checkpoint_filepath = './checkpoints/0201_main.h5'

# train2012 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='train')
# valid2012 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='validation')
print("Loading Test Data..")
test_data = tfds.load('voc', data_dir=DATASET_DIR, split='test')
# number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
number_test = 4952
print("Number of Test Files:", number_test)

with open('./pascal_labels.txt') as f:
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
validation_dataset = prepare_dataset(test_data, IMAGE_SIZE, BATCH_SIZE, target_transform, train=False)

print("Building CSNet Model with EfficientNet{0} backbone..".format(MODEL_NAME))
model = ssd(MODEL_NAME, pretrained=False)

print("Loading Checkpoint..")
model.load_weights(checkpoint_filepath)
model.summary()
validation_steps = number_test // BATCH_SIZE + 1
print("Number of Test Batches:", validation_steps)
#flops = get_flops(model, BATCH_SIZE)
#print(f"FLOPS: {flops}")

test_bboxes = []
test_labels = []
test_difficults = []
for sample in test_data:
    label = sample['objects']['label'].numpy()
    bbox = sample['objects']['bbox'].numpy()[:,[1, 0, 3, 2]]
    is_difficult = sample['objects']['is_difficult'].numpy()

    test_bboxes.append(bbox)
    test_labels.append(label)
    test_difficults.append(is_difficult)

print("Evaluating..")
pred_bboxes = []
pred_labels = []
pred_scores = []
for batch in tqdm(validation_dataset, total=validation_steps):
    pred = model.predict_on_batch(batch)
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
                   use_07_metric=True)
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