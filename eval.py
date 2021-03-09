import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from utils.priors import *
from model.model_builder import ssd
import os
from preprocessing import pascal_prepare_dataset
from preprocessing import coco_prepare_dataset
from utils.model_post_processing import post_process  #
from utils.model_evaluation import eval_detection_voc
from tqdm import tqdm
from pprint import pprint
import csv

DATASET_DIR = './datasets/'
IMAGE_SIZE = [384, 384]
BATCH_SIZE = 1
MODEL_NAME = 'B0'
checkpoint_filepath = './checkpoints/coco_0304_ep31_map41.h5'
TRAIN_MODE = 'coco'  # 'voc' or 'coco'

if TRAIN_MODE == 'voc':
    test_data = tfds.load('voc', data_dir=DATASET_DIR, split='test')
    number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
    print("테스트 데이터 개수:", number_test)
    CLASSES_NUM = 21
    with open('./pascal_labels.txt') as f:
        CLASSES = f.read().splitlines()

else:
    test_data, test_info = tfds.load('coco/2017', data_dir=DATASET_DIR, split='validation', with_info=True)

    # TODO TEST 데이터셋 다운로드 후 재구축
    test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
    test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))
    # test_data = test_data.filter(lambda x: tf.reduce_all(tf.less_equal(tf.size(x['objects']['bbox']), 0)))
    # test_data = test_data.filter(lambda x: tf.reduce_all(tf.less_equal(tf.size(x['objects']['label']), 0)))

    number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()

    print("테스트 데이터 개수:", number_test)
    CLASSES_NUM = 81
    with open('./coco_labels.txt') as f:
        CLASSES = f.read().splitlines()

iou_threshold = 0.5
center_variance = 0.1
size_variance = 0.2

specs = [
    Spec(48, 8, BoxSizes(38, 77), [2]),  # 0.1
    Spec(24, 16, BoxSizes(77, 142), [2, 3]),  # 0.2
    Spec(12, 32, BoxSizes(142, 207), [2, 3]),  # 0.37
    Spec(6, 64, BoxSizes(207, 273), [2, 3]),  # 0.54
    Spec(3, 128, BoxSizes(273, 337), [2]),  # 0.71
    Spec(1, 384, BoxSizes(337, 403), [2])  # 0.88 , max 1.05
]

priors = create_priors_boxes(specs, IMAGE_SIZE[0])
target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

# instantiate the datasets
validation_dataset = pascal_prepare_dataset(test_data, IMAGE_SIZE, BATCH_SIZE, target_transform, TRAIN_MODE,
                                            train=False)
coco_dataset = coco_prepare_dataset(test_data, IMAGE_SIZE, BATCH_SIZE, target_transform, TRAIN_MODE, train=False)

print("백본 EfficientNet{0} .".format(MODEL_NAME))
model = ssd(TRAIN_MODE, MODEL_NAME, pretrained=False)

print("Loading Checkpoint..")
model.load_weights(checkpoint_filepath)
model.summary()
test_steps = number_test // BATCH_SIZE +1
print("테스트 배치 개수:", test_steps)
# flops = get_flops(model, BATCH_SIZE)
# print(f"FLOPS: {flops}")

test_bboxes = []
test_labels = []
import json


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if TRAIN_MODE == 'coco':
    test_difficults = []
    use_07_metric = False

    img_id = []
    cat_id = []
    pred_boxes = []
    pred_scores = []
    pred_list = []
    img_shapes = []
    for sample in test_data:
        img_shapes.append(sample['image'].shape)
        img_id.append(np.int(sample['image/id'].numpy().astype('int32').item()))
        cat_id.append(sample['objects']['label'].numpy().astype('int32').tolist())

    for x in tqdm(coco_dataset, total=test_steps):
        pred = model.predict_on_batch(x)
        predictions = post_process(pred, target_transform, classes=CLASSES_NUM)
        for boxes, scores, labels in predictions:
            pred_boxes.append(np.round(boxes.tolist(),2).astype('float32'))
            pred_scores.append(np.round(scores.tolist(),2).astype('float32'))


    for index in range(len(img_id)):
        for i in range(len(cat_id[index])):
            bbox = pred_boxes[index][i]
            xmin, ymin, xmax, ymax = bbox
            xmin = xmin * img_shapes[index][1]
            ymin = ymin * img_shapes[index][0]
            xmax = xmax * img_shapes[index][1]
            ymax = ymax * img_shapes[index][0]
            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin
            total_predictions = {"image_id": img_id[index],
                             "category_id": cat_id[index][i],
                             "bbox": [x, y, w, h],
                             "score": pred_scores[index][i]
                             }

            pred_list.append(total_predictions)

    with open('datasets/coco_predictions.json', 'w') as f:
        json.dump(pred_list, f, ensure_ascii=False, indent="\t", cls=NumpyEncoder)

else:
    test_difficults = []
    use_07_metric = True

    for sample in test_data:
        label = sample['objects']['label'].numpy()
        bbox = sample['objects']['bbox'].numpy()[:, [1, 0, 3, 2]]

        is_difficult = sample['objects']['is_crowd'].numpy()
        test_difficults.append(is_difficult)
        test_bboxes.append(bbox)
        test_labels.append(label)

    print("Evaluating..")
    pred_bboxes = []
    pred_labels = []
    pred_scores = []
    for x, y in tqdm(validation_dataset, total=test_steps):
        pred = model.predict_on_batch(x)
        predictions = post_process(pred, target_transform, classes=CLASSES_NUM)
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
    # print("*"*100)
    print("AP 결과")
    ap_dict = dict(zip(CLASSES, answer['ap']))
    pprint(ap_dict)
    # print("*"*100)
    print("mAP결과:", answer['map'])

    w = csv.writer(open("eval.csv", "w"))
    w.writerow(["Class", "Average Precision"])
    for key, val in ap_dict.items():
        w.writerow([key, val])


