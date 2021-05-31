import sys
import time

import tensorflow as tf
import tensorflow_datasets as tfds
from preprocessing import test_priors_datasets
from config import *

DATASET_DIR = './datasets/'
DATASET_TARGET = 'voc'

if DATASET_TARGET == 'voc':
    train_pascal_12 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='train')
    valid_train_12 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='validation')

    train_pascal_07 = tfds.load("voc", data_dir=DATASET_DIR, split='train')
    valid_train_07 = tfds.load("voc", data_dir=DATASET_DIR, split='validation')

    train_data = train_pascal_07.concatenate(valid_train_07).concatenate(train_pascal_12).concatenate(valid_train_12)

elif DATASET_TARGET == 'coco':
    train_data = tfds.load('coco/2017', data_dir=DATASET_DIR, split='train')
    train_data = train_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
    train_data = train_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))


x_min = 204
x_max = 204 # 3202881

# 두번째 124 3854806
# 130 4145063
best_score = 0
best_x_min = 0
best_x_max = 0

def create_specs(x_min, x_max):
    specs =  [  Spec(64, 8, BoxSizes(19, 38), [2]), # 0.029
                Spec(32, 16, BoxSizes(x_min, x_max), [2]), # 0.08
                Spec(16, 32, BoxSizes(102, 112), [2]), # 0.238 -> 0.199
                Spec(8, 64, BoxSizes(204, 224), [2]), # 0.4
                Spec(4, 128, BoxSizes(332, 347), [2]), # 0.65
                         ]
    priors = create_priors_boxes(specs, 512)
    target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)
    return target_transform

for i in range(100):
        i *= 2
        target_transform = create_specs(x_min=x_min+i, x_max=x_max+i)
        dataset = test_priors_datasets(train_data, [512, 512], target_transform, batch_size=1)

        sum = 0
        for locations , labels in dataset:
            labels = tf.argmax(labels, axis=2)
            mask = labels > 0
            pos_labels = tf.boolean_mask(labels, mask)

            gt_locations = tf.reshape(tf.boolean_mask(locations, mask), [-1, 4])
            time.sleep((1000))
            sum += tf.reduce_sum(labels)


        if best_score <= sum:
            best_score = sum
            best_x_min = x_min+i
            best_x_max = x_max+i
            print(best_score)
            print("best_x_min",best_x_min)
            print("best_x_max",best_x_max)

print("final \n\n\n")
print("best_x_min", best_x_min)
print("best_x_max", best_x_max)



