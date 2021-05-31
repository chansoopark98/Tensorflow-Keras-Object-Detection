import tensorflow as tf
import tensorflow_datasets as tfds
from preprocessing import test_priors_datasets
from config import *

DATASET_DIR = './datasets/'
train_pascal_12 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='train')
valid_train_12 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='validation')

train_pascal_07 = tfds.load("voc", data_dir=DATASET_DIR, split='train')
valid_train_07 = tfds.load("voc", data_dir=DATASET_DIR, split='validation')

train_data = train_pascal_07.concatenate(valid_train_07).concatenate(train_pascal_12).concatenate(valid_train_12)

specs = set_priorBox('B0')
print(specs)

priors = create_priors_boxes(specs, 512)
target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

dataset = test_priors_datasets(train_data, [512, 512], target_transform, batch_size=128)

sum = 0
for image, locations, labels in dataset:
    mask = labels > 0
    labels = tf.boolean_mask(labels, mask)
    sum += tf.reduce_sum(labels)

print(sum)


