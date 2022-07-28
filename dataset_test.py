import tensorflow_datasets as tfds
import tensorflow as tf
from preprocessing import prepare_dataset

from config import *

from preprocessing import prepare_dataset



test_data = tfds.load('voc', data_dir='./datasets', split='train')

CLASSES_NUM = 21
IMAGE_SIZE = (512, 512)

specs = set_priorBox('B0')
priors = create_priors_boxes(specs, IMAGE_SIZE[0])
target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

validation_dataset = prepare_dataset(test_data, (512, 512), 1,
                                            target_transform, CLASSES_NUM, train=False)


for x, y_true in validation_dataset.take(100):

    labels = tf.argmax(y_true[:,:,:CLASSES_NUM], axis=2)
    print(labels)
