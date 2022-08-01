from matplotlib.pyplot import box
import tensorflow_datasets as tfds
import tensorflow as tf
from utils.misc import draw_bounding, CLASSES
import cv2

from config import *

from utils.load_datasets import GenerateDatasets



test_data = tfds.load('display_detection', data_dir='./datasets', split='train')

CLASSES_NUM = 21
IMAGE_SIZE = (512, 512)

specs = set_priorBox('B0')
priors = create_priors_boxes(specs, IMAGE_SIZE[0])
target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

dataset_config = GenerateDatasets(data_dir='./datasets/',
                    image_size=(300, 300),
                    batch_size=1, image_norm_type='torch',
                    target_transform=target_transform, dataset_name='custom_dataset')

# test_dataset= dataset_config.get_testData(test_data=dataset_config.valid_data)

for sample in test_data.take(100):
    
    # image = sample['image'].numpy()
    # labels = sample['objects']['label'].numpy() +1
    # boxes = sample['objects']['bbox'].numpy() 

    image = sample['image'].numpy()
    labels = sample['label'].numpy() +1
    boxes = sample['bbox'].numpy() 
    
    # y_min, x_min, y_max, y_max -> x_min, y_min, x_max, y_max
    convert_boxes = boxes.copy()
    convert_boxes[:, [0,2]] = boxes.copy()[:, [1,3]]
    convert_boxes[:, [1,3]] = boxes.copy()[:, [0,2]]
    
    # y_min = tf.where(tf.greater_equal(boxes[:,0], boxes[:,2]), tf.cast(0, dtype=tf.float32), boxes[:,0])
    # x_max = tf.where(tf.greater_equal(x_min, boxes[:,3]), tf.cast(x_min+0.1, dtype=tf.float32), boxes[:,3])
    # y_max = tf.where(tf.greater_equal(y_min, boxes[:,2]), tf.cast(y_min+0.1, dtype=tf.float32), boxes[:,2])
    # boxes = tf.stack([x_min, y_min, x_max, y_max], axis=1)
    print(convert_boxes.shape)
    draw_bounding(img = image, bboxes=convert_boxes, labels=labels, img_size=image.shape[:2], label_list=CLASSES)

    print('Image {0} \n  Labels {1} \n Bbox {2}'.format(image.shape, labels, boxes))
    cv2.imshow('test', image)
    cv2.waitKey(0)