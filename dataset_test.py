from matplotlib.pyplot import box
import tensorflow_datasets as tfds
import tensorflow as tf
from utils.misc import draw_bounding, CLASSES
import cv2
from config import *
from utils.load_datasets import GenerateDatasets

test_data = tfds.load('coex_hand', data_dir='./datasets/', split='train')

for sample in test_data.take(1000):
    
    try:
        
        image = sample['image'].numpy()
        labels = sample['label'].numpy()
        labels = tf.where(labels>=0, 1, 0)
        boxes = sample['bbox'].numpy() 
        
        # y_min, x_min, y_max, y_max -> x_min, y_min, x_max, y_max
        convert_boxes = boxes.copy()
        convert_boxes[:, [0,2]] = boxes.copy()[:, [1,3]]
        convert_boxes[:, [1,3]] = boxes.copy()[:, [0,2]]
        # convert_boxes[:, [0,2]] = boxes.copy()[:, [0,2]]
        # convert_boxes[:, [1,3]] = boxes.copy()[:, [1, 3]]

        
        # y_min = tf.where(tf.greater_equal(boxes[:,0], boxes[:,2]), tf.cast(0, dtype=tf.float32), boxes[:,0])
        # x_max = tf.where(tf.greater_equal(x_min, boxes[:,3]), tf.cast(x_min+0.1, dtype=tf.float32), boxes[:,3])
        # y_max = tf.where(tf.greater_equal(y_min, boxes[:,2]), tf.cast(y_min+0.1, dtype=tf.float32), boxes[:,2])
        # boxes = tf.stack([x_min, y_min, x_max, y_max], axis=1)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        draw_bounding(img = image, bboxes=convert_boxes, labels=labels, scores=None, img_size=image.shape[:2], label_list=COCO_CLASSES)

        # print('Image {0} \n  Labels {1} \n Bbox {2}'.format(image.shape, labels, boxes))
        for i in range(len(boxes[0])):
            
            if boxes[0][i] > 1.0 or boxes[0][i] <= 0.:
                print(boxes[0][i])    
                print('bbox is out of 0-1 value')
                
            elif boxes.size == 0:
                print('size is zero')
        
        
        cv2.imshow('test', image)
        cv2.waitKey(0)

    except Exception as e:
        raise print('Dataset is out of range {0}'.format(e))