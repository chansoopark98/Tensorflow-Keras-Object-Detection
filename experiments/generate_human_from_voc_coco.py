import tensorflow_datasets as tfds
import argparse
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')

OUTPUT_PATH = './experiments/human_data/'
IMG_OUTPUT_PATH = OUTPUT_PATH + 'image/'
BOX_OUTPUT_PATH = OUTPUT_PATH + 'bbox/'
LABEL_OUTPUT_PATH = OUTPUT_PATH + 'label/'
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(IMG_OUTPUT_PATH, exist_ok=True)
os.makedirs(BOX_OUTPUT_PATH, exist_ok=True)
os.makedirs(LABEL_OUTPUT_PATH, exist_ok=True)

args = parser.parse_args()
DATASET_DIR = args.dataset_dir

AUTO = tf.data.experimental.AUTOTUNE


train_pascal_12 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='train')
valid_train_12 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='validation')

train_pascal_07 = tfds.load("voc", data_dir=DATASET_DIR, split='train')
valid_train_07 = tfds.load("voc", data_dir=DATASET_DIR, split='validation')

test_pascal_07 = tfds.load('voc', data_dir=DATASET_DIR, split='test')

pascal_data = train_pascal_12.concatenate(valid_train_12).concatenate(train_pascal_07).concatenate(valid_train_07).concatenate(test_pascal_07)


number_pascal = pascal_data.reduce(0, lambda x, _: x + 1).numpy()
print("PASCAL VOC 총 데이터 개수", number_pascal)


coco_train = tfds.load('coco/2017', data_dir=DATASET_DIR, split='train')
coco_train = coco_train.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
coco_train = coco_train.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))

coco_valid = tfds.load('coco/2017', data_dir=DATASET_DIR, split='validation')
coco_valid = coco_valid.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
coco_valid = coco_valid.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))

coco_data = coco_train.concatenate(coco_valid)

number_coco = coco_data.reduce(0, lambda x, _: x + 1).numpy()
print("COCO2017 총 데이터 개수", number_coco)

sample_idx = 0

print('convert voc')
for sample in tqdm(pascal_data, total=number_pascal):
    sample_idx += 1
    img = sample['image'].numpy()
    bbox = sample['objects']['bbox']
    labels = sample['objects']['label']

    bool_mask = tf.where(labels==14, True, False)
    

    if tf.reduce_any(bool_mask) == True:
        new_bbox = tf.boolean_mask(bbox, bool_mask).numpy().tolist()
        new_labels = tf.boolean_mask(labels, bool_mask).numpy().tolist()
        
        
        # for i in range(len(new_bbox)):
        #     print(i)
        #     print(new_bbox[i])
        #     print(new_labels[i])

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(IMG_OUTPUT_PATH + str(sample_idx) + '_voc_.png', img)

        with open(BOX_OUTPUT_PATH + str(sample_idx) + '_voc_.txt', "w") as file:
            for i in range(len(new_bbox)):
                file.writelines(str(new_bbox[i]) + '\n')

        with open(LABEL_OUTPUT_PATH + str(sample_idx)  +'_voc_.txt', "w") as file:
            for i in range(len(new_labels)):
                file.writelines(str(new_labels[i]) + '\n')

print('convert coco')
for sample in tqdm(coco_data, total=number_coco):
    sample_idx += 1
    img = sample['image'].numpy()
    bbox = sample['objects']['bbox']
    labels = sample['objects']['label']

    bool_mask = tf.where(labels==0, True, False)
    

    if tf.reduce_any(bool_mask) == True:
        new_bbox = tf.boolean_mask(bbox, bool_mask).numpy().tolist()
        new_labels = tf.boolean_mask(labels, bool_mask).numpy().tolist()
        
        
        # for i in range(len(new_bbox)):
        #     print(i)
        #     print(new_bbox[i])
        #     print(new_labels[i])

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(IMG_OUTPUT_PATH + str(sample_idx) + '_coco_.png', img)

        with open(BOX_OUTPUT_PATH + str(sample_idx) + '_coco_.txt', "w") as file:
            for i in range(len(new_bbox)):
                file.writelines(str(new_bbox[i]) + '\n')

        with open(LABEL_OUTPUT_PATH + str(sample_idx)  +'_coco_.txt', "w") as file:
            for i in range(len(new_labels)):
                file.writelines(str(new_labels[i]) + '\n')