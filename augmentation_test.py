import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
from utils.augmentations import *
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
args = parser.parse_args()
DATASET_DIR = args.dataset_dir

dataset = tfds.load('voc/2012', data_dir=DATASET_DIR, split='train')

for sample in dataset:
    image = tf.cast(sample['image'], tf.float32)
    labels = sample['objects']['label']+1
    boxes = sample['objects']['bbox']

    origin_image = tf.cast(image, tf.uint8)
    plt.imsave('./aug_test_images/기본이미지.png', origin_image.numpy()/255)

    image_saturation = tf.image.random_saturation(image, lower=0.5, upper=1.5) # 랜덤 채도
    plt.imsave('./aug_test_images/랜덤채도.png', image_saturation.numpy()/255)

    image_brightness = tf.image.random_brightness(image, max_delta=0.15) # 랜덤 밝기
    image_brightness  = tf.cast(image_brightness, tf.uint8)
    plt.imsave('./aug_test_images/랜덤밝기.png', image_brightness.numpy())

    image_contrast = tf.image.random_contrast(image, lower=0.5, upper=1.5) # 랜덤 대비
    image_contrast = tf.cast(image_contrast, tf.uint8)
    plt.imsave('./aug_test_images/랜덤대비.png', image_contrast.numpy() / 255)

    image_hue = tf.image.random_hue(image, max_delta=0.2) # 랜덤 휴 트랜스폼
    plt.imsave('./aug_test_images/랜덤휴.png', image_hue.numpy() / 255)

    image_lightning = random_lighting_noise(image)
    plt.imsave('./aug_test_images/랜덤라이트닝노이즈.png', image_lightning.numpy() / 255)

    image_expand, boxes_expand = expand(image, boxes)
    plt.imsave('./aug_test_images/랜덤확장.png', image_expand.numpy() / 255)

    image_random_crop, boxes_crop, labels_crop = random_crop(image, boxes, labels) # 랜덤 자르기
    plt.imsave('./aug_test_images/랜덤크롭.png', image_random_crop.numpy() / 255)


    image_flip, boxes_flip = random_flip(image, boxes) # 랜덤 뒤집기
    plt.imsave('./aug_test_images/랜덤플립.png', image_flip.numpy() / 255)


    plt.imshow((origin_image/255))
    plt.show()


