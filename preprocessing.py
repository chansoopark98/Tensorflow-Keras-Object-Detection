from tensorflow.keras.applications.imagenet_utils import preprocess_input

import tensorflow as tf
from utils.augmentations import *

AUTO = tf.data.experimental.AUTOTUNE


@tf.function
def data_augment(image, boxes, labels):
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5) # 랜덤 채도
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.15) # 랜덤 밝기
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5) # 랜덤 대비
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_hue(image, max_delta=0.2) # 랜덤 휴 트랜스폼
    image = random_lighting_noise(image)
    image, boxes = expand(image, boxes)
    image, boxes, labels = random_crop(image, boxes, labels) # 랜덤 자르기
    image, boxes = random_flip(image, boxes) # 랜덤 뒤집기

    return (image, boxes, labels)


def prepare_input(sample, convert_to_normal=True):
    img = tf.cast(sample['image'], dtype=tf.float32)
    # img = img - image_mean 이미지 평균
    labels = sample['objects']['label']+1
    bbox = sample['objects']['bbox']
    if convert_to_normal: # ymin=0.3, xmin=0.8, ymax=0.5, xmax=1.0
        x_min = tf.where(tf.greater_equal(bbox[:,1], bbox[:,3]), tf.cast(0, dtype=tf.float32), bbox[:,1])
        y_min = tf.where(tf.greater_equal(bbox[:,0], bbox[:,2]), tf.cast(0, dtype=tf.float32), bbox[:,0])
        x_max = tf.where(tf.greater_equal(x_min, bbox[:,3]), tf.cast(x_min+0.1, dtype=tf.float32), bbox[:,3])
        y_max = tf.where(tf.greater_equal(y_min, bbox[:,2]), tf.cast(y_min+0.1, dtype=tf.float32), bbox[:,2])
        bbox = tf.stack([x_min, y_min, x_max, y_max], axis=1)

        # bbox = tf.stack([bbox[:,1], bbox[:,0], bbox[:,3], bbox[:,2]], axis=1)

    # filter_nan = lambda x: not tf.reduce_any(tf.math.is_nan(img)) and not tf.math.is_nan(img)
    #
    # train_data = train_data.filter(filter_nan)
    img = preprocess_input(img, mode='torch')

    # img = tf.cast(img, tf.float32) # 형변환

    #image_mean = (0.485, 0.456, 0.406)
    #image_std = (0.229, 0.224, 0.225)
    #img = (img - image_mean) / image_std # 데이터셋 pascal 평균 분산치 실험
    return (img, bbox, labels)

def prepare_cocoEval_input(sample):
    img = tf.cast(sample['image'], dtype=tf.float32)

    # img_shape = sample['image'].shape

    img_id = sample['image/id']



    img = preprocess_input(img, mode='torch')
    img = tf.image.resize(img, [512, 512])
    # return (img, img_shape , img_id, cat_id)
    return (img, img_id)



# 타겟 연결 오리지날
def join_target(image, bbox, labels, image_size, target_transform, classes):
    locations, labels = target_transform(tf.cast(bbox, tf.float32), labels)
    labels = tf.one_hot(labels, classes, axis=1, dtype=tf.float32) ### 1 -> classes
    targets = tf.concat([labels, locations], axis=1)

    return (tf.image.resize(image, image_size), targets)

# def join_target(image, bbox, labels, image_size, target_transform, classes):
#   locations, labels = target_transform(tf.cast(bbox, tf.float32), labels)
#   labels = tf.one_hot(labels, classes, axis=1, dtype=tf.float32) ### 1 -> classes
#   targets = tf.concat([labels, locations], axis=1)
#   return image, targets

def coco_prepare_dataset(dataset, image_size, batch_size, target_transform, train_mode, train=False):
    classes = 81

    if train:
        dataset = dataset.map(prepare_input, num_parallel_calls=AUTO)
        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
        dataset = dataset.map(lambda image, boxes,
                                       labels: join_target(image, boxes, labels, image_size, target_transform, classes),
                                num_parallel_calls=AUTO)
        dataset = dataset.padded_batch(batch_size)
        dataset = dataset.prefetch(AUTO)
    else:
        dataset = dataset.map(prepare_input, num_parallel_calls=AUTO)
        dataset = dataset.map(lambda image, boxes,
                                   labels: join_target(image, boxes, labels, image_size, target_transform, classes),
                            num_parallel_calls=AUTO)
        dataset = dataset.padded_batch(batch_size)
        dataset = dataset.prefetch(AUTO)

    return dataset

def coco_eval_dataset(dataset, image_size, batch_size, target_transform, train_mode, train=False):
    classes = 81

    dataset = dataset.map(prepare_cocoEval_input, num_parallel_calls=AUTO)

    dataset = dataset.padded_batch(batch_size)
    dataset = dataset.prefetch(AUTO)



    return dataset


def pascal_prepare_dataset(dataset, image_size, batch_size, target_transform, train_mode, train=False):
    classes = 21

    dataset = dataset.map(prepare_input, num_parallel_calls=AUTO)
    if train:
        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.map(lambda image, boxes,
                               labels: join_target(image, boxes, labels, image_size, target_transform, classes),
                        num_parallel_calls=AUTO)
    dataset = dataset.padded_batch(batch_size)
    dataset = dataset.prefetch(AUTO)

    return dataset

# predict 할 때
def prepare_for_prediction(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img, [512, 512])
    img = preprocess_input(img, mode='torch')

    return img
    
def decode_img(img,  image_size=[384, 384]):
    # 텐서 변환
    img = tf.image.decode_jpeg(img, channels=3)
    # 이미지 리사이징
    return tf.image.resize(img, image_size)
    
