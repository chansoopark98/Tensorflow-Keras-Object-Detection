import os
import tensorflow_datasets as tfds
import tensorflow as tf
from .augmentations import *
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from typing import Union

AUTO = tf.data.experimental.AUTOTUNE

class DataLoadHandler:
    def __init__(self, data_dir: str, dataset_name: str):
        """
        This class performs pre-process work for each dataset and load tfds.
        Args:
            data_dir     (str)   : Dataset relative path ( default : './datasets/' )
            dataset_name (str)   : Tensorflow dataset name (e.g: 'voc or coco')
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.__select_dataset()


    def __select_dataset(self):
        try:
            if self.dataset_name == 'voc':
                self.num_classes = 21
                self.dataset_list = self.__load_voc()
                self.image_key = 'image'
                self.label_key = 'label'
                self.bbox_key = 'bbox'

            elif self.dataset_name == 'coco':
                self.num_classes = 81
                self.dataset_list = self.__load_coco()
                self.image_key = 'image'
                self.label_key = 'label'
                self.bbox_key = 'bbox'

            else:
                self.dataset_list = self.__load_custom_dataset(dataset_name=self.dataset_name)
                self.image_key = 'image'
                self.label_key = 'label'
                self.bbox_key = 'bbox'

        except Exception as error:
            print('Cannot find dataset_name! \n your dataset is {0}. \
                  Currently available default dataset types are: \
                  voc, coco, custom_datasaet. \n Error log : {1} '.format(self.dataset_name, error))

        self.train_data, self.number_train, self.valid_data, self.number_valid, self.test_data, self.number_test = self.dataset_list

    def __load_voc(self, use_train_val: bool = True):
        
        if os.path.isdir(self.data_dir + 'voc') == False:
            raise Exception(
                'Cannot find PASCAL VOC Tensorflow Datasets. Please download VOC data. \
                 You can download use dataset_download.py')


        train_pascal_12 = tfds.load('voc/2012', data_dir=self.data_dir, split='train')
        valid_train_12 = tfds.load('voc/2012', data_dir=self.data_dir, split='validation')

        train_pascal_07 = tfds.load("voc", data_dir=self.data_dir, split='train')
        valid_train_07 = tfds.load("voc", data_dir=self.data_dir, split='validation')

        test_data = tfds.load("voc", data_dir=self.data_dir, split='test')

        if use_train_val:
            train_data = train_pascal_07.\
                concatenate(train_pascal_12).concatenate(valid_train_12).concatenate(valid_train_07)
            valid_data = test_data
        else:
            train_data = train_pascal_07.concatenate(train_pascal_12)
            valid_data = valid_train_07.concatenate(valid_train_12)


        number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
        number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()
        number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()

        print("Nuber of train dataset = {0}".format(number_train))
        print("Nuber of validation dataset = {0}".format(number_valid))
        print("Nuber of test dataset = {0}".format(number_test))

        return (train_data, number_train, valid_data, number_valid, test_data, number_test)


    def __load_coco(self):

        if os.path.isdir(self.data_dir + 'coco') == False:
            raise Exception(
                'Cannot find COCO2017 Tensorflow Datasets. Please download COCO2017 data. \
                 You can download use dataset_download.py')


        train_data = tfds.load('coco/2017', data_dir=self.data_dir, split='train')
        # Remove blank label files
        train_data = train_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
        train_data = train_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))

        valid_data = tfds.load('coco/2017', data_dir=self.data_dir, split='validation')
        # Remove blank label files
        valid_data = valid_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
        valid_data = valid_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))
    
        test_data = tfds.load('coco/2017', data_dir=self.data_dir, split='test')

        number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
        number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()
        number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()

        print("Nuber of train dataset = {0}".format(number_train))
        print("Nuber of validation dataset = {0}".format(number_valid))
        print("Nuber of test dataset = {0}".format(number_test))

        return (train_data, number_train, valid_data, number_valid, test_data, number_test)


    def __load_custom_dataset(self, dataset_name):
        """
            Loads a custom dataset specified by the user.
            For custom datasets, you don't need to specify self.num_classes directly, just count the classes yourself.
            e.g. If the dataset has at most 2 integer labels (class 1: 0, class 2: 1, class 3: 2)
            The output of self.num_classes is 4 (foreground class 3 + background 1)
        """
        
        if os.path.isdir(self.data_dir + dataset_name) == False:
            raise Exception(
                'Cannot find your custom dataset -> {0}. Please download VOC data. \
                 You can download use dataset_download.py'.format(dataset_name))

        # train_data = tfds.load(dataset_name, data_dir=self.data_dir, split='train[10%:]')
        # valid_data = tfds.load(dataset_name, data_dir=self.data_dir, split='train[:10%]')
        train_data = tfds.load(dataset_name, data_dir=self.data_dir, split='train')
        valid_data = tfds.load(dataset_name, data_dir=self.data_dir, split='validation')
        test_data = valid_data    

        number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
        number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()
        number_test = number_valid

        print("Nuber of train dataset = {0}".format(number_train))
        print("Nuber of validation dataset = {0}".format(number_valid))
        print("Nuber of test dataset = {0}".format(number_test))

        # calculate number of classes
        total_data = train_data.concatenate(valid_data)

        best_label_value = 0
        for sample in total_data.take(number_train + number_valid):
            max_label = tf.math.reduce_max(sample['label'])

            if max_label >= best_label_value:
                best_label_value = max_label
        
        # The maximum value of the label starts at index 0, so we need to add +1,
        # And we add a background class to add a total of 2.
        self.num_classes = best_label_value.numpy() + 2
        
        return (train_data, number_train, valid_data, number_valid, test_data, number_test)


class GenerateDatasets(DataLoadHandler):
    def __init__(self, data_dir: str, image_size: tuple, batch_size: int, image_norm_type: str,
                 target_transform: object,
                 dataset_name: str = 'voc'):
        """
        Args:
            data_dir         (str)    : Dataset relative path (default : './datasets/').
            image_size       (tuple)  : Model input image resolution.
            batch_size       (int)    : Batch size.
            image_norm_type  (str)    : Input RGB image scaling type(normalize, tf or torch)
            target_transform (object) : Class instance defining prior box.
            dataset_name     (str)    : Tensorflow dataset name (e.g: 'cityscapes').
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.image_norm_type = image_norm_type
        self.target_transform = target_transform
        self.dataset_name = dataset_name
        super().__init__(data_dir=self.data_dir, dataset_name=self.dataset_name)

    @tf.function    
    def preprocess_test(self, sample: dict) -> Union[tf.Tensor, tf.Tensor, tf.Tensor]:
        image = tf.cast(sample[self.image_key], dtype=tf.float32)
        if self.dataset_name == 'custom_dataset':
            labels = sample[self.label_key] + 1
            bbox = sample[self.bbox_key]
        else:
            # PASCAL VOC or COCO2017 dataset
            labels = sample['objects'][self.label_key] + 1
            bbox = sample['objects'][self.bbox_key]
        
        x_min = tf.where(tf.greater_equal(bbox[:,1], bbox[:,3]), tf.cast(0, dtype=tf.float32), bbox[:,1])
        y_min = tf.where(tf.greater_equal(bbox[:,0], bbox[:,2]), tf.cast(0, dtype=tf.float32), bbox[:,0])
        x_max = tf.where(tf.greater_equal(x_min, bbox[:,3]), tf.cast(x_min+0.1, dtype=tf.float32), bbox[:,3])
        y_max = tf.where(tf.greater_equal(y_min, bbox[:,2]), tf.cast(y_min+0.1, dtype=tf.float32), bbox[:,2])
        bbox = tf.stack([x_min, y_min, x_max, y_max], axis=1)

        if self.image_norm_type == 'torch':
            image = preprocess_input(image, mode='torch')
        elif self.image_norm_type == 'tf':
            image = preprocess_input(image, mode='tf')
        else:
            image /= 255
            
        return (image, bbox, labels) 


    @tf.function
    def preprocess(self, sample: dict, clip_bbox: bool = True) -> Union[tf.Tensor, tf.Tensor, tf.Tensor]:
        image = tf.cast(sample[self.image_key], dtype=tf.float32)
        
        if self.dataset_name == 'custom_dataset':
            labels = sample[self.label_key] + 1
            bbox = sample[self.bbox_key]
        else:
            # PASCAL VOC or COCO2017 dataset
            labels = sample['objects'][self.label_key] + 1
            bbox = sample['objects'][self.bbox_key]

        x_min = tf.where(tf.greater_equal(bbox[:,1], bbox[:,3]), tf.cast(0, dtype=tf.float32), bbox[:,1])
        y_min = tf.where(tf.greater_equal(bbox[:,0], bbox[:,2]), tf.cast(0, dtype=tf.float32), bbox[:,0])
        x_max = tf.where(tf.greater_equal(x_min, bbox[:,3]), tf.cast(x_min+0.1, dtype=tf.float32), bbox[:,3])
        y_max = tf.where(tf.greater_equal(y_min, bbox[:,2]), tf.cast(y_min+0.1, dtype=tf.float32), bbox[:,2])
        bbox = tf.stack([x_min, y_min, x_max, y_max], axis=1)

        if self.image_norm_type == 'torch':
            image = preprocess_input(image, mode='torch')
        elif self.image_norm_type == 'tf':
            image = preprocess_input(image, mode='tf')
        else:
            image /= 255
            
        return (image, bbox, labels)
    
    
    @tf.function
    def augmentation(self, image: tf.Tensor, boxes: tf.Tensor, labels: tf.Tensor)-> Union[tf.Tensor, tf.Tensor, tf.Tensor]:
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


    def join_target(self, image: tf.Tensor, bbox: tf.Tensor, labels: tf.Tensor) -> Union[tf.Tensor, tf.Tensor]:
        locations, labels = self.target_transform(tf.cast(bbox, tf.float32), labels)
        labels = tf.one_hot(labels, self.num_classes, axis=1, dtype=tf.float32)
        targets = tf.concat([labels, locations], axis=1)
        resized_img = tf.image.resize(image, self.image_size)

        return (resized_img, targets)


    def get_trainData(self, train_data):
        train_data = train_data.shuffle(256)
        train_data = train_data.map(self.preprocess, num_parallel_calls=AUTO)
        train_data = train_data.map(self.augmentation, num_parallel_calls=AUTO)
        train_data = train_data.map(lambda image, boxes, labels: self.join_target(image, boxes, labels))
        train_data = train_data.padded_batch(self.batch_size)
        train_data = train_data.prefetch(AUTO)
        train_data = train_data.repeat()

        return train_data


    def get_validData(self, valid_data):
        valid_data = valid_data.map(self.preprocess, num_parallel_calls=AUTO)
        valid_data = valid_data.map(lambda image, boxes, labels: self.join_target(image, boxes, labels))
        valid_data = valid_data.padded_batch(self.batch_size).prefetch(AUTO)

        return valid_data


    def get_testData(self, test_data):
        test_data = test_data.map(self.preprocess_test)
        test_data = test_data.map(lambda image, boxes, labels: self.join_target(image, boxes, labels))
        test_data = test_data.batch(self.batch_size).prefetch(AUTO)

        return test_data



