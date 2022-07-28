import os
import tensorflow_datasets as tfds
import tensorflow as tf
from preprocessing import prepare_dataset

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
        self.__select_dataset(dataset_name)

    def __select_dataset(self):
        try:
            if self.dataset_name == 'voc':
                self.dataset_list = self.__load_voc()
                self.image_key = 'image'
                self.label_key = 'label'
                self.box_key = 'bbox'

            elif self.dataset_name == 'coco':
                self.dataset_list = self.__load_custom_dataset()
                self.image_key = 'image'
                self.label_key = 'label'
                self.box_key = 'bbox'
            else:
                raise Exception('Cannot find dataset_name! \n your dataset is {0}.'.format(
                    self.dataset_name))

            self.train_data, self.number_train, self.valid_data, self.number_valid = self.dataset_list
        
        except Exception as error:
            print('Cannot select dataset. \n {0}'.format(error))


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
                concatenate(train_pascal_12).concatenate(valid_train_12)
            valid_data = valid_train_07
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


        number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
        number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()

        print("Nuber of train dataset = {0}".format(number_train))
        print("Nuber of validation dataset = {0}".format(number_valid))

        return (train_data, number_train, valid_data, number_valid)


class GenerateDatasets:
    def __init__(self, mode, data_dir, image_size, batch_size, target):
        """
             Args:
                 mode: the type of dataset to load ( params : 'voc' or 'coco' )
                 data_dir: Dataset relative path ( default : './datasets/' )
                 image_size: size of image resolution according to EfficientNet backbone
                 batch_size: batch size size
                 target: target transform function
         """
        self.mode = mode
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.target = target

        self.num_classes = None
        self.training_dataset = None
        self.validation_dataset = None

        self.number_train = 0
        self.number_test = 0

        self.load_datasets()

    def load_datasets(self):
        if self.mode == 'voc':
            self.num_classes = 21

            train_pascal_12 = tfds.load('voc/2012', data_dir=self.data_dir, split='train')
            valid_train_12 = tfds.load('voc/2012', data_dir=self.data_dir, split='validation')

            train_pascal_07 = tfds.load("voc", data_dir=self.data_dir, split='train')
            valid_train_07 = tfds.load("voc", data_dir=self.data_dir, split='validation')

            train_data = train_pascal_07.concatenate(valid_train_07).\
                concatenate(train_pascal_12).concatenate(valid_train_12)
            test_data = tfds.load("voc", data_dir=self.data_dir, split='test')

        else:
            self.num_classes = 81

            train_data = tfds.load('coco/2017', data_dir=self.data_dir, split='train')
            train_data = train_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
            train_data = train_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))

            test_data = tfds.load('coco/2017', data_dir=self.data_dir, split='validation')
            test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
            test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))


        self.number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
        self.number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()

        self.training_dataset = prepare_dataset(train_data, self.image_size, self.batch_size,
                                                self.target, self.num_classes, train=True)
        self.validation_dataset = prepare_dataset(test_data, self.image_size, self.batch_size,
                                                  self.target, self.num_classes, train=False)



