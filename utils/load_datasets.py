import tensorflow_datasets as tfds
import tensorflow as tf
from preprocessing import prepare_dataset


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



