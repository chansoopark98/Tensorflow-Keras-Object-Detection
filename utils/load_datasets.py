import tensorflow_datasets as tfds
import tensorflow as tf
from preprocessing import prepare_dataset, cityScapes


class GenerateDatasets:
    def __init__(self, mode, data_dir, image_size, batch_size, target):
        """
        Args:
            mode: 불러올 데이터셋 종류입니다 ( params : 'voc' or 'coco' )
            data_dir: 데이터셋 상대 경로 ( default : './datasets/' )
            image_size: EfficientNet 백본에 따른 이미지 해상도 크기
            batch_size: 배치 사이즈 크기
            target: target 변환 함수
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
        print("학습 데이터 개수", self.number_train)
        self.number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
        print("테스트 데이터 개수:", self.number_test)

        self.training_dataset = prepare_dataset(train_data, self.image_size, self.batch_size,
                                                self.target, self.num_classes, train=True)
        self.validation_dataset = prepare_dataset(test_data, self.image_size, self.batch_size,
                                                  self.target, self.num_classes, train=False)




class CityScapes:
    def __init__(self, data_dir, image_size, batch_size):
        """
        Args:
            data_dir: 데이터셋 상대 경로 ( default : './datasets/' )
            image_size: EfficientNet 백본에 따른 이미지 해상도 크기
            batch_size: 배치 사이즈 크기
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size

        self.num_classes = None
        self.training_dataset = None
        self.validation_dataset = None

        self.number_train = 0
        self.number_valid = 0
        self.number_test = 0

        self.load_datasets()

    def load_datasets(self):
        self.num_classes = 20

        train_data = tfds.load('cityscapes/semantic_segmentation', data_dir=self.data_dir, split='train'
                             )
        valid_data = tfds.load('cityscapes/semantic_segmentation', data_dir=self.data_dir, split='validation'
                             )
        test_data = tfds.load('cityscapes/semantic_segmentation', data_dir=self.data_dir, split='test'
                            )

        # self.number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
        self.number_train = 2975
        print("학습 데이터 개수", self.number_train)
        # self.number_test = valid_data.reduce(0, lambda x, _: x + 1).numpy()
        self.number_valid = 500
        print("검증 데이터 개수:", self.number_valid)

        self.number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
        print("검증 데이터 개수:", self.number_test)



        self.training_dataset = cityScapes(train_data, self.image_size, self.batch_size, train=True)
        self.validation_dataset = cityScapes(valid_data, self.image_size, self.batch_size, train=False)