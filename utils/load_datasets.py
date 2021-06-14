import tensorflow_datasets as tfds
import tensorflow as tf

class GenerateDatasets:
    def __init__(self,  mode, data_dir):
        """

        Args:
            mode: 학습 데이터셋 ( voc or coco )
        """
        self.mode = mode
        self.data_dir = data_dir

    def load_datasets(self):
        if self.mode == 'voc':
            self.classes = 21

            train_pascal_12 = tfds.load('voc/2012', data_dir=self.data_dir, split='train')
            valid_train_12 = tfds.load('voc/2012', data_dir=self.data_dir, split='validation')

            train_pascal_07 = tfds.load("voc", data_dir=self.data_dir, split='train')
            valid_train_07 = tfds.load("voc", data_dir=self.data_dir, split='validation')

            train_data = train_pascal_07.concatenate(valid_train_07).concatenate(train_pascal_12).concatenate(
                valid_train_12)

            test_data = tfds.load('voc', data_dir=self.data_dir, split='test')

            number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
            print("학습 데이터 개수", number_train)
            number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
            print("테스트 데이터 개수:", number_test)


        elif self.mode == 'coco':
            self.classes = 81

            train_data = tfds.load('coco/2017', data_dir=self.data_dir, split='train')
            train_data = train_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
            train_data = train_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))

            test_data = tfds.load('coco/2017', data_dir=self.data_dir, split='validation')
            test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
            test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))

            number_train = 117266
            print("학습 데이터 개수", number_train)
            number_test = 4952
            print("테스트 데이터 개수:", number_test)


        else:
            raise print("parser에서 불러올 데이터셋 이름을 확인해주세요")







