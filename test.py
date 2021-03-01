import tensorflow_datasets as tfds
from utils.priors import *
import os
from preprocessing import prepare_dataset
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from model.model_builder import ssd
from tensorflow.keras.utils import plot_model
from calc_flops import get_flops


CONTINUE_TRAINING = False
SAVE_MODEL_NAME = '0225'
DATASET_DIR = './datasets/'
IMAGE_SIZE = [384, 384]
BATCH_SIZE = 16
MODEL_NAME = 'B0'
EPOCHS = 50
TRAIN_MODE = 'voc' # 'voc' or 'coco'
checkpoint_filepath = './checkpoints/'
base_lr = 0.00075

if TRAIN_MODE == 'voc':
    from model.pascal_loss import total_loss
    train_pascal_12 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='train')
    valid_train_12 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='validation')

    train_pascal_07 = tfds.load("voc", data_dir=DATASET_DIR, split='train')
    valid_train_07 = tfds.load("voc", data_dir=DATASET_DIR, split='validation')

    train_data = train_pascal_07.concatenate(valid_train_07).concatenate(train_pascal_12).concatenate(valid_train_12)

    test_data = tfds.load('voc', data_dir=DATASET_DIR, split='test')

    number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
    print("학습 데이터 개수", number_train)
    number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
    print("테스트 데이터 개수:", number_test)
    optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)

else :
    from model.coco_loss import total_loss
    train_data = tfds.load('coco/2017', data_dir=DATASET_DIR, split='train')
    train_data = train_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
    train_data = train_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))

    test_data = tfds.load('coco/2017', data_dir=DATASET_DIR, split='validation')
    test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
    test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))





    # number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
    number_train = 117266
    print("학습 데이터 개수", number_train)
    # number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
    number_test = 4952
    print("테스트 데이터 개수:", number_test)



    optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)
    images = []
    labels = []
    bboxes =[]

buff = train_data.take(1000)
a =list(buff.as_numpy_iterator())

for i in a:
    images = i['image']
    bboxes = i['objects']['bbox']
    labels = i['objects']['label']
    print(labels, bboxes)



images=[]
bboxes=[]
labels=[]
a = []
buff = []


