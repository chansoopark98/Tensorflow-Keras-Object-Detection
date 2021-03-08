import tensorflow_datasets as tfds
from utils.priors import *
import os
from preprocessing import pascal_prepare_dataset
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from model.model_builder import ssd
from tensorflow.keras.utils import plot_model
from calc_flops import get_flops
import argparse
import time



parser = argparse.ArgumentParser()

parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=32)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=100)
parser.add_argument("--image_size",     type=int,   help="모델 입력 이미지 크기 설정", default=384)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.001)
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름", default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='B0')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='coco')
parser.add_argument("--pretrain_mode",  type=bool,  help="저장되어 있는 가중치 로드", default=False)

args = parser.parse_args()
BATCH_SIZE = args.batch_size
EPOCHS = args.epoch
IMAGE_SIZE = [args.image_size, args.image_size]
base_lr = args.lr
SAVE_MODEL_NAME = args.model_name
DATASET_DIR = args.dataset_dir
checkpoint_filepath = args.checkpoint_dir
MODEL_NAME = args.backbone_model
TRAIN_MODE = args.train_dataset
CONTINUE_TRAINING = args.pretrain_mode



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


