import tensorflow_datasets as tfds
from utils.priors import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='coco')


args = parser.parse_args()
DATASET_DIR = args.dataset_dir
TRAIN_MODE = args.train_dataset



if TRAIN_MODE == 'voc':
    train_pascal_12 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='train')
    valid_train_12 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='validation')

    train_pascal_07 = tfds.load("voc", data_dir=DATASET_DIR, split='train')
    valid_train_07 = tfds.load("voc", data_dir=DATASET_DIR, split='validation')

    test_data = tfds.load('voc', data_dir=DATASET_DIR, split='test')

    train_data = train_pascal_07.concatenate(train_pascal_12)
    valid_data = valid_train_07.concatenate(valid_train_12)

    number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
    print("학습 데이터 개수", number_train)

    number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()
    print("검증 데이터 개수", number_valid)

    number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
    print("테스트 데이터 개수:", number_test)


else :
    train_data = tfds.load('coco/2017', data_dir=DATASET_DIR, split='train')
    valid_data = tfds.load('coco/2017', data_dir=DATASET_DIR, split='validation')
    test_data = tfds.load('coco/2017', data_dir=DATASET_DIR, split='test')

    number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
    print("학습 데이터 개수", number_train)

    number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()
    print("검증 데이터 개수", number_valid)

    number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
    print("테스트 데이터 개수:", number_test)

