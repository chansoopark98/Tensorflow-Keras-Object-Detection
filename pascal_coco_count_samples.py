import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='voc')

args = parser.parse_args()

TRAIN_MODE = args.train_dataset
DATASET_DIR = args.dataset_dir

if TRAIN_MODE == 'voc':
    train_pascal_12 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='train')
    valid_train_12 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='validation')
    train_pascal_07 = tfds.load("voc", data_dir=DATASET_DIR, split='train')
    valid_train_07 = tfds.load("voc", data_dir=DATASET_DIR, split='validation')
    train_data = train_pascal_07.concatenate(valid_train_07).concatenate(train_pascal_12).concatenate(valid_train_12)

else:
    train_data = tfds.load('coco/2017', data_dir=DATASET_DIR, split='train')
    train_data = train_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
    train_data = train_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))

    test_data = tfds.load('coco/2017', data_dir=DATASET_DIR, split='validation')
    test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
    test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))
    train_data = train_data.concatenate(test_data)

total_labels =[]

for sample in train_data:
    img = sample['image'].numpy()
    w = img.shape[1]
    h = img.shape[0]

    label = sample['objects']['label'].numpy()
    bbox = sample['objects']['bbox'].numpy()[:, [1, 0, 3, 2]]

    for i in label:
        total_labels.append(i)


for i in range(20):
    print(total_labels.count(i))


"""
pascal voc classes count
1285
1208
1820
1397
2116
909
4008
1616
4338
1058
1057
2079
1156
1141
15576
1724
1347
1211
984
1193
"""