import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='coco')

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

obj_lower_10 = 0
obj_10 = 0
obj_20 = 0
obj_30 = 0
obj_40 = 0
obj_50 = 0
obj_60 = 0
obj_70 = 0
obj_80 = 0
obj_90 = 0
obj_100 = 0
obj_110 = 0
obj_120 = 0
obj_130 = 0
obj_140 = 0
obj_150 = 0

non_obj = 0
for sample in train_data:
    img = sample['image'].numpy()
    w = img.shape[1]
    h = img.shape[0]

    label = sample['objects']['label'].numpy()
    bbox = sample['objects']['bbox'].numpy()[:, [1, 0, 3, 2]]

    for i in range(len(bbox)):
        sample_bbox = bbox[i]
        sample_bbox[0] *= w
        sample_bbox[2] *= w
        sample_bbox[1] *= h
        sample_bbox[3] *= h

        x_size = sample_bbox[2] - sample_bbox[0]
        y_size = sample_bbox[3] - sample_bbox[1]

        bbox_area = x_size * y_size

        if bbox_area <= 10 ** 2:
            obj_lower_10 += 1
        elif bbox_area <= 20 ** 2:
            obj_10 += 1
        elif bbox_area <= 30 ** 2:
            obj_20 += 1
        elif bbox_area <= 40 ** 2:
            obj_30 += 1
        elif bbox_area <= 50 ** 2:
            obj_40 += 1
        elif bbox_area <= 60 ** 2:
            obj_50 += 1
        elif bbox_area <= 70 ** 2:
            obj_60 += 1
        elif bbox_area <= 80 ** 2:
            obj_70 += 1
        elif bbox_area <= 90 ** 2:
            obj_80 += 1
        elif bbox_area <= 100 ** 2:
            obj_90 += 1
        elif bbox_area <= 110 ** 2:
            obj_100 += 1
        elif bbox_area <= 120 ** 2:
            obj_110 += 1
        elif bbox_area <= 130 ** 2:
            obj_120 += 1
        elif bbox_area <= 140 ** 2:
            obj_130 += 1
        elif bbox_area <= 150 ** 2:
            obj_140 += 1
        elif bbox_area <= 160 ** 2:
            obj_150 += 1



print("obj_lower_10", obj_lower_10)
print("obj_10", obj_10)
print("obj_20", obj_20)
print("obj_30", obj_30)
print("obj_40", obj_40)
print("obj_50", obj_50)
print("obj_60", obj_60)
print("obj_70", obj_70)
print("obj_80", obj_80)
print("obj_90", obj_90)
print("obj_100", obj_100)
print("obj_110", obj_110)
print("obj_120", obj_120)
print("obj_130", obj_130)
print("obj_140", obj_140)
print("obj_150", obj_150)

sizes = ['obj_lower_10', 'obj_10', 'obj_20', 'obj_30', 'obj_40', 'obj_50',  'obj_60', 'obj_70', 'obj_80', 'obj_90', 'obj_100', 'obj_110', 'obj_120', 'obj_130'
         , 'obj_140', 'obj_150']
value = [obj_lower_10,
obj_10,
obj_20,
obj_30,
obj_40,
obj_50,
obj_60,
obj_70,
obj_80,
obj_90,
obj_100,
obj_110,
obj_120,
obj_130,
obj_140,
obj_150,
]

## 데이터
## 시각화
plt.figure(figsize=(20, 20))  ## Figure 생성 사이즈는 10 by 10
xtick_label_position = list(range(len(sizes)))  ## x축 눈금 라벨이 표시될 x좌표
plt.xticks(xtick_label_position, sizes)  ## x축 눈금 라벨 출력

## 바 차트 출력, 막대기 색깔을 초록색으로 설정
plt.bar(xtick_label_position, value, color='green')

plt.title('COCO OBJECT SIZE', fontsize=25)  ## 타이틀 출력
plt.xlabel('Object scale', fontsize=15)  ## x축 라벨 출력
plt.ylabel('samples', fontsize=15)  ## y축 라벨 출력
plt.show()

