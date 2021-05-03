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

s_obj = 0
m_obj = 0
l_obj = 0
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

        if bbox_area <= 32 ** 2:
            s_obj += 1
        elif bbox_area > 32 ** 2 and bbox_area < 96 ** 2 :
            m_obj += 1
        elif bbox_area >= 96 ** 2 :
            l_obj += 1
        else:
            non_obj += 1



print("s_obj", s_obj)
print("m_obj", m_obj)
print("l_obj", l_obj)
print("total", s_obj+m_obj+l_obj)

sizes = ['small', 'medium', 'large']
value = [s_obj, m_obj, l_obj]

## 데이터
## 시각화
plt.figure(figsize=(10, 10))  ## Figure 생성 사이즈는 10 by 10
xtick_label_position = list(range(len(sizes)))  ## x축 눈금 라벨이 표시될 x좌표
plt.xticks(xtick_label_position, sizes)  ## x축 눈금 라벨 출력

## 바 차트 출력, 막대기 색깔을 초록색으로 설정
plt.bar(xtick_label_position, value, color='green')

plt.title('PASCAL VOC OBJECT SIZE', fontsize=20)  ## 타이틀 출력
plt.xlabel('Object scale')  ## x축 라벨 출력
plt.ylabel('samples')  ## y축 라벨 출력
plt.show()

