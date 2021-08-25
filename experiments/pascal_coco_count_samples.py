import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import argparse
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='../datasets/')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='coco')

args = parser.parse_args()

TRAIN_MODE = args.train_dataset
DATASET_DIR = args.dataset_dir

if TRAIN_MODE == 'voc':
    train_pascal_12 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='train')
    valid_train_12 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='validation')
    train_pascal_07 = tfds.load("voc", data_dir=DATASET_DIR, split='train')
    valid_train_07 = tfds.load("voc", data_dir=DATASET_DIR, split='validation')
    train_data = train_pascal_07.concatenate(train_pascal_12)

else:
    train_data = tfds.load('coco/2017', data_dir=DATASET_DIR, split='train')
    train_data = train_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
    train_data = train_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))

    test_data = tfds.load('coco/2017', data_dir=DATASET_DIR, split='validation')
    test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
    test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))
    train_data = train_data.concatenate(test_data)

total_labels =[]

train_data = train_data.take(1000)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

def get_cmap(n, name='tab20'):
    return plt.cm.get_cmap(name, n)



for sample in train_data:
    img = sample['image'].numpy()
    # w = img.shape[1]
    # h = img.shape[0]

    label = sample['objects']['label'].numpy()
    bbox = sample['objects']['bbox'].numpy()[:, [1, 0, 3, 2]]

    for i in range(len(bbox)):
        xmin, ymin, xmax, ymax = bbox[i]

        # area = np.pi * (15 *label[i])**2

        w = xmax-xmin
        h = ymax-ymin

        # plt.scatter(x=w, y=h, cmap=get_cmap(label[i]), alpha=0.7)
        ax.scatter(w, h, label[i], c=label[i], alpha=0.5, cmap='jet')
        # ax.scatter(xs, ys, zs, marker=m)



# plt.legend()
# plt.grid(True)
# plt.title('VOC labels')
# plt.savefig('./voc_labels.png', dpi=600)
# plt.show()

ax.set_xlabel('Width')
ax.set_ylabel('Height')
ax.set_zlabel('Class')
plt.title('COCO labels')
plt.savefig('./coco_labels.png', dpi=1000)
plt.show()


# with open('../pascal_labels.txt') as f:
#     CLASSES = f.read().splitlines()
#     for i in range(20):
#         print(str(CLASSES[i]) + ' : ' + str(total_labels.count(i)))
#
#
# """
# pascal voc classes count
# 1285
# 1208
# 1820
# 1397
# 2116
# 909
# 4008
# 1616
# 4338
# 1058
# 1057
# 2079
# 1156
# 1141
# 15576
# 1724
# 1347
# 1211
# 984
# 1193
# """
#
#
def plot_labels(labels, save_dir=''):
    # plot dataset labels
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes

    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_xlabel('classes')
    ax[1].scatter(b[0], b[1], c=hist2d(b[0], b[1], 90), cmap='jet')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[2].scatter(b[2], b[3], c=hist2d(b[2], b[3], 90), cmap='jet')
    ax[2].set_xlabel('width')
    ax[2].set_ylabel('height')
    plt.savefig(Path(save_dir) / 'labels.png', dpi=200)
    plt.close()