import tensorflow_datasets as tfds
DATASET_DIR = './datasets/'

train_coco = tfds.load('coco/2017', data_dir=DATASET_DIR, split='train')
valid_coco = tfds.load('coco/2017', data_dir=DATASET_DIR, split='validation')
test_coco = tfds.load('coco/2017', data_dir=DATASET_DIR, split='test')

train_pascal_12, info = tfds.load('voc/2012', data_dir=DATASET_DIR, split='train', with_info=True)
valid_train_12 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='validation')

train_pascal_07 = tfds.load("voc", data_dir=DATASET_DIR, split='train')
valid_train_07 = tfds.load("voc", data_dir=DATASET_DIR, split='validation')
test_data = tfds.load('voc', data_dir=DATASET_DIR, split='test')
