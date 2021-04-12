import tensorflow_datasets as tfds
from utils.priors import *
import argparse
import time
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from model.model_builder import model_build
from metrics import f1score, precision, recall , cross_entropy, localization


parser = argparse.ArgumentParser()

parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=1)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=200)
parser.add_argument("--image_size",     type=int,   help="모델 입력 이미지 크기 설정", default=512)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.001)
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름", default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,   help="텐서보드 저장 경로", default='tensorboard')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='B2')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='voc')
parser.add_argument("--pretrain_mode",  type=bool,  help="저장되어 있는 가중치 로드", default=False)

MODEL_INPUT_SIZE = {
    'B0': 512,
    'B1': 576,
    'B2': 640,
    'B3': 704,
    'B4': 768,
    'B5': 832,
    'B6': 896,
    'B7': 960
}

args = parser.parse_args()
BATCH_SIZE = args.batch_size
EPOCHS = args.epoch
# IMAGE_SIZE = [args.image_size, args.image_size]
base_lr = args.lr
SAVE_MODEL_NAME = args.model_name
DATASET_DIR = args.dataset_dir
CHECKPOINT_DIR = args.checkpoint_dir
TENSORBOARD_DIR = args.tensorboard_dir
MODEL_NAME = args.backbone_model
TRAIN_MODE = args.train_dataset
CONTINUE_TRAINING = args.pretrain_mode
IMAGE_SIZE = [MODEL_INPUT_SIZE[MODEL_NAME], MODEL_INPUT_SIZE[MODEL_NAME]]
print("입력 이미지 크기 : ", IMAGE_SIZE)


os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if TRAIN_MODE == 'voc':
    num_classes = 21
else:
    num_classes = 81

# TODO https://www.tensorflow.org/datasets/api_docs/python/tfds/testing/mock_data VOC+COCO 무작위 데이터 생성

iou_threshold = 0.5
center_variance = 0.1
size_variance = 0.2


## test ok
# specs = [
#             Spec(64, 8, BoxSizes(51, 123), [2]),  # 0.1
#             Spec(32, 16, BoxSizes(123, 189), [2]),  # 0.24
#             Spec(16, 32, BoxSizes(189, 256), [2, 3]),  # 0.37
#             Spec(8, 64, BoxSizes(256, 323), [2, 3]),  # 0.5
#             Spec(4, 128, BoxSizes(323, 389), [2]),  # 0.63
#             Spec(2, 256, BoxSizes(389, 461), [2]),  # 0.76
#             Spec(1, 512, BoxSizes(461, 538), [2]),  # 0.9
#         ]


""" 0408 test
{'aeroplane': 0.883235057374783,
 'bicycle': 0.873452960761183,
 'bird': 0.8355587097199219,
 'boat': 0.7985908363112868,
 'bottle': 0.6096242308587395,
 'bus': 0.850391878651511,
 'car': 0.877372139861527,
 'cat': 0.873653518498321,
 'chair': 0.680718161177171,
 'cow': 0.8131513275288116,
 'diningtable': 0.7358944506308274,
 'dog': 0.8564532217146804,
 'horse': 0.8819337612866569,
 'motorbike': 0.8767863612438392,
 'person': 0.8469665526542779,
 'pottedplant': 0.6518036318412295,
 'sheep': 0.8102691940574812,
 'sofa': 0.7508058777838824,
 'train': 0.8942437806005586,
 'tvmonitor': 0.7973264964656708}
mAP결과: 0.809911607451118
"""
# specs = [
#             Spec(int(IMAGE_SIZE[0]/16), int(IMAGE_SIZE[0]/32),
#                  BoxSizes(int(IMAGE_SIZE[0]*0.1), int(IMAGE_SIZE[0]*0.24)), [2, 3]),  # 0.2
#             Spec(int(IMAGE_SIZE[0]/32), int(IMAGE_SIZE[0]/16),
#                  BoxSizes(int(IMAGE_SIZE[0]*0.24), int(IMAGE_SIZE[0]*0.37)), [2, 3]),  # 0.37
#             Spec(int(IMAGE_SIZE[0]/64), int(IMAGE_SIZE[0]/8),
#                  BoxSizes(int(IMAGE_SIZE[0]*0.5), int(IMAGE_SIZE[0]*0.63)), [2, 3]),  # 0.54
#             Spec(int(IMAGE_SIZE[0]/128), int(IMAGE_SIZE[0]/4),
#                  BoxSizes(int(IMAGE_SIZE[0]*0.63), int(IMAGE_SIZE[0]*0.76)), [2]),  # 0.71
#             Spec(int(IMAGE_SIZE[0] / 256), int(IMAGE_SIZE[0]/2),
#                  BoxSizes(int(IMAGE_SIZE[0] * 0.76), int(IMAGE_SIZE[0] * 0.9)), [2]) # 0.88 / 0.95
#         ]

specs = [
            Spec(int(IMAGE_SIZE[0]/16), int(IMAGE_SIZE[0]/32),
                 BoxSizes(int(IMAGE_SIZE[0]*0.1), int(IMAGE_SIZE[0]*0.24)), [2, 3]),  # 0.2
            Spec(int(IMAGE_SIZE[0]/32), int(IMAGE_SIZE[0]/16),
                 BoxSizes(int(IMAGE_SIZE[0]*0.24), int(IMAGE_SIZE[0]*0.37)), [2, 3]),  # 0.37
            Spec(int(IMAGE_SIZE[0]/64), int(IMAGE_SIZE[0]/8),
                 BoxSizes(int(IMAGE_SIZE[0]*0.45), int(IMAGE_SIZE[0]*0.58)), [2, 3]),  # 0.54
            Spec(int(IMAGE_SIZE[0]/128), int(IMAGE_SIZE[0]/4),
                 BoxSizes(int(IMAGE_SIZE[0]*0.6), int(IMAGE_SIZE[0]*0.76)), [2]),  # 0.71
            Spec(int(IMAGE_SIZE[0] / 256), int(IMAGE_SIZE[0]/2),
                 BoxSizes(int(IMAGE_SIZE[0] * 0.76), int(IMAGE_SIZE[0] * 0.9)), [2]) # 0.88 / 0.95
        ]


print(specs)

priors = create_priors_boxes(specs, IMAGE_SIZE[0])
target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

if TRAIN_MODE == 'voc':
    from model.pascal_loss import total_loss
    from preprocessing import pascal_prepare_dataset
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
    # optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr, momentum=0.9)

    training_dataset = pascal_prepare_dataset(train_data, IMAGE_SIZE, BATCH_SIZE,
                                              target_transform, TRAIN_MODE, train=True)
    validation_dataset = pascal_prepare_dataset(test_data, IMAGE_SIZE, BATCH_SIZE,
                                                target_transform, TRAIN_MODE, train=False)

else :
    from model.coco_loss import total_loss
    from preprocessing import coco_prepare_dataset

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
    training_dataset = coco_prepare_dataset(train_data, IMAGE_SIZE, BATCH_SIZE,
                                              target_transform, TRAIN_MODE, train=True)
    validation_dataset = coco_prepare_dataset(test_data, IMAGE_SIZE, BATCH_SIZE,
                                                target_transform, TRAIN_MODE, train=False)


print("백본 EfficientNet{0} .".format(MODEL_NAME))
model = model_build(TRAIN_MODE, MODEL_NAME, image_size=IMAGE_SIZE)

if CONTINUE_TRAINING is True:
    model.load_weights(CHECKPOINT_DIR + '0217_main' + '.h5')

steps_per_epoch = number_train // BATCH_SIZE
validation_steps = number_test // BATCH_SIZE
print("학습 배치 개수:", steps_per_epoch)
print("검증 배치 개수:", validation_steps)
model.summary()


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint(CHECKPOINT_DIR + SAVE_MODEL_NAME + '.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, write_graph=True, write_images=True)

from callbacks import MyCallback
testCallBack = MyCallback('test', TENSORBOARD_DIR)



model.compile(
    optimizer=optimizer,
    loss=total_loss,
    metrics=[f1score, precision, recall, cross_entropy, localization]
)
    # metrics=[precision, recall, f1score])

history = model.fit(training_dataset,
                    validation_data=validation_dataset,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    epochs=EPOCHS,
                    callbacks=[reduce_lr, checkpoint, tensorboard, testCallBack])

