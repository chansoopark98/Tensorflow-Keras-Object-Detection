import tensorflow_datasets as tfds
# from utils.priors import *
import argparse
import time
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from callbacks import Scalar_LR
from model.model_builder import model_build
#from metrics import f1score, precision, recall , cross_entropy, localization
from metrics import CreateMetrics
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from config import *

# multi-gpu 중 하나만 실행되는 경우에 아래 주석 해제
#tf.compat.v1.disable_eager_execution()
#tf.keras.backend.clear_session()

policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
mixed_precision.set_policy(policy)
#tf.compat.v1.enable_eager_execution()
#tf.config.optimizer.set_jit(True)

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=2)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=200)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.001)
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름", default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,   help="텐서보드 저장 경로", default='tensorboard')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='B0')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='voc')
parser.add_argument("--transfer_learning",  type=bool,  help="전이 학습 처음엔 false 두번째 true", default=False)

MODEL_INPUT_SIZE = {
    'B0': 512,
    'B1': 512,
    'B2': 512,
    'B3': 512,
    'B4': 512,
    'B5': 512,
    'B6': 512,
    'B7': 512
}

args = parser.parse_args()

BATCH_SIZE = args.batch_size
EPOCHS = args.epoch
base_lr = args.lr
SAVE_MODEL_NAME = args.model_name
DATASET_DIR = args.dataset_dir
CHECKPOINT_DIR = args.checkpoint_dir
TENSORBOARD_DIR = args.tensorboard_dir
MODEL_NAME = args.backbone_model
TRAIN_MODE = args.train_dataset
TRANSFER_LEARNING = args.transfer_learning
IMAGE_SIZE = [MODEL_INPUT_SIZE[MODEL_NAME], MODEL_INPUT_SIZE[MODEL_NAME]]
print("입력 이미지 크기 : ", IMAGE_SIZE)

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if TRAIN_MODE == 'voc':
    num_classes = 21

else:
    num_classes = 81

# TODO https://www.tensorflow.org/datasets/api_docs/python/tfds/testing/mock_data VOC+COCO 무작위 데이터 생성

# for voc
specs = set_priorBox(MODEL_NAME)
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
    # from model.focal_loss import total_loss
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

testCallBack = Scalar_LR('test', TENSORBOARD_DIR)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, min_lr=1e-5, verbose=1)
checkpoint = ModelCheckpoint(CHECKPOINT_DIR + TRAIN_MODE + '_' + SAVE_MODEL_NAME + '.h5',
                             monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, write_graph=True, write_images=True)


if TRANSFER_LEARNING is False:
    polyDecay = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.01,
                                                              decay_steps=number_train // BATCH_SIZE,
                                                              end_learning_rate=0.001, power=0.5)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(polyDecay)

    model = model_build(TRAIN_MODE, MODEL_NAME, image_size=IMAGE_SIZE, backbone_trainable=False)
    callback = [checkpoint]

else:
    weight_name = '0421'
    polyDecay = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.001,
                                                              decay_steps=number_train // BATCH_SIZE,
                                                              end_learning_rate=0.0001, power=0.5)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(polyDecay)
    model = model_build(TRAIN_MODE, MODEL_NAME, image_size=IMAGE_SIZE, backbone_trainable=True)
    model.load_weights(CHECKPOINT_DIR + weight_name + '.h5')
    callback = [reduce_lr, checkpoint]

steps_per_epoch = number_train // BATCH_SIZE
validation_steps = number_test // BATCH_SIZE
print("학습 배치 개수:", steps_per_epoch)
print("검증 배치 개수:", validation_steps)

optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

model.summary()

metric = CreateMetrics(num_classes)

model.compile(
    optimizer=optimizer,
    loss=total_loss,
    metrics=[metric.precision, metric.recall, metric.cross_entropy, metric.localization]
)

history = model.fit(training_dataset,
            validation_data=validation_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=EPOCHS,
            callbacks=callback,
            )

